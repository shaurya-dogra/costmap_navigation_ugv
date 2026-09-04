#!/usr/bin/env python3
"""
perception_core.py - self-calibrating traversability costmap  (SIH PS 26126)
============================================================================

    camera frame  ->  metric depth (Depth Anything V2 metric)  ->  points in the
    OPTICAL frame  ->  per-frame ground-plane estimate (RANSAC)  ->  points in a
    GROUND-ALIGNED robot frame  ->  cost = max(semantic, geometry)  ->  grid

Why this module exists
----------------------
`costmap_prototype.py` trusts a measured camera height and pitch and forces the
ground to Z = 0 from those constants. On a laptop, a hand-held phone, or a rover
whose suspension moves, the height and tilt are neither known nor constant, roll
is never exactly zero, and every height threshold is then measured against the
wrong datum. The result on the MacBook webcam was a costmap that did not match
the floor in front of it.

Here the camera pose relative to the ground is a MEASUREMENT taken every frame:

  * depth is metric, so the distance from the lens to the ground is observable;
  * a RANSAC plane is fitted to ground-labelled pixels in the optical frame;
  * the plane normal gives pitch and roll, its offset gives height;
  * every point is rotated into a frame whose Z axis is the plane normal, so
    "height above ground" is literally the Z coordinate.

`--pitch/--height` survive only as optional LOCKS (rigid rig, or a cross-check
against the simulator's known mount), never as requirements.

Frames
------
optical : OpenCV camera frame, X right, Y down, Z forward (metres)
robot   : X forward (optical axis projected onto the ground), Y left, Z up,
          origin on the ground directly beneath the lens
grid    : axis 0 = X forward (row 0 nearest), axis 1 = Y left (col 0 = rightmost)

The costmap rules are the validated ones from `costmap_prototype.Costmap.build`
(semantic vote, evidence floors, positive/negative obstacle consensus,
`cost = max(...)`, UNKNOWN never free, inflation skirt), ported without the dead
code. Neural models are wrapped at the bottom of the file behind lazy imports so
everything above them runs with numpy + OpenCV alone (see
`test_perception_core.py`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

LETHAL, UNKNOWN = 254, 255


# ----------------------------------------------------------------------------
# 1. configuration
# ----------------------------------------------------------------------------

@dataclass
class CoreCfg:
    # --- image + intrinsics (pixels) -----------------------------------------
    w: int = 1280
    h: int = 720
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 640.0
    cy: float = 360.0

    # --- costmap grid, robot frame ------------------------------------------
    x_min: float = 0.3
    x_max: float = 8.0
    y_min: float = -4.0
    y_max: float = 4.0
    res: float = 0.10

    # --- depth validity ------------------------------------------------------
    min_depth: float = 0.2
    max_depth: float = 12.0
    stride: int = 2                 # use every Nth pixel; grid stats do not need more

    # --- geometry thresholds (metres relative to the fitted ground) -----------
    obstacle_h: float = 0.25        # above ground -> positive obstacle (LETHAL)
    ditch_h: float = -0.20          # below ground -> negative obstacle (LETHAL)
    max_obstacle_h: float = 2.0     # above this is overhead clearance, ignored
    ditch_max_range: float = 8.0    # do not trust negative obstacles further out
    hole_rule: bool = True          # UNKNOWN gap with measured ground beyond it = depression
    hole_min_cells: int = 4         # gap must be at least this long (0.4 m) to count
    hole_max_range: float = 9.0     # beyond this, sampling gaps appear naturally

    # --- evidence floors ----------------------------------------------------
    min_cell_pts: int = 3
    geo_min_pts: int = 2
    geo_min_frac: float = 0.20
    sem_lethal_frac: float = 0.25
    robot_radius: float = 0.35

    # --- ground-plane estimation -------------------------------------------
    plane_near_range: float = 5.0   # fit on the ground the robot is about to drive on...
    plane_max_range: float = 10.0   # ...widening to this only if the near field is sparse
    plane_lower_frac: float = 0.65  # candidates come from the lower 65 % of the image
    plane_fallback_frac: float = 0.40   # ...or the lower 40 % if semantics gives nothing
    plane_min_pts: int = 300
    plane_gate: float = 0.05        # inlier band, metres ...
    plane_gate_rel: float = 0.01    # ... plus this fraction of depth ...
    plane_gate_max: float = 0.10    # ... capped well below |ditch_h|: a surface deep
                                    # enough to be a ditch must never count as ground
    plane_iters: int = 150          # RANSAC hypotheses per frame
    plane_max_pts: int = 4000       # subsample candidates to this many
    plane_max_pitch_deg: float = 60.0   # steeper than this is a wall, not ground
    plane_max_roll_deg: float = 45.0    # no rig rolls more than this; beyond it the fit is a wall
    plane_min_height: float = 0.03
    plane_max_height: float = 5.0
    plane_ema: float = 0.5          # blend of new estimate per frame
    plane_jump_deg: float = 8.0     # bigger change than this needs strong support
    plane_jump_m: float = 0.15
    plane_jump_conf: float = 0.60   # ...namely at least this inlier ratio
    plane_hold_frames: int = 5      # hold the last plane this long, then relock
    plane_low_conf: float = 0.30    # below this the estimate is flagged

    # --- optional locks: None = estimate ------------------------------------
    lock_height: Optional[float] = None
    lock_pitch: Optional[float] = None   # radians, positive = nose down
    lock_roll: Optional[float] = None    # radians, positive = right side lower

    # --- relative-depth fallback --------------------------------------------
    nominal_height: float = 0.60    # used only when depth carries no scale

    LETHAL: int = LETHAL
    UNKNOWN: int = UNKNOWN

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.res))

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / self.res))


def intrinsics_from_hfov(w: int, h: int, hfov_deg: float):
    """Square-pixel pinhole intrinsics from a horizontal field of view."""
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return fx, fx, w / 2.0, h / 2.0


def intrinsics_from_vfov(w: int, h: int, vfov_deg: float):
    """Three.js style: PerspectiveCamera.fov is the VERTICAL field of view."""
    fy = (h / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    return fy, fy, w / 2.0, h / 2.0


RIG_PRESETS = {
    # hfov is a starting point; run calibrate.py for exact numbers.
    "macbook": dict(hfov=78.0),
    "phone":   dict(hfov=68.6),   # fx = 940 at 1280 px
    "sim":     dict(hfov=None),   # exact intrinsics arrive in every frame header
}


# ----------------------------------------------------------------------------
# 2. semantics -> cost  (keyword table matched against ADE20K label names)
# ----------------------------------------------------------------------------

SEMANTIC_COST = [
    (("road", "sidewalk", "path", "runway"),                       0),
    (("floor", "carpet", "rug", "mat", "land", "field"),          20),
    (("grass", "dirt track"),                                     40),
    (("earth", "sand", "hill"),                                   80),
    (("water", "river", "sea", "lake", "swimming", "waterfall",
      "fountain"),                                               254),   # flat hazard: always lethal
    (("tree", "palm", "plant", "flower", "rock", "stone", "mountain",
      "building", "house", "skyscraper", "hovel", "tower", "wall",
      "fence", "railing", "bannister", "pole", "column",
      "streetlight", "traffic light", "signboard", "person", "animal",
      "car", "truck", "bus", "van", "minibike", "bicycle", "boat",
      "ship", "airplane", "bed", "chair", "sofa", "table", "desk",
      "wardrobe", "cabinet", "shelf", "armchair", "seat", "door",
      "bench", "stairs", "stairway", "step", "box", "barrel", "tent",
      "bridge", "pier", "ashcan", "sculpture", "grandstand", "booth",
      "tank", "cradle", "pot", "vase", "basket", "bag", "plaything"), 250),   # TALL lethal, see below
    (("sky", "ceiling"),                                          -1),   # ignore
]
DEFAULT_SEMANTIC_COST = 100      # unrecognised: uncertain, neither free nor lethal
GROUND_COST_MAX = 80             # sem_cost <= this counts as "ground" for the plane fit
TALL_LETHAL = 250                # label says "something with height is here"
TALL_DEMOTED = 150               # ...but the geometry measured flat ground: high cost, not lethal

# WHY TWO KINDS OF LETHAL LABEL
# A wall, a tree, a car or a person is lethal AND has height, so a correct label
# is always confirmed by the height channel. A label like that on a cell whose
# points all lie within a few centimetres of the fitted ground is therefore a
# mislabel (a flat grey floor read as "wall" is the classic case), and treating
# it as lethal blocks drivable ground. Such cells are demoted to TALL_DEMOTED:
# still expensive, so the planner avoids them when it can, never free.
# Water has no height. It cannot be verified by geometry, so it stays 254.


def build_cost_lut(id2label) -> np.ndarray:
    """One cost per class id, derived from the model's own label names."""
    n = len(id2label)
    lut = np.full(n, DEFAULT_SEMANTIC_COST, dtype=np.int16)
    for i in range(n):
        name = str(id2label[i]).lower()
        for keys, cost in SEMANTIC_COST:
            if any(k in name for k in keys):
                lut[i] = cost
                break
    return lut


# ----------------------------------------------------------------------------
# 3. geometry: pixels -> optical-frame points
# ----------------------------------------------------------------------------

_RAY_CACHE: dict = {}


def pixel_rays(cfg: CoreCfg, stride: int):
    """Normalised ray directions (xn, yn) for the strided pixel lattice, cached."""
    key = (cfg.w, cfg.h, cfg.fx, cfg.fy, cfg.cx, cfg.cy, stride)
    hit = _RAY_CACHE.get(key)
    if hit is None:
        u = np.arange(0, cfg.w, stride, dtype=np.float32)
        v = np.arange(0, cfg.h, stride, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        xn = (uu - cfg.cx) / cfg.fx
        yn = (vv - cfg.cy) / cfg.fy
        hit = (xn, yn, vv)
        if len(_RAY_CACHE) > 8:
            _RAY_CACHE.clear()
        _RAY_CACHE[key] = hit
    return hit


def backproject_optical(depth: np.ndarray, cfg: CoreCfg, stride: int):
    """
    depth (H, W) metres along the optical axis -> Xc, Yc, Zc arrays on the strided
    lattice, plus the pixel row of every sample (for the lower-image prior).
    """
    xn, yn, rows = pixel_rays(cfg, stride)
    z = depth[::stride, ::stride].astype(np.float32)
    return xn * z, yn * z, z, rows


# ----------------------------------------------------------------------------
# 4. the ground plane
# ----------------------------------------------------------------------------

def normal_from_angles(pitch: float, roll: float) -> np.ndarray:
    """
    Unit 'up' vector expressed in the OPTICAL frame for a camera pitched down by
    `pitch` and rolled by `roll` (positive = right side lower).
    """
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array([sr * cp, -cr * cp, -sp], dtype=np.float64)


def angles_from_normal(n: np.ndarray):
    """Inverse of normal_from_angles -> (pitch, roll) in radians."""
    n = n / np.linalg.norm(n)
    pitch = math.atan2(-n[2], math.hypot(n[0], n[1]))
    roll = math.atan2(n[0], -n[1])
    return pitch, roll


def rotation_from_normal(n: np.ndarray) -> np.ndarray:
    """
    3x3 matrix R with P_robot = R @ P_optical:
      row 0 = forward (optical axis projected onto the ground plane)
      row 1 = left
      row 2 = up (the plane normal)
    """
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    f = z - np.dot(z, n) * n
    nf = np.linalg.norm(f)
    if nf < 1e-6:                      # camera looking straight down: pick image-up
        f = np.array([0.0, -1.0, 0.0]) - np.dot([0.0, -1.0, 0.0], n) * n
        nf = np.linalg.norm(f)
    f /= nf
    l = np.cross(n, f)
    return np.stack([f, l, n]).astype(np.float64)


@dataclass
class Plane:
    n: np.ndarray                      # unit normal in the optical frame, points UP
    d: float                           # n . p + d = 0  ->  d = camera height
    confidence: float = 0.0            # inlier ratio of the last fit
    ok: bool = False                   # a usable estimate exists
    source: str = "none"               # "fit" | "held" | "locked" | "none"
    n_candidates: int = 0
    n_inliers: int = 0

    @property
    def height(self) -> float:
        return float(self.d)

    @property
    def pitch(self) -> float:
        return angles_from_normal(self.n)[0]

    @property
    def roll(self) -> float:
        return angles_from_normal(self.n)[1]

    @property
    def R(self) -> np.ndarray:
        return rotation_from_normal(self.n)

    def as_dict(self) -> dict:
        return dict(height=round(self.height, 3), pitch_deg=round(math.degrees(self.pitch), 2),
                    roll_deg=round(math.degrees(self.roll), 2), confidence=round(self.confidence, 3),
                    ok=bool(self.ok), source=self.source, inliers=int(self.n_inliers),
                    candidates=int(self.n_candidates))


def _fit_plane_lsq(P: np.ndarray):
    """Least-squares plane through points P (N,3) -> unit normal, d (n.p + d = 0)."""
    c = P.mean(axis=0)
    Q = P - c
    # smallest singular vector of the 3x3 scatter matrix
    _, _, vt = np.linalg.svd(Q.T @ Q)
    n = vt[-1]
    d = -float(np.dot(n, c))
    return n, d


def _orient_up(n: np.ndarray, d: float):
    """Flip so the camera (origin) is on the positive side: d > 0 means 'ground below'."""
    if d < 0:
        return -n, -d
    return n, d


class GroundPlaneEstimator:
    """
    Per-frame RANSAC ground plane in the optical frame, with temporal hold.

    The inlier band is a FIXED small distance (plus a depth-proportional term for
    sensor noise), never a data-driven MAD. At a step-down the ground is bimodal
    and a data-driven band widens to swallow both surfaces, tilting the plane
    through the step so the drop measures zero height. A fixed band below
    |ditch_h| cannot do that: anything deep enough to be a ditch is by
    definition too deep to be ground.
    """

    def __init__(self, cfg: CoreCfg, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.prev: Optional[Plane] = None
        self.reject_streak = 0

    def reset(self):
        self.prev = None
        self.reject_streak = 0

    # -- candidate selection --------------------------------------------------
    def _candidates(self, Xc, Yc, Zc, sem_cost, valid, rows):
        cfg = self.cfg
        h = cfg.h
        lower = rows >= (1.0 - cfg.plane_lower_frac) * h
        ground_lbl = (sem_cost >= 0) & (sem_cost <= GROUND_COST_MAX)
        # Near field first. The plane is the datum every height threshold is
        # measured against, so it must describe the ground the robot is about to
        # drive on, not whichever surface fills the most pixels. Approaching a
        # kerb drop the lower surface beyond the lip grows until it outvotes the
        # road under the robot; restricting to the near field makes that
        # impossible and costs nothing on flat ground.
        for rng in (cfg.plane_near_range, cfg.plane_max_range):
            near = valid & (Zc < rng)
            m = near & lower & ground_lbl
            if m.sum() >= cfg.plane_min_pts:
                return m
        for rng in (cfg.plane_near_range, cfg.plane_max_range):
            m = valid & (Zc < rng) & (rows >= (1.0 - cfg.plane_fallback_frac) * h)
            if m.sum() >= cfg.plane_min_pts:
                return m
        return m

    def _gate(self, Z):
        cfg = self.cfg
        return np.minimum(cfg.plane_gate + cfg.plane_gate_rel * Z, cfg.plane_gate_max)

    # -- one RANSAC round ----------------------------------------------------
    def _ransac(self, P: np.ndarray, Z: np.ndarray, seed_plane: Optional[Plane]):
        cfg = self.cfg
        N = len(P)
        gate = self._gate(Z)                                    # (N,)
        K = cfg.plane_iters
        idx = self.rng.integers(0, N, size=(K, 3))
        A, B, C = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
        n = np.cross(B - A, C - A)                              # (K,3)
        norm = np.linalg.norm(n, axis=1)
        good = norm > 1e-9
        n = n[good] / norm[good, None]
        d = -np.einsum("ij,ij->i", n, A[good])
        # orient every hypothesis so d > 0 (camera above ground)
        flip = d < 0
        n[flip] *= -1
        d[flip] *= -1
        if seed_plane is not None and seed_plane.ok:
            n = np.vstack([seed_plane.n[None, :], n])
            d = np.concatenate([[seed_plane.d], d])
        # plausibility: normal must point roughly toward image-up (rejects walls)
        # and the camera must be at a sane height.
        max_p = math.radians(cfg.plane_max_pitch_deg)
        up_ok = (-n[:, 1]) >= math.cos(max_p)
        roll_ok = np.abs(np.arctan2(n[:, 0], -n[:, 1])) <= math.radians(cfg.plane_max_roll_deg)
        up_ok &= roll_ok
        h_ok = (d >= cfg.plane_min_height) & (d <= cfg.plane_max_height)
        keep = up_ok & h_ok
        if not keep.any():
            return None
        n, d = n[keep], d[keep]
        resid = np.abs(P @ n.T + d[None, :])                    # (N, K')
        inl = resid <= gate[:, None]
        # near points carry more information about the ground under the robot
        wgt = (1.0 / np.maximum(Z, 0.5))[:, None]
        score = (inl * wgt).sum(axis=0)
        best = int(np.argmax(score))
        mask = inl[:, best]
        return mask

    def estimate(self, Xc, Yc, Zc, sem_cost, valid, rows) -> Plane:
        cfg = self.cfg

        # ---- fully locked rig: nothing to estimate ---------------------------
        if cfg.lock_height is not None and cfg.lock_pitch is not None:
            n = normal_from_angles(cfg.lock_pitch, cfg.lock_roll or 0.0)
            pl = Plane(n=n, d=float(cfg.lock_height), confidence=1.0, ok=True, source="locked")
            self.prev = pl
            return pl

        m = self._candidates(Xc, Yc, Zc, sem_cost, valid, rows)
        n_cand = int(m.sum())
        if n_cand < cfg.plane_min_pts:
            return self._hold("too few ground candidates", n_cand)

        P = np.stack([Xc[m], Yc[m], Zc[m]], axis=1).astype(np.float64)
        Z = Zc[m].astype(np.float64)
        if len(P) > cfg.plane_max_pts:
            sel = self.rng.choice(len(P), cfg.plane_max_pts, replace=False)
            P, Z = P[sel], Z[sel]

        mask = self._ransac(P, Z, self.prev)
        if mask is None or mask.sum() < max(30, 0.05 * len(P)):
            return self._hold("no plausible plane", n_cand)

        # refine: two least-squares passes on the inliers with the same gate
        n, d = _fit_plane_lsq(P[mask])
        n, d = _orient_up(n, d)
        for _ in range(2):
            resid = np.abs(P @ n + d)
            mask = resid <= self._gate(Z)
            if mask.sum() < 30:
                break
            n, d = _fit_plane_lsq(P[mask])
            n, d = _orient_up(n, d)

        conf = float(mask.mean())
        pitch, roll = angles_from_normal(n)
        if (abs(pitch) > math.radians(cfg.plane_max_pitch_deg)
                or abs(roll) > math.radians(cfg.plane_max_roll_deg)
                or not (cfg.plane_min_height <= d <= cfg.plane_max_height)):
            return self._hold("refined plane implausible", n_cand)

        # ---- partial locks -------------------------------------------------
        if cfg.lock_pitch is not None or cfg.lock_roll is not None:
            p0, r0 = angles_from_normal(n)
            n = normal_from_angles(cfg.lock_pitch if cfg.lock_pitch is not None else p0,
                                   cfg.lock_roll if cfg.lock_roll is not None else r0)
        if cfg.lock_height is not None:
            d = float(cfg.lock_height)

        new = Plane(n=n, d=float(d), confidence=conf, ok=True, source="fit",
                    n_candidates=n_cand, n_inliers=int(mask.sum()))

        # ---- temporal: jump gate + EMA ---------------------------------------
        prev = self.prev
        if prev is not None and prev.ok:
            ang = math.degrees(math.acos(float(np.clip(np.dot(prev.n, new.n), -1, 1))))
            jump = ang > cfg.plane_jump_deg or abs(new.d - prev.d) > cfg.plane_jump_m
            if jump and new.confidence < cfg.plane_jump_conf and self.reject_streak < cfg.plane_hold_frames:
                self.reject_streak += 1
                held = Plane(n=prev.n, d=prev.d, confidence=prev.confidence * 0.8, ok=True,
                             source="held", n_candidates=n_cand, n_inliers=prev.n_inliers)
                self.prev = held
                return held
            a = cfg.plane_ema if not jump else 1.0
            nb = a * new.n + (1 - a) * prev.n
            nb /= np.linalg.norm(nb)
            new = Plane(n=nb, d=a * new.d + (1 - a) * prev.d, confidence=new.confidence,
                        ok=True, source="fit", n_candidates=n_cand, n_inliers=new.n_inliers)
        self.reject_streak = 0
        self.prev = new
        return new

    def _hold(self, why: str, n_cand: int) -> Plane:
        prev = self.prev
        if prev is not None and prev.ok and self.reject_streak < cfg_hold(self.cfg):
            self.reject_streak += 1
            held = Plane(n=prev.n, d=prev.d, confidence=prev.confidence * 0.7, ok=True,
                         source="held", n_candidates=n_cand, n_inliers=prev.n_inliers)
            self.prev = held
            return held
        self.reject_streak += 1
        return Plane(n=np.array([0.0, -1.0, 0.0]), d=0.0, confidence=0.0, ok=False,
                     source="none", n_candidates=n_cand)


def cfg_hold(cfg: CoreCfg) -> int:
    # held planes decay; after this many frames without support we admit we are lost
    return cfg.plane_hold_frames * 2


def to_ground_frame(Xc, Yc, Zc, plane: Plane):
    """Optical points -> robot frame (X fwd, Y left, Z above ground)."""
    R = plane.R
    X = R[0, 0] * Xc + R[0, 1] * Yc + R[0, 2] * Zc
    Y = R[1, 0] * Xc + R[1, 1] * Yc + R[1, 2] * Zc
    Z = R[2, 0] * Xc + R[2, 1] * Yc + R[2, 2] * Zc + plane.d
    return X, Y, Z


# ----------------------------------------------------------------------------
# 5. the costmap
# ----------------------------------------------------------------------------

def build_costmap(X, Y, Z, sem_cost, valid, cfg: CoreCfg, cam_height: float = 0.6) -> np.ndarray:
    """
    Robot-frame points -> (nx, ny) uint8 grid.

      semantic  : a VOTE. A cell is LETHAL when >= sem_lethal_frac of its points
                  (and >= geo_min_pts of them) carry a lethal label; otherwise its
                  cost is the mean of the NON-lethal points. Averaging in either
                  direction was measured to be wrong (see costmap_prototype).
      geometry  : positive obstacle when enough points sit above obstacle_h,
                  negative obstacle (ditch) when enough sit below ditch_h,
                  each needing geo_min_pts AND geo_min_frac of the cell's points.
      fusion    : cost = max(semantic, geometry). Under-sampled cells -> UNKNOWN.
    """
    nx, ny = cfg.nx, cfg.ny
    n = nx * ny

    Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()
    sc = sem_cost.ravel().astype(np.float32)
    ok = valid.ravel() & np.isfinite(Zf) & (sc >= 0) & (Zf < cfg.max_obstacle_h)

    ix = np.floor((Xf - cfg.x_min) / cfg.res).astype(np.int32)
    iy = np.floor((Yf - cfg.y_min) / cfg.res).astype(np.int32)
    ok &= (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    idx = (ix[ok] * ny + iy[ok])
    z = Zf[ok]
    s = sc[ok]
    xr = Xf[ok]

    count = np.bincount(idx, minlength=n).astype(np.float32)
    seen = count >= max(1, cfg.min_cell_pts)

    # ---- semantic vote ----------------------------------------------------
    flat_leth = s >= cfg.LETHAL                       # water etc.: unconditional
    tall_leth = (s >= TALL_LETHAL) & (s < cfg.LETHAL)  # wall/tree/person: needs height
    leth = flat_leth | tall_leth
    vote_min = np.maximum(cfg.sem_lethal_frac * count, cfg.geo_min_pts)
    flat_n = np.bincount(idx[flat_leth], minlength=n).astype(np.float32)
    tall_n = np.bincount(idx[tall_leth], minlength=n).astype(np.float32)
    nl_n = np.bincount(idx[~leth], minlength=n).astype(np.float32)
    nl_sum = np.bincount(idx[~leth], weights=s[~leth], minlength=n).astype(np.float32)
    # is the cell measurably flat ground? (all points near the plane, enough of them)
    high_n = np.bincount(idx[z >= 0.5 * cfg.obstacle_h], minlength=n)
    low_n = np.bincount(idx[z <= 0.5 * cfg.ditch_h], minlength=n)
    flat_cell = seen & (high_n == 0) & (low_n == 0)
    sem_flat_lethal = seen & (flat_n >= vote_min)
    sem_tall_vote = seen & (tall_n >= vote_min)
    sem_tall_lethal = sem_tall_vote & ~flat_cell
    sem_demoted = sem_tall_vote & flat_cell
    benign = np.where(nl_n > 0, nl_sum / np.maximum(nl_n, 1.0), 0.0)
    semantic = np.where(sem_flat_lethal | sem_tall_lethal, float(cfg.LETHAL),
                        np.where(sem_demoted, float(TALL_DEMOTED), np.where(seen, benign, 0.0)))

    # ---- geometry consensus ----------------------------------------------
    pos = z > cfg.obstacle_h
    neg = (z < cfg.ditch_h) & (xr <= cfg.ditch_max_range)
    pos_n = np.bincount(idx[pos], minlength=n).astype(np.float32)
    neg_n = np.bincount(idx[neg], minlength=n).astype(np.float32)
    min_sup = np.clip(cfg.geo_min_frac * count, cfg.geo_min_pts, 6.0)
    geo_lethal = seen & (count >= cfg.geo_min_pts) & ((pos_n >= min_sup) | (neg_n >= min_sup))

    cost = np.where(geo_lethal, float(cfg.LETHAL), semantic)
    cost = np.where(seen, cost, float(cfg.UNKNOWN))
    grid = cost.reshape(nx, ny).astype(np.uint8)

    # ---- the hole rule -----------------------------------------------------
    # A downward-looking camera sees continuous ground. A run of cells with NO
    # measurement at all, with measured ground both before AND beyond it along
    # the viewing direction, is not "no information": the surface there dipped
    # out of sight, or it would have been measured. That is exactly a trench or
    # a kerb drop - invisible to pixels, its floor hidden by its own lip.
    # "Expensive but passable" UNKNOWN let the planner cross a 1.4 m trench for
    # the price of a 4 m detour, so the rover drove into it.
    #   * count >= 1 (not min_cell_pts): sparse sampling leaves 1-2 points per
    #     cell, a real occlusion leaves none.
    #   * a run whose nearest measured cell BEHIND it is lethal is the shadow of
    #     a positive obstacle, not a hole: left UNKNOWN.
    #   * runs shorter than hole_min_cells are ignored; nothing beyond
    #     hole_max_range, where sampling gaps appear naturally.
    if cfg.hole_rule:
        any_pt = (count >= 1).reshape(nx, ny)
        leth_cell = (grid == cfg.LETHAL)
        xmax_i = int(np.clip(round((cfg.hole_max_range - cfg.x_min) / cfg.res), 0, nx))
        behind = np.maximum.accumulate(any_pt, axis=0)
        ahead = np.maximum.accumulate(any_pt[::-1], axis=0)[::-1]
        gap = (~any_pt) & behind & ahead
        gap[xmax_i:, :] = False
        if gap.any():
            # index of the nearest measured cell behind each cell (forward fill)
            # anything lethal behind the gap, in this column or its neighbours,
            # makes the gap an occlusion shadow rather than a hole
            leth_wide = cv2.dilate(leth_cell.astype(np.uint8), np.ones((1, 3), np.uint8)).astype(bool)
            shadow_of_obstacle = np.maximum.accumulate(leth_wide, axis=0)
            gap &= ~shadow_of_obstacle
            # A run only counts if it is longer than the natural sampling gap at
            # that range: ground rows are stride * x^2 / (fy * cam_height) apart,
            # so a low camera looking far ahead leaves large, honest gaps.
            xs = cfg.x_min + (np.arange(nx) + 0.5) * cfg.res
            ys = cfg.y_min + (np.arange(ny) + 0.5) * cfg.res
            r2 = xs[:, None] ** 2 + ys[None, :] ** 2                       # range^2 per cell
            samp = cfg.stride * r2 / (max(cfg.fy, 1.0) * max(cam_height, 0.05))
            min_len = np.maximum(cfg.hole_min_cells, np.ceil(3.0 * samp / cfg.res)).astype(np.int32)
            # where the natural row spacing already exceeds two cells the ground is
            # honestly sparse: no hole verdict there, whatever the run length
            gap &= samp <= 3.0 * cfg.res
            g = gap
            start = g & ~np.vstack([np.zeros((1, ny), bool), g[:-1]])
            rid = np.cumsum(start, axis=0)
            keep = np.zeros_like(g)
            for c in np.nonzero(g.any(axis=0))[0]:
                col = g[:, c]
                ids = rid[:, c][col]
                lengths = np.bincount(ids)
                starts = np.nonzero(start[:, c])[0]
                ok_run = lengths[ids] >= min_len[starts[ids - 1], c]
                keep[col, c] = ok_run
            grid[keep] = cfg.LETHAL
    return inflate(grid, cfg)


def inflate(grid: np.ndarray, cfg: CoreCfg) -> np.ndarray:
    """Grow LETHAL by the robot radius (253) with a decaying skirt to 2 radii."""
    lethal = grid == cfg.LETHAL
    if not lethal.any():
        return grid
    dist = cv2.distanceTransform((~lethal).astype(np.uint8), cv2.DIST_L2, 3) * cfg.res
    r = cfg.robot_radius
    skirt = np.where(dist < r, 253.0,
                     np.where(dist < 2 * r, 200.0 * np.exp(-2.0 * (dist - r) / r), 0.0))
    # UNKNOWN cells inside the skirt take the skirt cost too: a never-measured
    # cell right next to a trench edge is not a cheap place to drive.
    # Only the INNER skirt (within one robot radius) applies to them; further out
    # they stay UNKNOWN, so an unmeasured cell can never read as cheap.
    unk = grid == cfg.UNKNOWN
    out = np.maximum(grid.astype(np.float32), np.where(unk, 0.0, skirt))
    out[unk] = np.where(dist[unk] < r, 253.0, float(cfg.UNKNOWN))
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# 6. the whole front-end, one call per frame
# ----------------------------------------------------------------------------

@dataclass
class CoreResult:
    grid: np.ndarray
    plane: Plane
    scale: float = 1.0
    timing_ms: dict = field(default_factory=dict)
    n_points: int = 0
    warnings: list = field(default_factory=list)


class PerceptionCore:
    """
    depth (metres or unscaled) + per-pixel semantic cost  ->  costmap.

    `depth_is_metric=False` means the depth carries no absolute scale (relative
    Depth Anything). The plane is then fitted in unscaled units and the cloud is
    rescaled so the estimated camera height equals `cfg.nominal_height`. That is
    the old `recover_scale` idea, but using the fitted plane rather than a fixed
    pitch, so tilt is still measured, only the scale is assumed.
    """

    def __init__(self, cfg: CoreCfg, seed: int = 0):
        self.cfg = cfg
        self.planes = GroundPlaneEstimator(cfg, seed=seed)

    def reset(self):
        self.planes.reset()

    def process(self, depth: np.ndarray, sem_cost: np.ndarray, depth_is_metric: bool = True) -> CoreResult:
        cfg = self.cfg
        t0 = time.perf_counter()
        if depth.shape != (cfg.h, cfg.w):
            depth = cv2.resize(depth, (cfg.w, cfg.h), interpolation=cv2.INTER_NEAREST)
        if sem_cost.shape != (cfg.h, cfg.w):
            sem_cost = cv2.resize(sem_cost.astype(np.float32), (cfg.w, cfg.h),
                                  interpolation=cv2.INTER_NEAREST)
        s = cfg.stride
        Xc, Yc, Zc, rows = backproject_optical(depth, cfg, s)
        sc = sem_cost[::s, ::s]
        if depth_is_metric:
            valid = np.isfinite(Zc) & (Zc > cfg.min_depth) & (Zc < cfg.max_depth)
        else:
            valid = np.isfinite(Zc) & (Zc > 1e-3)
        t1 = time.perf_counter()

        plane = self.planes.estimate(Xc, Yc, Zc, sc, valid, rows)
        scale = 1.0
        warnings = []
        if plane.ok and not depth_is_metric:
            scale = cfg.nominal_height / max(plane.d, 1e-3)
            Xc, Yc, Zc = Xc * scale, Yc * scale, Zc * scale
            plane = Plane(n=plane.n, d=plane.d * scale, confidence=plane.confidence, ok=True,
                          source=plane.source, n_candidates=plane.n_candidates,
                          n_inliers=plane.n_inliers)
            valid = valid & (Zc > cfg.min_depth) & (Zc < cfg.max_depth)
        t2 = time.perf_counter()

        if not plane.ok:
            grid = np.full((cfg.nx, cfg.ny), cfg.UNKNOWN, np.uint8)
            warnings.append("ground plane lost: map is UNKNOWN")
            return CoreResult(grid=grid, plane=plane, scale=scale, n_points=int(valid.sum()),
                              warnings=warnings,
                              timing_ms=dict(backproject=(t1 - t0) * 1e3, plane=(t2 - t1) * 1e3, costmap=0.0))

        if plane.confidence < cfg.plane_low_conf:
            warnings.append(f"low ground-plane confidence {plane.confidence:.2f}")
        if not (0.05 <= plane.height <= 3.0):
            warnings.append(f"implausible camera height {plane.height:.2f} m")

        X, Y, Z = to_ground_frame(Xc, Yc, Zc, plane)
        grid = build_costmap(X, Y, Z, sc, valid, cfg, cam_height=plane.height)
        t3 = time.perf_counter()
        return CoreResult(grid=grid, plane=plane, scale=scale, n_points=int(valid.sum()),
                          warnings=warnings,
                          timing_ms=dict(backproject=(t1 - t0) * 1e3, plane=(t2 - t1) * 1e3,
                                         costmap=(t3 - t2) * 1e3))


# ----------------------------------------------------------------------------
# 7. rendering
# ----------------------------------------------------------------------------

def cell_to_px(ix, iy, nx, ny, scale):
    """Grid cell -> pixel in render_costmap(): +X draws up, +Y (left) draws left."""
    return (int((ny - 1 - iy) * scale + scale // 2), int((nx - 1 - ix) * scale + scale // 2))


def px_to_cell(px, py, nx, ny, scale):
    return int(nx - 1 - py // scale), int(ny - 1 - px // scale)


def render_costmap(grid, cfg: CoreCfg, scale=5, path=None, goal=None, aim=None,
                   cmd=None, status=None, plane: Optional[Plane] = None, extra_lines=()):
    nx, ny = grid.shape
    g = grid[::-1, ::-1]
    img = np.zeros((*g.shape, 3), np.uint8)
    unk = g == cfg.UNKNOWN
    v = g[~unk].astype(np.float32) / 253.0
    img[~unk] = np.clip(np.stack([(60 + 40 * v), (220 * (1 - v)), (60 + 180 * v)], -1), 0, 255).astype(np.uint8)
    img[unk] = (70, 70, 70)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    cv2.circle(img, (w // 2, h - 4), 6, (255, 255, 255), -1)
    for m in range(1, int(cfg.x_max) + 1):
        y = int(h - (m - cfg.x_min) / cfg.res * scale)
        if 0 < y < h:
            cv2.line(img, (0, y), (w, y), (110, 110, 110) if m % 2 else (140, 140, 140), 1)
            if m % 2 == 0:
                cv2.putText(img, f"{m}m", (4, y - 4), 0, 0.4, (200, 200, 200), 1)
    if goal is not None:
        gp = cell_to_px(goal[0], goal[1], nx, ny, scale)
        cv2.drawMarker(img, gp, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
    if path:
        pts = np.array([cell_to_px(ix, iy, nx, ny, scale) for ix, iy in path], np.int32)
        cv2.polylines(img, [pts], False, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.polylines(img, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
        if aim is not None and 0 <= aim < len(path):
            cv2.circle(img, tuple(pts[aim]), 5, (255, 255, 255), -1)
            cv2.circle(img, tuple(pts[aim]), 5, (0, 0, 0), 1)
    y0 = 18
    if status:
        cv2.putText(img, status, (8, y0), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA); y0 += 18
    if cmd is not None:
        vv, om = cmd
        txt = "STOP" if vv <= 1e-6 else f"v {vv:.2f} m/s  w {om:+.2f} rad/s"
        cv2.putText(img, txt, (8, y0), 0, 0.45, (220, 220, 220), 1, cv2.LINE_AA); y0 += 16
    for line in extra_lines:
        cv2.putText(img, line, (8, y0), 0, 0.42, (200, 200, 200), 1, cv2.LINE_AA); y0 += 15
    if plane is not None:
        col = (120, 255, 120) if plane.ok and plane.confidence >= cfg.plane_low_conf else (80, 80, 255)
        txt = (f"h {plane.height:.2f}m  pitch {math.degrees(plane.pitch):+.1f}  roll "
               f"{math.degrees(plane.roll):+.1f}  conf {plane.confidence:.2f} {plane.source}")
        cv2.putText(img, txt, (8, h - 10), 0, 0.42, col, 1, cv2.LINE_AA)
    return img


def render_depth(depth: np.ndarray, max_range: float = 12.0) -> np.ndarray:
    """Turbo colourmap of metric depth: red = near, blue = far, black = no data."""
    d = np.where(depth > 0, depth, max_range)
    norm = 1.0 - np.clip(d / max_range, 0.0, 1.0)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth <= 0] = 0
    return img


# ----------------------------------------------------------------------------
# 8. neural models (lazy imports; nothing above needs torch)
# ----------------------------------------------------------------------------

def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DepthModel:
    """
    Depth Anything V2 wrapper.

      kind="metric"   : ...-Metric-Outdoor-Small-hf, output in metres (default)
      kind="relative" : ...-Small-hf, output is disparity; returns 1/disp (unscaled)

    `is_metric` tells PerceptionCore whether to trust the scale.
    """
    MODELS = {
        "metric": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "metric-indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "relative": "depth-anything/Depth-Anything-V2-Small-hf",
    }

    def __init__(self, kind: str = "metric", device: Optional[str] = None, res: int = 336):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.kind = kind
        self.is_metric = kind.startswith("metric")
        self.device = device or pick_device()
        name = self.MODELS[kind]
        self.proc = AutoImageProcessor.from_pretrained(name, size={"height": res, "width": res})
        self.model = AutoModelForDepthEstimation.from_pretrained(name).to(self.device).eval()
        self._torch = torch
        self.prev = None

    def __call__(self, rgb: np.ndarray, smooth: float = 0.0) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            x = self.proc(images=rgb, return_tensors="pt").to(self.device)
            z = self.model(**x).predicted_depth
            z = torch.nn.functional.interpolate(z[:, None].float(), size=rgb.shape[:2],
                                                mode="bilinear", align_corners=False)[0, 0]
            out = z.cpu().numpy()
        if not self.is_metric:
            out = 1.0 / np.maximum(out, 1e-3)
        if smooth > 0 and self.prev is not None and self.prev.shape == out.shape:
            out = smooth * self.prev + (1 - smooth) * out
        self.prev = out
        return out.astype(np.float32)


class SemanticModel:
    """YOLO26 ADE20K semantic segmentation -> per-pixel cost via build_cost_lut."""

    def __init__(self, weights: str, device: Optional[str] = None, imgsz: int = 640):
        from ultralytics import YOLO
        self.device = device or pick_device()
        self.model = YOLO(weights)
        self.names = self.model.names
        self.lut = build_cost_lut(self.names)
        self.imgsz = imgsz
        self.palette = _palette(len(self.names))

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        res = self.model.predict(bgr, device=self.device, imgsz=self.imgsz, verbose=False)[0]
        sm = getattr(res, "semantic_mask", None)
        if sm is None:
            return np.full((h, w), -1, np.int32)
        lab = sm.data.cpu().numpy()
        if lab.ndim == 3:
            lab = lab[0]
        if lab.shape != (h, w):
            lab = cv2.resize(lab.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return lab.astype(np.int32)

    def cost(self, labels: np.ndarray) -> np.ndarray:
        out = np.full(labels.shape, -1.0, np.float32)
        ok = labels >= 0
        out[ok] = self.lut[labels[ok]]
        return out

    def overlay(self, bgr: np.ndarray, labels: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        ok = labels >= 0
        if not ok.any():
            return bgr
        col = np.zeros_like(bgr)
        col[ok] = self.palette[labels[ok] % len(self.palette)]
        return cv2.addWeighted(bgr, 1 - alpha, col, alpha, 0)


def _palette(n: int) -> np.ndarray:
    import colorsys
    cols = []
    hue = 0.15
    for i in range(max(n, 1)):
        hue = (hue + 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85 if i % 2 == 0 else 0.95, 0.95 if i % 3 else 0.85)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))
    return np.array(cols, np.uint8)
