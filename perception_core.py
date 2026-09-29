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
    # (lo, hi) camera heights this rig can physically have. A fitted plane outside it is
    # not a measurement of the ground (a metric depth model trained at car height reads
    # a 10 cm FPV camera as 2.5 m up) and the map is reported UNKNOWN. None keeps the
    # old behaviour: a warning outside 0.05-3.0 m, and the map is built anyway.
    plane_plausible: Optional[tuple] = None
    # (h, w) bool, True = the vehicle's own body in frame. Ground behind it is OCCLUDED,
    # not a hole: the hole rule skips cells whose ground projects onto the mask.
    ego_mask: Optional[np.ndarray] = None

    # --- optional locks: None = estimate ------------------------------------
    lock_height: Optional[float] = None
    lock_pitch: Optional[float] = None   # radians, positive = nose down
    lock_roll: Optional[float] = None    # radians, positive = right side lower

    # --- relative-depth fallback --------------------------------------------
    nominal_height: float = 0.60    # used only when depth carries no scale
    bootstrap_pitch: Optional[float] = None   # radians, nose-down positive. Seeds the AFFINE
                                    # solve on frame 1 only, before any plane has been
                                    # fitted. NOT a lock: RANSAC overrides it immediately.
    affine_depth: bool = False      # relative depth is affine, not just scaled: solve
                                    # 1/Z = a*disp + b from ground planarity (see
                                    # solve_affine_depth). Off = the old 1/disp + scale.

    # --- lens ---------------------------------------------------------------
    dist: tuple = ()                # OpenCV distortion coeffs (k1 k2 p1 p2 k3...).
                                    # Empty = the frames are already rectified, which
                                    # is true of the simulator and assumed of webcams.

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


def rover_cfg(w: int = 640, h: int = 480, hfov: float = 62.2, cam_height: float = 0.17,
              robot_radius: float = 0.20, **over) -> "CoreCfg":
    """
    CoreCfg for a small UGV whose camera sits ~6-7 inches (0.15-0.18 m) up.

    Defaults match the MEASURED rig: Raspberry Pi Camera v2 (IMX219) at 640x480 taken
    from the FULL-FOV 1640x1232 sensor mode, whose optics are 62.2 x 48.8 degrees, so
    fx = fy ~ 530. 4:3 is deliberate, not incidental: the binding constraint here is how
    many ground rows the sensor gets, and vertical FOV is what buys them - cropping to
    16:9 would throw away a third of the ground for nothing.

    WARNING: asking picamera2 for a 640x480 SENSOR mode selects a 1280x960 crop of the
    array (39 % of its width), collapsing the field of view to ~26.5 degrees and halving
    the map's width with no error anywhere. Always pin
    `sensor={"output_size": (1640, 1232)}`. See rover_agent.py.

    WHY THESE NUMBERS AND NOT THE DEFAULTS
    --------------------------------------
    Ground samples thin out as `stride * r^2 / (fy * cam_height)` (the same expression
    the hole rule uses). At 0.17 m and fy ~ 530 the spacing is 1.1 cm at 1 m, 4.4 cm at
    2 m and 18 cm at 4 m: the whole of 3-8 m lands in about a dozen pixel rows, where
    one pixel of error is a quarter-metre of range. An 8 m map at this mount height is
    not sparse, it is fiction. So:

      x_max 2.6 m    honest sensing horizon. At 0.8 m/s with 200 ms of link latency
                     and 1 m/s^2 braking the rover stops in 0.5 m - a 5x margin.
                     Everything beyond this comes from the fused map, not this frame.
      res 0.05       a 0.25 m chassis cannot be planned for on a 0.10 m grid.
      obstacle_h     0.10 m: a rover this size is stopped by what a car drives over.
      ditch_h        -0.08 m, and plane_gate_max must stay under it, so the inlier
                     band tightens from 0.10 to 0.05.
      stride 1       at stride 2 the hole rule's own sampling gate switches negative
                     obstacles off at 2.2 m - inside the driving envelope.
      plane_near*    fit the ground the rover is about to cross (1.5 m), not 5 m of
                     mostly-empty horizon.

    `obstacle_h` is deliberately 0.10 and not the 0.06 the chassis would like: with
    monocular depth the plane residual is 5-8 cm at 2 m, so a 6 cm threshold would
    fire on noise. Stereo (2.4 cm at 2 m) is what buys the lower number.
    """
    fx, fy, cx, cy = intrinsics_from_hfov(w, h, hfov)
    base = dict(
        w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy,
        x_min=0.20, x_max=2.60, y_min=-1.30, y_max=1.30, res=0.05,
        min_depth=0.10, max_depth=4.0, stride=1,
        obstacle_h=0.10, ditch_h=-0.08, max_obstacle_h=0.80,
        ditch_max_range=2.50, hole_max_range=2.60, hole_min_cells=4,
        min_cell_pts=3, geo_min_pts=2, robot_radius=robot_radius,
        plane_near_range=1.50, plane_max_range=2.50,
        plane_lower_frac=0.55, plane_fallback_frac=0.35, plane_min_pts=300,
        plane_gate=0.02, plane_gate_rel=0.01, plane_gate_max=0.05,
        plane_max_pitch_deg=45.0, plane_min_height=0.05, plane_max_height=1.0,
        nominal_height=cam_height, affine_depth=False,
        bootstrap_pitch=math.radians(12.0),   # the recommended mount tilt; seeds frame 1 only
    )
    base.update(over)
    return CoreCfg(**base)


RIG_PRESETS = {
    # hfov is a starting point; run calibrate.py for exact numbers.
    "macbook": dict(hfov=78.0),
    "phone":   dict(hfov=68.6),   # fx = 940 at 1280 px
    "sim":     dict(hfov=None),   # exact intrinsics arrive in every frame header
    # A UGV camera sits ~0.17 m up, so ground samples thin out as r^2/(fy*h): a wide
    # lens throws away the range resolution the low mount already made scarce. 60 deg
    # still spans +/-1.4 m at 2.5 m, far more than a 0.25 m chassis needs.
    "rover":   dict(hfov=60.0),
    # ESP32-S3 camera boards (OV2640/OV3660, stock ~66 deg DIAGONAL lens -> ~55 deg
    # horizontal; wide-lens variants are closer to 100). 4:3 sensor modes, so the
    # processing frame keeps 4:3 instead of stretching to 16:9.
    "esp32":   dict(hfov=56.0, aspect=4 / 3),
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


class Undistorter:
    """
    Pinhole is a lie on a real lens. `backproject_optical` assumes straight lines stay
    straight, so barrel distortion bends the ground plane upward towards the image
    edges and manufactures LETHAL cells along both sides of the path. Rectify once, up
    front, and every equation downstream becomes true again.

    Maps are built once per (size, K, dist) and reused; an empty `dist` is a no-op, so
    the simulator and any already-rectified source pay nothing.
    """

    def __init__(self, w: int, h: int, K: np.ndarray, dist: np.ndarray):
        self.w, self.h = w, h
        self.K = np.asarray(K, np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, np.float64).ravel()
        # newCameraMatrix = K keeps the intrinsics the rest of the pipeline already
        # holds; alpha is irrelevant because we are not changing K.
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.K, (w, h), cv2.CV_16SC2)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


_UNDISTORT_CACHE: dict = {}


def undistort(img: np.ndarray, cfg: CoreCfg) -> np.ndarray:
    """Rectify `img` for cfg's intrinsics + distortion. No coeffs -> returned as is."""
    if not len(cfg.dist) or not np.any(np.asarray(cfg.dist, np.float64)):
        return img
    h, w = img.shape[:2]
    key = (w, h, cfg.fx, cfg.fy, cfg.cx, cfg.cy, tuple(np.round(np.asarray(cfg.dist, float), 8)))
    u = _UNDISTORT_CACHE.get(key)
    if u is None:
        K = np.array([[cfg.fx, 0, cfg.cx], [0, cfg.fy, cfg.cy], [0, 0, 1]], np.float64)
        u = Undistorter(w, h, K, np.asarray(cfg.dist, np.float64))
        if len(_UNDISTORT_CACHE) > 4:
            _UNDISTORT_CACHE.clear()
        _UNDISTORT_CACHE[key] = u
    return u(img)


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


def solve_affine_depth(disp: np.ndarray, sem_cost: np.ndarray, cfg: CoreCfg,
                       plane: "Optional[Plane]" = None, height: Optional[float] = None,
                       stride: int = 4, iters: int = 3, min_pts: int = 400):
    """
    Recover METRIC depth from a relative Depth Anything disparity map.

    THE PROBLEM
    -----------
    Relative Depth Anything is affine-invariant in DISPARITY: true inverse depth is
    `1/Z = a*disp + b` for a scale `a` and a shift `b` the network never reports. Taking
    `Z = 1/disp` silently assumes `b == 0`; when it is not, the cloud is WARPED, not
    merely mis-scaled, and no rescaling repairs it.

    WHY FLATNESS ALONE CANNOT FIX IT  (measured, not assumed)
    ---------------------------------------------------------
    The obvious idea is to solve both unknowns from the ground being flat. For a point
    on a plane `n.P + d = 0`, with ray `r = (xn, yn, 1)` and `m = -n/d`:

        a*disp + b  =  m_x*xn + m_y*yn + m_z

    which looks like a linear homogeneous system in (a, b, m_x, m_y, m_z). It is - but
    it is RANK DEFICIENT BY CONSTRUCTION. The column multiplying `b` is all +1 and the
    column multiplying `m_z` is all -1, so `(0, 1, 0, 0, 1)` is an exact null direction:
    adding the same delta to `b` and to `m_z` changes nothing. On real data the design
    matrix has TWO vanishing singular values and the SVD returns that trivial direction.

    Physically: on coplanar points, a constant added to inverse depth is
    indistinguishable from the plane sitting at a different distance. Only the
    combination `c = m_z - b` is observable. Worse, the degeneracy survives the obvious
    remedy - imposing a known camera height `|m| = 1/h` supplies one equation for the
    two remaining unknowns, leaving a one-parameter family.

    So the shift is NOT recoverable from flat ground, with or without a tape measure.

    WHAT DOES WORK
    --------------
    Fix the plane's ORIENTATION independently and the rest follows. Given a known
    normal `n` and height `h`, `m = -n/h` is fully determined, and

        a*disp + b = m.r

    is then an ordinary two-unknown least squares over the ground pixels. Sources for
    that orientation, in increasing order of quality:

      * the previous frame's fitted plane (what `PerceptionCore` passes) - the plane
        moves slowly, so last frame's estimate is a good constraint for this one;
      * the measured mount pitch, as the bootstrap seed on the very first frame;
      * an IMU gravity vector - per-frame, assumption-free (see the UpReference plan);
      * SLAM map points, which are NOT coplanar and so determine (a, b) outright.

    CAUTION - THIS IS A FIXED POINT, MEASURED
    -----------------------------------------
    Feeding back the previous frame's plane makes the loop SELF-CONFIRMING: seeded at
    12 deg against a true 10 deg, the estimate sits at 12.79 deg for as many iterations
    as you care to run and never migrates toward the truth. The solve forces the ground
    onto whatever orientation it was handed, and RANSAC then rediscovers that same
    orientation. `a` does converge (0.693 against a true 0.700); the orientation and `b`
    do not.

    So this path is only trustworthy when the orientation comes from something that
    cannot be biased by the depth itself - an IMU gravity vector, or non-coplanar SLAM
    map points. Until one of those exists, prefer `kind="metric"`: on the same synthetic
    scene it recovers pitch and height exactly, where `1/disp` loses the plane entirely
    (h = 0.000, pitch = 0.00). That is why `rover_cfg` ships `affine_depth=False`.

    Returns `(a, b, info)`; depth is `1/(a*disp + b)`. Returns `(None, None, info)` when
    no orientation constraint is supplied or the fit is implausible - the caller then
    falls back and says so, rather than silently trusting a degenerate solve.
    """
    info: dict = {"n_pts": 0, "residual": float("nan"), "why": ""}

    if plane is None or not plane.ok:
        info["why"] = ("no plane constraint: the affine shift is unidentifiable from "
                       "coplanar points alone (see docstring)")
        return None, None, info
    h = float(height if height is not None else plane.d)
    if not (1e-3 < h < 100.0):
        info["why"] = f"implausible height constraint {h}"
        return None, None, info

    d = disp[::stride, ::stride].astype(np.float64)
    sc = sem_cost[::stride, ::stride]
    xn, yn, rows = pixel_rays(cfg, stride)
    xn, yn = xn.astype(np.float64), yn.astype(np.float64)

    m = np.isfinite(d) & (d > 1e-6)
    m &= rows >= (1.0 - cfg.plane_lower_frac) * cfg.h
    ground = m & (sc >= 0) & (sc <= GROUND_COST_MAX)
    if ground.sum() < min_pts:
        ground = m & (rows >= (1.0 - cfg.plane_fallback_frac) * cfg.h)
        info["why"] = "semantic ground too small; fell back to lower-image prior"
    if ground.sum() < min_pts:
        info["why"] = f"only {int(ground.sum())} ground candidates (< {min_pts})"
        return None, None, info

    D, XN, YN = d[ground], xn[ground], yn[ground]
    info["n_pts"] = int(D.size)

    # orientation is GIVEN, so m is fully determined and only (a, b) remain
    mv = -np.asarray(plane.n, np.float64) / h
    target = mv[0] * XN + mv[1] * YN + mv[2]

    keep = np.ones(D.size, bool)
    a = b = 0.0
    for _ in range(max(1, iters)):
        A = np.stack([D[keep], np.ones(int(keep.sum()))], axis=1)
        sol, *_ = np.linalg.lstsq(A, target[keep], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        resid = np.abs(a * D + b - target)
        cut = np.quantile(resid, 0.80)      # obstacles are outliers to a plane
        keep = resid <= max(cut, 1e-9)
        if keep.sum() < min_pts // 2:
            break

    if a <= 0:
        info["why"] = f"degenerate solve (a={a:.4g} <= 0): disparity carries no depth signal"
        return None, None, info
    inv = a * D + b
    good = inv > 1e-6
    if good.mean() < 0.5:
        info["why"] = f"only {good.mean():.0%} of ground pixels solve to positive depth"
        return None, None, info
    med = float(np.median(1.0 / inv[good]))
    if not (0.2 <= med <= cfg.max_depth):
        info["why"] = f"implausible median ground depth {med:.1f} m"
        return None, None, info

    info["residual"] = float(np.median(np.abs(a * D + b - target)))
    info["height"] = round(h, 3)
    info["pitch_deg"] = round(math.degrees(plane.pitch), 2)
    info["median_ground_depth"] = round(med, 2)
    info["source"] = plane.source
    return float(a), float(b), info


def depth_from_affine(disp: np.ndarray, a: float, b: float, cfg: CoreCfg) -> np.ndarray:
    """Apply a `solve_affine_depth` solution. Non-positive inverse depth -> 0 (invalid)."""
    inv = a * disp.astype(np.float32) + b
    out = np.zeros_like(inv, dtype=np.float32)
    ok = inv > 1e-6
    out[ok] = 1.0 / inv[ok]
    return out


def pixel_to_ground(u: float, v: float, cfg: CoreCfg, plane: Plane):
    """
    A clicked pixel -> the metric point on the GROUND it refers to, in the robot frame.

    This is how a destination is named outdoors when there is no map and no GPS: the
    operator points at a patch of ground in the live view and the ray through that
    pixel is intersected with the plane that was just fitted to the real ground. No
    localisation, no prior map, no survey - only the geometry already measured this
    frame.

    Returns `(x_forward, y_left)` in metres, or None when the ray never meets the
    ground (at or above the horizon, or the plane is not currently known).
    """
    if plane is None or not plane.ok:
        return None
    r = np.array([(float(u) - cfg.cx) / cfg.fx, (float(v) - cfg.cy) / cfg.fy, 1.0])
    nr = float(np.dot(plane.n, r))
    # plane.n points UP and plane.d > 0, so a downward ray has n.r < 0. Anything else
    # points at or above the horizon and has no ground intersection in front of us.
    if nr > -1e-4:
        return None
    t = -plane.d / nr
    if not np.isfinite(t) or t <= 0:
        return None
    P = t * r
    R = plane.R
    x = float(R[0] @ P)
    y = float(R[1] @ P)
    return x, y


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

def ego_occluded_cells(cfg: CoreCfg, plane: "Plane") -> Optional[np.ndarray]:
    """(nx, ny) bool: cells whose ground point projects onto cfg.ego_mask (or behind the
    camera), i.e. ground the vehicle's own body hides. None without a mask."""
    if cfg.ego_mask is None:
        return None
    xs = cfg.x_min + (np.arange(cfg.nx) + 0.5) * cfg.res
    ys = cfg.y_min + (np.arange(cfg.ny) + 0.5) * cfg.res
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X, Y, np.full_like(X, -plane.d)], -1) @ plane.R        # R^T (p - d z)
    z = P[..., 2]
    u = np.round(cfg.fx * P[..., 0] / np.maximum(z, 1e-6) + cfg.cx).astype(np.int64)
    v = np.round(cfg.fy * P[..., 1] / np.maximum(z, 1e-6) + cfg.cy).astype(np.int64)
    m = cfg.ego_mask
    inb = (z > 0) & (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
    occ = z <= 0
    occ[inb] = m[v[inb], u[inb]]
    return cv2.dilate(occ.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def build_costmap(X, Y, Z, sem_cost, valid, cfg: CoreCfg, cam_height: float = 0.6,
                  inflated: bool = True, occluded: Optional[np.ndarray] = None) -> np.ndarray:
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
        if occluded is not None:
            gap &= ~occluded             # hidden by our own chassis: occluded, not a hole
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
    return inflate(grid, cfg) if inflated else grid


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
    depth_kind: str = "metric"
    affine: Optional[dict] = None      # solve_affine_depth() diagnostics, when used
    raw: Optional[np.ndarray] = None   # `grid` BEFORE inflation: what a map should remember


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

    def process(self, depth: np.ndarray, sem_cost: np.ndarray, depth_is_metric: bool = True,
                depth_kind: Optional[str] = None) -> CoreResult:
        """
        `depth_kind` overrides the `depth_is_metric` flag and selects how `depth` is read:

          "metric"    : already metres (simulator, stereo, metric Depth Anything)
          "scaled"    : unscaled depth; the fitted plane rescales it to nominal_height
          "disparity" : RAW disparity from relative Depth Anything. Both the affine
                        scale AND shift are solved from ground planarity first
                        (solve_affine_depth), because 1/disp alone warps the cloud.

        The default keeps the old two-valued behaviour, so existing callers are
        unaffected.
        """
        cfg = self.cfg
        t0 = time.perf_counter()
        if depth_kind is None:
            depth_kind = "metric" if depth_is_metric else "scaled"
        if depth.shape != (cfg.h, cfg.w):
            depth = cv2.resize(depth, (cfg.w, cfg.h), interpolation=cv2.INTER_NEAREST)
        if sem_cost.shape != (cfg.h, cfg.w):
            sem_cost = cv2.resize(sem_cost.astype(np.float32), (cfg.w, cfg.h),
                                  interpolation=cv2.INTER_NEAREST)

        affine_info = None
        pre_warn = []
        if depth_kind == "disparity":
            # The affine SHIFT is unidentifiable from flat ground alone (see
            # solve_affine_depth), so the solve needs the plane's orientation from
            # somewhere else. Last frame's fit is the natural source - the plane moves
            # slowly - with the measured mount pitch seeding the very first frame.
            seed = self.planes.prev
            if (seed is None or not seed.ok) and cfg.bootstrap_pitch is not None:
                seed = Plane(n=normal_from_angles(cfg.bootstrap_pitch, 0.0),
                             d=cfg.nominal_height, confidence=0.0, ok=True, source="bootstrap")
            a, b, affine_info = solve_affine_depth(depth, sem_cost, cfg, plane=seed,
                                                   height=cfg.nominal_height)
            if a is None:
                # 1/disp is geometrically wrong but monotonic in range, so the plane
                # estimator still has something to chew on; say so loudly, never pretend.
                pre_warn.append("affine depth solve failed: " + str(affine_info.get("why", "?")))
                depth = 1.0 / np.maximum(depth.astype(np.float32), 1e-3)
                depth_kind = "scaled"
            else:
                depth = depth_from_affine(depth, a, b, cfg)
                affine_info = dict(affine_info, a=round(a, 6), b=round(b, 6))
                depth_kind = "metric"
        depth_is_metric = depth_kind == "metric"

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
        warnings = list(pre_warn)
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
                              warnings=warnings, depth_kind=depth_kind, affine=affine_info, raw=grid,
                              timing_ms=dict(backproject=(t1 - t0) * 1e3, plane=(t2 - t1) * 1e3, costmap=0.0))

        if plane.confidence < cfg.plane_low_conf:
            warnings.append(f"low ground-plane confidence {plane.confidence:.2f}")
        if cfg.plane_plausible is not None and not (cfg.plane_plausible[0] <= plane.height <= cfg.plane_plausible[1]):
            grid = np.full((cfg.nx, cfg.ny), cfg.UNKNOWN, np.uint8)
            warnings.append(f"implausible camera height {plane.height:.2f} m "
                            f"(rig {cfg.plane_plausible[0]:.2f}-{cfg.plane_plausible[1]:.2f}): map is UNKNOWN")
            return CoreResult(grid=grid, plane=plane, scale=scale, n_points=int(valid.sum()),
                              warnings=warnings, depth_kind=depth_kind, affine=affine_info, raw=grid,
                              timing_ms=dict(backproject=(t1 - t0) * 1e3, plane=(t2 - t1) * 1e3, costmap=0.0))
        if not (0.05 <= plane.height <= 3.0):
            warnings.append(f"implausible camera height {plane.height:.2f} m")

        X, Y, Z = to_ground_frame(Xc, Yc, Zc, plane)
        raw = build_costmap(X, Y, Z, sc, valid, cfg, cam_height=plane.height, inflated=False,
                            occluded=ego_occluded_cells(cfg, plane))
        grid = inflate(raw, cfg)
        t3 = time.perf_counter()
        return CoreResult(grid=grid, plane=plane, scale=scale, n_points=int(valid.sum()),
                          warnings=warnings, depth_kind=depth_kind, affine=affine_info, raw=raw,
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
      kind="affine"   : the same relative weights, but returns the RAW disparity so
                        PerceptionCore can solve BOTH affine unknowns from ground
                        planarity (solve_affine_depth). Preferred on a fixed rig: the
                        metric models are trained on car-height driving scenes and
                        hold neither scale nor shape at a 0.17 m mount.

    `is_metric` tells PerceptionCore whether to trust the scale; `is_disparity` tells
    it the array is disparity, not depth.
    """
    MODELS = {
        "metric": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "metric-indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "relative": "depth-anything/Depth-Anything-V2-Small-hf",
        "affine": "depth-anything/Depth-Anything-V2-Small-hf",
    }
    #: what PerceptionCore.process should be told about each kind's output
    DEPTH_KIND = {"metric": "metric", "metric-indoor": "metric",
                  "relative": "scaled", "affine": "disparity"}

    def __init__(self, kind: str = "metric", device: Optional[str] = None, res: int = 336):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.kind = kind
        self.is_metric = kind.startswith("metric")
        self.is_disparity = kind == "affine"
        self.depth_kind = self.DEPTH_KIND.get(kind, "metric")
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
        if not self.is_metric and not self.is_disparity:
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
