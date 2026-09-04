#!/usr/bin/env python3
"""
Monocular traversability costmap prototype - SIH PS 26126
---------------------------------------------------------
Phone camera -> metric depth + terrain semantics + object detection
             -> ground-plane projection -> top-down cost grid.

cost(cell) = max( semantic_cost , height_cost , object_cost )

Run:  python costmap_prototype.py --source 0
      python costmap_prototype.py --source http://192.168.1.7:8080/video
      python costmap_prototype.py --source clip.mp4
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse, heapq, time
from math import hypot as _hypot
import numpy as np
import cv2
import torch

# ----------------------------------------------------------------------------
# 1. CONFIGURATION  (measure these, do not guess)
# ----------------------------------------------------------------------------

class Cfg:
    # --- camera intrinsics: get these from calibrate.py, they are per-phone ---
    fx = 940.0; fy = 940.0; cx = 640.0; cy = 360.0
    proc_w, proc_h = 1280, 720        # frames are resized to this before use

    # --- rig geometry: measure with a tape ---
    cam_height  = 0.60                # metres above the ground
    cam_pitch   = np.deg2rad(12.0)    # positive = nose down

    # --- costmap grid, robot-centric, X forward / Y left ---
    x_min, x_max = 0.5, 10.0          # metres ahead
    y_min, y_max = -4.0, 4.0          # metres left/right
    res          = 0.10               # metres per cell

    # --- geometry thresholds ---
    obstacle_h   = 0.25               # z above plane -> positive obstacle
    ditch_h      = -0.20              # z below plane -> NEGATIVE obstacle (ditch)
    rough_h      = 0.15               # within-cell height spread that costs full
    max_obstacle_h = 2.0              # z above plane to ignore (overhead ceiling/sky clearance)

    plane_near_x = 4.0                # metres: fit the ground plane only this
                                      # far ahead, so a drop-off cannot re-anchor it
    plane_max_c  = 0.15               # metres: cam_height is measured, so the
                                      # plane must pass within this of Z=0 at the robot
    plane_gate   = 0.12               # metres: inlier band for the ground fit.
                                      # Must stay below |ditch_h| so a step-down
                                      # can never be absorbed as "ground".
    plane_max_slope = 0.36            # tan(20 deg): steeper than any ground a
                                      # UGV drives on, so a steeper fit is junk
    min_cell_pts = 3                  # fewer measurements than this in a cell is
                                      # not evidence - the cell stays UNKNOWN
    geo_min_pts  = 2                  # agreeing points needed before geometry
    geo_min_frac = 0.20               # ...and that share of the cell's points
    sem_lethal_frac = 0.25            # share of a cell's points that must be
                                      # lethal-labelled for the cell to be lethal
    robot_radius = 0.35               # metres, used for inflation
    max_depth    = 15.0               # ignore anything further, error grows fast

    LETHAL, UNKNOWN = 254, 255

    # --- planner: A* over the costmap ---
    goal_x       = 8.0                # metres ahead: where we are trying to get
    goal_y       = 0.0                # metres left of centre
    plan_unknown_cost = 200.0         # UNKNOWN is expensive to cross, never blocked
    plan_cost_weight  = 6.0           # how hard cost bends the path (0 = shortest)

    # --- pure-pursuit drive command ---
    lookahead    = 1.5                # metres along the path to aim at
    stop_dist    = 0.7                # metres: plan shorter than this -> stop
    v_max        = 1.0                # m/s on a clear straight path
    w_max        = 1.5                # rad/s steering saturation
    turn_slow    = 0.6                # forward-speed de-rating per unit curvature


# ----------------------------------------------------------------------------
# 2. SEMANTIC COST TABLE  (keyword -> cost, matched against ADE20K label names)
# ----------------------------------------------------------------------------

SEMANTIC_COST = [
    (("road", "path", "sidewalk", "dirt track", "runway"),      0),
    (("floor", "land", "field", "carpet", "rug", "mat"),       20),
    (("grass",),                                               40),
    (("earth", "sand", "hill"),                                80),
    (("water", "river", "sea", "lake", "swimming"),           254),
    (("tree", "plant", "rock", "mountain", "stone",
      "building", "wall", "fence", "person", "car", "truck",
      "bed", "chair", "sofa", "table", "desk", "wardrobe",
      "cabinet", "shelf", "armchair", "seat", "door", "bench"), 254),
    (("sky", "ceiling"),                                       -1),   # -1 = ignore overhead
]
DEFAULT_SEMANTIC_COST = 100   # unrecognised class: uncertain, not free


def build_cost_lut(id2label):
    """One cost per class id, derived from the model's own label names."""
    lut = np.full(len(id2label), DEFAULT_SEMANTIC_COST, dtype=np.int16)
    for i in range(len(id2label)):
        name = id2label[i].lower()
        for keys, cost in SEMANTIC_COST:
            if any(k in name for k in keys):
                lut[i] = cost
                break
    return lut


# ----------------------------------------------------------------------------
# 3. MODELS
# ----------------------------------------------------------------------------

def pick_device():
    if torch.cuda.is_available():                       return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Absolute path to the YOLO semantics model from object segmentation
_OBJ_SEG_DIR = os.path.join(os.path.dirname(__file__), "..", "object segmentation")
_SEM_S_MODEL  = os.path.join(_OBJ_SEG_DIR, "yolo26s-sem-ade20k.pt")   # Higher-capacity ADE20K (sharp floor/wall)
_SEM_N_MODEL  = os.path.join(_OBJ_SEG_DIR, "yolo26n-sem-ade20k.pt")
_SEM_MODEL    = _SEM_S_MODEL if os.path.exists(_SEM_S_MODEL) else _SEM_N_MODEL


class Perception:
    def __init__(self, device, depth_res=336):
        import sys
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        from ultralytics import YOLO

        self.device = device
        self.half = (device == "cuda")

        # --- Depth: Depth Anything V2 Small (from v. fast depth) ---
        d = "depth-anything/Depth-Anything-V2-Small-hf"
        if depth_res:
            self.dp = AutoImageProcessor.from_pretrained(d, size={"height": depth_res, "width": depth_res})
        else:
            self.dp = AutoImageProcessor.from_pretrained(d)
        self.dm = AutoModelForDepthEstimation.from_pretrained(d).to(device).eval()

        # --- Semantics: YOLO26n ADE20K (150 classes) from object segmentation ---
        print(f"[Perception] Loading YOLO semantic model: {os.path.basename(_SEM_MODEL)}")
        self.yolo_sem = YOLO(_SEM_MODEL)
        self.cost_lut = build_cost_lut(self.yolo_sem.names)

        # Visual color palette from object segmentation colors.py
        if _OBJ_SEG_DIR not in sys.path:
            sys.path.insert(0, _OBJ_SEG_DIR)
        from colors import ColorSegregator
        self.segregator = ColorSegregator("vivid")
        self.seg_color_lut = np.array([self.segregator.get_color(i) for i in range(160)], dtype=np.uint8)

        if self.half:
            self.dm = self.dm.half()

        # Temporal stabilization and dynamic range tracking from v. fast depth
        self.prev_disp = None
        self.smoothed_min = 0.0
        self.smoothed_max = 1.0
        self.has_initial_range = False

    @torch.inference_mode()
    def depth(self, rgb):
        """
        Depth pipeline from v. fast depth:
        - 336x336 ViT patch-aligned input
        - Depth Anything V2 Small
        - Motion-adaptive temporal stabilization
        - Disparity to metric depth conversion
        """
        x = self.dp(images=rgb, return_tensors="pt").to(self.device)
        if self.half: x = {k: (v.half() if v.is_floating_point() else v)
                           for k, v in x.items()}
        z = self.dm(**x).predicted_depth
        try:
            disp = torch.nn.functional.interpolate(
                    z[:, None].float(), size=rgb.shape[:2],
                    mode="bilinear", align_corners=False)[0, 0]
        except Exception:
            disp = torch.nn.functional.interpolate(
                    z[:, None].float().cpu(), size=rgb.shape[:2],
                    mode="bilinear", align_corners=False)[0, 0]
        disp_np = disp.cpu().numpy()

        # Motion-adaptive exponential temporal filter from v. fast depth
        if self.prev_disp is not None and self.has_initial_range:
            diff = np.abs(disp_np - self.prev_disp)
            rng = max(self.smoothed_max - self.smoothed_min, 0.1)
            rel_diff = np.clip(diff / rng, 0.0, 1.0)
            # smoothstep(0.04, 0.25, rel_diff)
            t = np.clip((rel_diff - 0.04) / (0.25 - 0.04), 0.0, 1.0)
            motion_factor = t * t * (3.0 - 2.0 * t)
            alpha = 0.22 * (1.0 - motion_factor) + 1.0 * motion_factor
            disp_np = self.prev_disp * (1.0 - alpha) + disp_np * alpha

        self.prev_disp = disp_np.copy()

        # Dynamic range tracking with EMA smoothing from v. fast depth
        cur_min, cur_max = float(disp_np.min()), float(disp_np.max())
        if not self.has_initial_range:
            self.smoothed_min = cur_min
            self.smoothed_max = cur_max if cur_max > cur_min else cur_min + 1.0
            self.has_initial_range = True
        else:
            self.smoothed_min = self.smoothed_min * 0.85 + cur_min * 0.15
            self.smoothed_max = self.smoothed_max * 0.85 + cur_max * 0.15

        # Metric depth in metres: inversely proportional to disparity
        metric_depth = 3.0 / np.maximum(disp_np, 0.05)
        return metric_depth, disp_np

    def render_depth_colormap(self, disp):
        """Turbo colormap from v. fast depth: red=close (high disparity), blue=far (low disparity)."""
        rng = max(self.smoothed_max - self.smoothed_min, 1e-4)
        norm = np.clip((disp - self.smoothed_min) / rng, 0.0, 1.0)
        return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    def semantics(self, bgr):
        """
        Semantic scene parsing via YOLO26n-sem-ade20k (150 ADE20K classes).
        Returns per-pixel class id map (HxW, int32).
        """
        h, w = bgr.shape[:2]
        res = self.yolo_sem.predict(bgr, device=self.device, imgsz=640, verbose=False)[0]
        if res.semantic_mask is not None:
            sm = res.semantic_mask.data.cpu().numpy()   # (H, W) uint8
            if sm.shape != (h, w):
                sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
            return sm.astype(np.int32)
        return np.full((h, w), -1, dtype=np.int32)

    def render_semantics_overlay(self, frame, labels, alpha=0.35):
        """Blends vibrant color-coded semantics onto camera frame from object segmentation palette."""
        valid = labels >= 0
        if not valid.any():
            return frame
        colored = np.zeros_like(frame)
        colored[valid] = self.seg_color_lut[labels[valid] % len(self.seg_color_lut)]
        return cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)


# ----------------------------------------------------------------------------
# 4. GEOMETRY: pixels -> metric 3D points in the robot frame
# ----------------------------------------------------------------------------

def backproject(depth, cfg):
    """Pixel grid + depth -> 3D points in the ROBOT frame (X fwd, Y left, Z up)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))

    # optical frame (OpenCV): X right, Y down, Z forward
    Xc = (u - cfg.cx) * depth / cfg.fx
    Yc = (v - cfg.cy) * depth / cfg.fy
    Zc = depth

    # re-axis to body-aligned camera frame: forward, left, up
    xf, yl, zu = Zc, -Xc, -Yc

    # undo the camera pitch, then lift by the measured camera height
    c, s = np.cos(cfg.cam_pitch), np.sin(cfg.cam_pitch)
    X = xf * c + zu * s
    Y = yl
    Z = -xf * s + zu * c + cfg.cam_height
    return X, Y, Z, (xf, zu)


def recover_scale(raw, sem_cost, cfg):
    """
    Monocular depth carries scale error. We know the camera height, so the
    ground must sit at Z = 0. Solve for the single scalar that puts it there.
        Z = s * (-xf*sin + zu*cos) + h  ->  s = h / (xf*sin - zu*cos)
    Estimated only over near, ground-labelled pixels.
    """
    xf, zu = raw
    c, s_ = np.cos(cfg.cam_pitch), np.sin(cfg.cam_pitch)
    denom = xf * s_ - zu * c
    ground = (sem_cost >= 0) & (sem_cost <= 80) & (denom > 1e-3) & (xf < 8.0)
    if ground.sum() < 300:
        # Robust fallback: lower visual field points (zu < 0) within near range
        lower = (denom > 1e-3) & (xf > 0.4) & (xf < 4.0) & (zu < 0)
        if lower.sum() >= 200:
            ground = lower
        else:
            return 1.0
    s = np.median(cfg.cam_height / denom[ground])
    return float(np.clip(s, 0.25, 3.5))


_PLANE_STATE = {"a": 0.0, "b": 0.0, "c": 0.0, "valid": False}


def _solve_plane_3x3(Xg, Yg, Zg):
    """Solve Z = a*X + b*Y + c via 3×3 normal equations — ~10× faster than lstsq."""
    # AtA = [[ΣX², ΣXY, ΣX],
    #         [ΣXY, ΣY², ΣY],
    #         [ΣX,  ΣY,  N ]]
    SX  = Xg.sum();  SY  = Yg.sum();  SZ  = Zg.sum()
    SX2 = (Xg * Xg).sum(); SY2 = (Yg * Yg).sum()
    SXY = (Xg * Yg).sum()
    SXZ = (Xg * Zg).sum(); SYZ = (Yg * Zg).sum()
    N   = float(len(Xg))
    AtA = np.array([[SX2, SXY, SX],
                    [SXY, SY2, SY],
                    [SX,  SY,  N ]], dtype=np.float64)
    AtZ = np.array([SXZ, SYZ, SZ], dtype=np.float64)
    try:
        a, b, c = np.linalg.solve(AtA, AtZ)
    except np.linalg.LinAlgError:
        a, b, c = 0.0, 0.0, np.median(Zg)
    return float(a), float(b), float(c)


def fit_ground_plane(X, Y, Z, sem_cost, near_x=None, valid=None, max_c=None,
                     gate=0.12, max_slope=None):
    """
    Robustly fit a linear ground plane  Z = a*X + b*Y + c  using normal
    equations (3×3 system — ~10× faster than SVD/lstsq).

    Steps:
      0. Drop pixels with no usable depth, then restrict to the NEAR field
         (X < near_x) if it still has enough support.
      1. Select floor-labelled pixels (sem_cost in [0..80]).
      2. Stride-subsample to ≤1500 points for speed.
      3. Seed the plane at Z = 0 (the robot is standing on it) and re-fit a few
         times, keeping only points within a FIXED ±`gate` of the current plane.
      4. Clamp the intercept, then EMA-smooth across frames to stop jitter.

    WHY THE NEAR-FIELD RESTRICTION EXISTS
    -------------------------------------
    The plane is the datum that `obstacle_h` and `ditch_h` are measured against,
    so it has to describe the ground the robot is ABOUT TO DRIVE ON, not
    whichever surface happens to fill the most pixels.

    Fitting over the whole frame breaks exactly where it matters most. Walking
    up to a kerb drop, the lower surface beyond the lip grows until it outvotes
    the road the robot is standing on; the fit then snaps down onto it, the
    step-down measures zero height, and the negative-obstacle rule goes silent
    at the last moment. Measured on the simulator, `c` jumped from 0.00 to
    -0.45 m at 1.5 m from the lip, lethal cells collapsed from 31% to 1.5%, and
    the robot drove straight off the edge.

    Anchoring to the near field costs nothing on flat ground and makes a
    receding slope read as an obstacle rather than as new ground - conservative
    in the direction this project always chooses.

    `valid` must exclude pixels with no depth measurement. Those backproject to
    (X=0, Y=0, Z=cam_height) - a phantom surface sitting on the camera itself -
    and a quarter of a typical frame is sky. Far-range ground used to outweigh
    them by sheer leverage, which hid the problem; restricting to the near field
    does not, and the fit runs away to c = +0.44 m on ground that is exactly
    flat. Pass the mask.

    `max_c` clamps the intercept. cam_height is physically measured, so the
    ground under the robot is Z = 0 by construction and the plane must pass
    through it; a slope tilts `a`/`b` but cannot move `c`. A large `c` only ever
    means the fit has latched onto some other surface.

    `max_slope` rejects the fit outright. No ground a UGV drives on tilts more
    than ~20 deg, so a steeper solution is not a ground plane - it is the fit
    having latched onto a wall, a desk, or a person filling the frame. Measured
    on a live indoor feed the fit returned a = +1.30 (52 deg), which pushed the
    correctly-labelled floor to Zr = -1.4 m and turned flat drivable floor into
    245 lethal "ditch" cells. When the tilt is impossible the honest answer is
    the measured datum we started from: flat ground at Z = 0.

    WHY THE INLIER GATE IS A FIXED DISTANCE AND NOT A MAD MULTIPLE
    -------------------------------------------------------------
    At a step-down the ground is bimodal: road at Z=0 and, past the lip, a
    second surface 0.45 m lower. A MAD-scaled threshold sees that spread as the
    noise level, widens to match, and calls both surfaces inliers - so least
    squares splits the difference and TILTS the plane through the step. The step
    then measures ~0 m of relative height and the negative-obstacle rule goes
    silent precisely as the robot arrives at the edge (measured: `a` swinging to
    -0.49 while the drop was 1 m away).

    A fixed gate below |ditch_h| makes that impossible: anything deep enough to
    be a ditch is by definition too deep to be ground, so it is rejected as an
    outlier and the plane stays on the surface the robot is on. Seeding at Z = 0
    rather than at the least-squares fit is what makes the first gate meaningful.
    """
    global _PLANE_STATE
    gm = (sem_cost >= 0) & (sem_cost <= 80) & np.isfinite(Z)
    if valid is not None:
        gm &= valid

    # Anchor on the near field; widen only if it is too sparse to fit.
    if near_x is not None:
        near = gm & (X < near_x)
        if near.sum() >= 200:
            gm = near

    cnt = gm.sum()
    if cnt < 200:
        if _PLANE_STATE["valid"]:
            return _PLANE_STATE["a"], _PLANE_STATE["b"], _PLANE_STATE["c"]
        return 0.0, 0.0, 0.0

    Xg = X[gm].ravel().astype(np.float64)
    Yg = Y[gm].ravel().astype(np.float64)
    Zg = Z[gm].ravel().astype(np.float64)

    # Strided subsampling — deterministic, cache-friendly, no random overhead
    max_pts = 1500
    if len(Xg) > max_pts:
        step = len(Xg) // max_pts
        Xg, Yg, Zg = Xg[::step], Yg[::step], Zg[::step]

    # Seed on the physical anchor: the ground beneath the robot is Z = 0.
    a, b, c = 0.0, 0.0, 0.0
    for _ in range(4):
        resid  = Zg - (a * Xg + b * Yg + c)
        inlier = np.abs(resid) <= gate
        if inlier.sum() < 80:
            break
        na, nb, nc = _solve_plane_3x3(Xg[inlier], Yg[inlier], Zg[inlier])
        if max_c is not None:
            nc = float(np.clip(nc, -max_c, max_c))
        if max_slope is not None and (abs(na) > max_slope or abs(nb) > max_slope):
            # Not a ground plane. Keep the measured datum instead of a wall.
            na, nb, nc = 0.0, 0.0, 0.0
        converged = (abs(na - a) < 1e-4 and abs(nb - b) < 1e-4
                     and abs(nc - c) < 1e-3)
        a, b, c = na, nb, nc
        if converged:
            break

    # EMA smoothing across frames (α=0.3 → new frame contributes 30%)
    alpha = 0.30
    if _PLANE_STATE["valid"]:
        a = alpha * a + (1 - alpha) * _PLANE_STATE["a"]
        b = alpha * b + (1 - alpha) * _PLANE_STATE["b"]
        c = alpha * c + (1 - alpha) * _PLANE_STATE["c"]
    _PLANE_STATE.update({"a": a, "b": b, "c": c, "valid": True})
    return float(a), float(b), float(c)



# ----------------------------------------------------------------------------
# 5. THE COSTMAP
# ----------------------------------------------------------------------------

class Costmap:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nx = int((cfg.x_max - cfg.x_min) / cfg.res)
        self.ny = int((cfg.y_max - cfg.y_min) / cfg.res)

    def cell_index(self, X, Y):
        ix = ((X - self.cfg.x_min) / self.cfg.res).astype(np.int32)
        iy = ((Y - self.cfg.y_min) / self.cfg.res).astype(np.int32)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix, iy, ok

    def build(self, X, Y, Z, sem_cost, boxes, depth, masks=None):
        """
        Costmap cell classification — redesigned for correctness and speed.

        Design principles:
          • Work at ½ input resolution (every 2nd pixel) — the grid is only 95×80
            cells, so 230K projected points give the same per-cell statistics as
            921K points at ¼ the bincount cost.
          • Semantic segmentation is the primary signal: floor → free, wall/
            furniture → lethal.  This is accurate and cheap.
          • Geometry is a fallback: only pixels inside semantically-floor cells are
            tested for obstacle height.  Walls are already lethal; checking them
            geometrically is redundant and expensive.
          • Roughness removed: monocular depth noise (σ ≈ 4–6 cm) is larger than
            real indoor roughness, making std-dev an unreliable signal.

        Cell cost scale:
          0        – true zero (used internally, collapsed to FREE before output)
          20       – floor / drivable surface
          100–200  – reserved for future mid-cost uses
          253      – inflation skirt around lethal
          254      – LETHAL (obstacle, wall, furniture, ditch)
          255      – UNKNOWN (no valid depth data)
        """
        cfg = self.cfg
        nx, ny = self.nx, self.ny
        flat = lambda ix, iy: ix * ny + iy        # row-major: axis0=X, axis1=Y
        n = nx * ny

        # ---- half-resolution subsampling ------------------------------------
        # The grid has 7 600 cells; full-res gives 121 pts/cell — wasteful.
        # Strided slice keeps every 2nd pixel → 230 K pts, ~8 pts/cell at 1 m.
        if X.ndim == 2:
            Xs  = X[::2, ::2].ravel()
            Ys  = Y[::2, ::2].ravel()
            Zs  = Z[::2, ::2].ravel()
            sc_s = sem_cost[::2, ::2].ravel().astype(np.float32)
            d_s  = depth[::2, ::2].ravel() if hasattr(depth, 'ndim') and depth.ndim == 2 else depth.ravel()[::4]
        else:
            Xs, Ys, Zs, sc_s, d_s = X, Y, Z, sem_cost.astype(np.float32), depth

        # ---- adaptive ground plane (fast 3×3 normal equations) --------------
        # d_ok gates the fit as well as the grid: a pixel with no depth is not
        # evidence about where the ground is.
        d_ok = (d_s > 0.3) & (d_s < cfg.max_depth)
        ga, gb, gc = fit_ground_plane(
            Xs, Ys, Zs, sc_s,
            near_x=getattr(cfg, "plane_near_x", None),
            valid=d_ok,
            max_c=getattr(cfg, "plane_max_c", None),
            gate=getattr(cfg, "plane_gate", 0.12),
            max_slope=getattr(cfg, "plane_max_slope", None))
        Zr = (Zs - (ga * Xs + gb * Ys + gc)).astype(np.float32)

        # ---- valid pixel filter ---------------------------------------------
        max_h = getattr(cfg, "max_obstacle_h", 2.0)
        valid = d_ok & np.isfinite(Zs) & (Zr < max_h) & (sc_s >= 0)
        ix, iy, ok = self.cell_index(Xs, Ys)
        m    = valid & ok
        idx  = flat(ix[m], iy[m])
        z    = Zr[m]
        sc   = sc_s[m]

        # ---- per-cell accumulation ------------------------------------------
        count = np.bincount(idx, minlength=n).astype(np.float32)
        ssum  = np.bincount(idx, weights=sc, minlength=n).astype(np.float32)

        # A cell backed by one or two stray pixels is not a measurement. On a
        # live feed 544 of 1309 occupied cells held <=2 points, and because the
        # old consensus floor was 1 point, a single noisy depth sample could
        # declare a cell LETHAL - the radial speckle that made the map unusable.
        # Under-sampled cells go back to UNKNOWN: expensive, never free.
        min_pts = max(1, int(getattr(cfg, "min_cell_pts", 1)))
        seen  = count >= min_pts
        safe  = np.maximum(count, 1.0)

        # ---- channel A: SEMANTIC  (what it looks like) ----------------------
        # A VOTE, not a mean. Averaging costs inside a cell dilutes danger: a
        # cell that is 40% wall and 60% floor averages to 114 and reads as
        # merely awkward terrain, when a wall in a 10 cm cell means the cell is
        # impassable. Averaging also works the other way - on a live feed every
        # single cell came out above 50 because a few high-cost pixels dragged
        # otherwise clean floor up, so nothing was ever reported drivable.
        #
        # So: a cell is lethal when a real share of its points say lethal, and
        # otherwise its cost is the mean of only the NON-lethal points, which is
        # what actually describes the terrain there.
        s_frac = float(getattr(cfg, "sem_lethal_frac", 0.25))
        s_pts  = float(getattr(cfg, "geo_min_pts", 2))
        leth_n = np.bincount(idx[sc >= cfg.LETHAL], minlength=n).astype(np.float32)
        nl     = sc < cfg.LETHAL
        nl_n   = np.bincount(idx[nl], minlength=n).astype(np.float32)
        nl_sum = np.bincount(idx[nl], weights=sc[nl], minlength=n).astype(np.float32)

        sem_lethal = seen & (leth_n >= np.maximum(s_frac * count, s_pts))
        benign     = np.where(nl_n > 0, nl_sum / np.maximum(nl_n, 1.0), 0.0)
        semantic   = np.where(sem_lethal, float(cfg.LETHAL),
                              np.where(seen, benign, 0.0))

        # ---- channel B: GEOMETRIC (height above fitted ground plane) --------
        # Only check pixels inside cells that semantics has NOT yet classified
        # as lethal.  This cuts the bincount size roughly in half for indoor scenes.
        floor_cell = (semantic < 127) & seen   # cells semantics thinks are floor-ish
        floor_px   = m & (ix.ravel() * ny + iy.ravel() < n)
        # Rebuild mask: only project pixels that land in floor-like cells
        cell_of_px = flat(ix[m], iy[m])
        px_in_floor_cell = floor_cell[cell_of_px]

        idx_f = idx[px_in_floor_cell]
        z_f   = z[px_in_floor_cell]

        cnt_f  = np.bincount(idx_f, minlength=n).astype(np.float32)
        pos_f  = np.bincount(idx_f[z_f > cfg.obstacle_h], minlength=n).astype(np.float32)
        neg_f  = np.bincount(idx_f[z_f < cfg.ditch_h],   minlength=n).astype(np.float32)

        # Consensus: a share of the cell's points must agree, and never fewer
        # than geo_min_pts of them. The absolute floor is what matters - at 8 m
        # a single depth sample carries metre-scale error, so one point below
        # ditch_h is noise, not a ditch.
        g_pts  = float(getattr(cfg, "geo_min_pts", 2))
        g_frac = float(getattr(cfg, "geo_min_frac", 0.20))
        min_sup    = np.clip(g_frac * cnt_f, g_pts, 6.0)
        enough     = cnt_f >= g_pts
        is_obstacle = floor_cell & enough & (pos_f >= min_sup)
        is_ditch    = floor_cell & enough & (neg_f >= min_sup)

        geo_lethal  = np.where(is_obstacle | is_ditch, float(cfg.LETHAL), 0.0)

        # ---- channel C: detected object boxes (fallback when no masks) ------
        obj = np.zeros(n, np.float32)
        if (obj == 0).all() and len(boxes) > 0:
            for x1, y1, x2, y2 in boxes:
                uu = int(np.clip((x1 + x2) / 2, 0, depth.shape[1] - 1))
                vv = int(np.clip(y2, 0, depth.shape[0] - 1))
                d  = depth[vv, uu]
                if not (0.3 < d < cfg.max_depth):
                    continue
                px_, py_ = X[vv, uu], Y[vv, uu]
                jx, jy, k = self.cell_index(np.array([px_]), np.array([py_]))
                if not k[0]:
                    continue
                half = max(1, int(((x2 - x1) * d / cfg.fx) / 2 / cfg.res))
                for aa in range(-half, half + 1):
                    for bb in range(-half, half + 1):
                        q, w_ = jx[0] + aa, jy[0] + bb
                        if 0 <= q < nx and 0 <= w_ < ny:
                            obj[flat(q, w_)] = cfg.LETHAL

        # ---- FUSE -----------------------------------------------------------
        cost = np.maximum(semantic, np.maximum(geo_lethal, obj))
        cost = np.where(seen, cost, cfg.UNKNOWN)
        grid = cost.reshape(nx, ny).astype(np.uint8)

        return self.inflate(grid)



    def inflate(self, grid):
        """Grow lethal cells by the robot radius, with a decaying skirt."""
        cfg = self.cfg
        lethal = (grid >= cfg.LETHAL) & (grid != cfg.UNKNOWN)
        if not lethal.any():
            return grid
        free = (~lethal).astype(np.uint8)
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 3) * cfg.res
        r = cfg.robot_radius
        skirt = np.where(dist < r, 253,
                 np.where(dist < 2 * r, (200 * np.exp(-2.0 * (dist - r) / r)), 0))
        out = np.maximum(grid.astype(np.float32),
                         np.where(grid == cfg.UNKNOWN, 0, skirt))
        out[grid == cfg.UNKNOWN] = cfg.UNKNOWN
        return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# 6. PLANNER: A* over the costmap, then a pure-pursuit drive command
# ----------------------------------------------------------------------------

_NB = ((-1, -1, 1.4142135), (-1, 0, 1.0), (-1, 1, 1.4142135),
       ( 0, -1, 1.0),                     ( 0, 1, 1.0),
       ( 1, -1, 1.4142135), ( 1, 0, 1.0), ( 1, 1, 1.4142135))


def traversal_cost(grid, cfg):
    """
    Split a costmap into (per-cell cost, blocked mask) for the planner.

    UNKNOWN is EXPENSIVE BUT PASSABLE. This is deliberate and is the third
    safety rule of the project:
      - treat UNKNOWN as free  -> the robot drives confidently into ditches
        that simply had no depth measurement.
      - treat UNKNOWN as blocked -> the robot freezes, because the far half of
        a monocular frame is always partly unmeasured.
    Expensive-but-passable means it will cross unmapped ground only when no
    measured route exists, which is the behaviour we actually want.

    Only true LETHAL (254) cells block. The 253 inflation skirt stays passable
    at very high cost, so a tight gap is squeezed through rather than the plan
    failing outright.
    """
    cost = grid.astype(np.float32)
    blocked = (grid >= cfg.LETHAL) & (grid != cfg.UNKNOWN)
    cost[grid == cfg.UNKNOWN] = cfg.plan_unknown_cost
    return cost, blocked


def goal_cell(cfg, nx, ny):
    """Cfg.goal_x/goal_y in metres -> grid cell, clamped into the map."""
    ix = int(round((cfg.goal_x - cfg.x_min) / cfg.res))
    iy = int(round((cfg.goal_y - cfg.y_min) / cfg.res))
    return int(np.clip(ix, 0, nx - 1)), int(np.clip(iy, 0, ny - 1))


def start_cell(cfg, nx, ny):
    """
    The robot sits at X=0, Y=0, which is BEHIND the grid (x_min = 0.5 m).
    The plan therefore starts at the nearest measured row, dead centre.
    """
    iy = int(round((0.0 - cfg.y_min) / cfg.res))
    return 0, int(np.clip(iy, 0, ny - 1))


def astar(grid, cfg, start=None, goal=None):
    """
    8-connected A* over the traversability grid.

    Returns (path, reached):
      path    - list of (ix, iy) cells from start to the best node found,
                [] if the robot is already boxed in.
      reached - True only if the actual goal cell was expanded.

    Step cost is  length_in_cells * (1 + plan_cost_weight * cell_cost/255),
    so the multiplier is >= 1 everywhere and straight-line cell distance stays
    an admissible heuristic.

    When the goal is unreachable the path to the expanded cell CLOSEST to the
    goal is returned with reached=False. Graceful degradation matters more than
    a binary answer here: the robot should still make progress toward the goal
    and re-plan next frame, rather than stopping because 8 m ahead is walled.
    """
    nx, ny = grid.shape
    if start is None:
        start = start_cell(cfg, nx, ny)
    if goal is None:
        goal = goal_cell(cfg, nx, ny)

    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < nx and 0 <= sy < ny):
        return [], False

    cost, blocked = traversal_cost(grid, cfg)

    # Something LETHAL is 0.5 m dead ahead. There is no safe first step, so
    # emit no path at all and let drive_command() bring the robot to a stop.
    if blocked[sx, sy]:
        return [], False

    # per-cell entry multiplier, flattened for fast scalar indexing
    mult = (1.0 + (cfg.plan_cost_weight / 255.0) * cost).ravel().tolist()
    blk = blocked.ravel().tolist()

    n = nx * ny
    g = [float("inf")] * n
    parent = [-1] * n
    closed = bytearray(n)

    s_i = sx * ny + sy
    g_i = gx * ny + gy
    g[s_i] = 0.0

    h0 = _hypot(gx - sx, gy - sy)
    open_ = [(h0, s_i)]
    best_i, best_h = s_i, h0
    reached = False

    push, pop = heapq.heappush, heapq.heappop
    while open_:
        _, i = pop(open_)
        if closed[i]:
            continue
        closed[i] = 1

        if i == g_i:
            best_i, reached = i, True
            break

        ix, iy = divmod(i, ny)
        h = _hypot(gx - ix, gy - iy)
        if h < best_h:
            best_h, best_i = h, i

        gi = g[i]
        for dx, dy, L in _NB:
            jx = ix + dx
            jy = iy + dy
            if jx < 0 or jx >= nx or jy < 0 or jy >= ny:
                continue
            j = jx * ny + jy
            if closed[j] or blk[j]:
                continue
            ng = gi + L * mult[j]
            if ng < g[j]:
                g[j] = ng
                parent[j] = i
                push(open_, (ng + _hypot(gx - jx, gy - jy), j))

    # walk parents back from the best node reached
    path = []
    i = best_i
    while i != -1:
        path.append(divmod(i, ny))
        i = parent[i]
    path.reverse()
    return path, reached


def path_metres(path, cfg):
    """Grid cells -> (X forward, Y left) metres in the robot frame."""
    return [(cfg.x_min + ix * cfg.res, cfg.y_min + iy * cfg.res)
            for ix, iy in path]


def drive_command(path, cfg):
    """
    Pure-pursuit steering from the planned path.

    Returns (v, omega, aim_index):
      v         - m/s forward, de-rated on tight curvature and on a short plan
      omega     - rad/s, POSITIVE = turn left (toward +Y), matching the grid
      aim_index - index into `path` of the lookahead point, for drawing

    No path means no safe motion, so the command is a full stop. That is the
    only correct default: an empty path is produced exactly when the cell in
    front of the robot is lethal.
    """
    if len(path) < 2:
        return 0.0, 0.0, None

    pts = path_metres(path, cfg)

    # first point at least `lookahead` metres from the robot at (0, 0)
    aim = len(pts) - 1
    for k, (x, y) in enumerate(pts):
        if _hypot(x, y) >= cfg.lookahead:
            aim = k
            break

    x, y = pts[aim]
    d = _hypot(x, y)
    if d < 1e-3 or x <= 0.0:
        return 0.0, 0.0, aim

    # How far the plan actually gets. A path that dead-ends just ahead means
    # the way is blocked, however straight its first metre looks, so speed is
    # scaled by reach and cut to zero inside stop_dist. Without this the robot
    # drives at v_max into a wall 0.8 m away because the aim point is clear.
    ex, ey = pts[-1]
    reach = _hypot(ex, ey)
    if reach < cfg.stop_dist:
        return 0.0, 0.0, aim

    # curvature of the arc from the robot through the aim point
    kappa = 2.0 * y / (d * d)
    v = cfg.v_max / (1.0 + cfg.turn_slow * abs(kappa))
    v *= min(1.0, reach / max(cfg.lookahead, 1e-3))
    omega = float(np.clip(v * kappa, -cfg.w_max, cfg.w_max))
    return float(v), omega, aim

# ----------------------------------------------------------------------------
# 7. VISUALISATION
# ----------------------------------------------------------------------------

def cell_to_px(ix, iy, nx, ny, scale):
    """
    Grid cell -> pixel in the rendered costmap.

    render() flips BOTH axes (rows so +X forward draws upward, cols so +Y left
    draws to screen left), so this has to flip with it. Anything drawn on the
    costmap must go through here, or it will silently land mirrored.
    """
    return (int((ny - 1 - iy) * scale + scale // 2),
            int((nx - 1 - ix) * scale + scale // 2))


def render(grid, cfg, scale=5, path=None, goal=None, aim=None, cmd=None,
           reached=None):
    nx, ny = grid.shape
    g = grid[::-1, ::-1]      # rows: X forward -> up.  cols: +Y left -> screen left
    img = np.zeros((*g.shape, 3), np.uint8)
    unk = g == cfg.UNKNOWN
    val = ~unk
    v = g[val].astype(np.float32) / 253.0
    img[val] = np.clip(np.stack([(60 + 40 * v), (220 * (1 - v)),
                                 (60 + 180 * v)], -1), 0, 255).astype(np.uint8)
    img[unk] = (70, 70, 70)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    cv2.circle(img, (w // 2, h - 4), 6, (255, 255, 255), -1)
    for m in range(2, int(cfg.x_max) + 1, 2):
        y = int(h - (m - cfg.x_min) / cfg.res * scale)
        if 0 < y < h:
            cv2.line(img, (0, y), (w, y), (120, 120, 120), 1)
            cv2.putText(img, f"{m}m", (4, y - 4), 0, 0.4, (200, 200, 200), 1)

    # ---- planned path -------------------------------------------------------
    if goal is not None:
        gp = cell_to_px(goal[0], goal[1], nx, ny, scale)
        colour = (0, 255, 255) if reached else (0, 160, 255)
        cv2.drawMarker(img, gp, colour, cv2.MARKER_TILTED_CROSS, 14, 2)

    if path:
        pts = np.array([cell_to_px(ix, iy, nx, ny, scale) for ix, iy in path],
                       np.int32)
        # dark casing first so the path stays readable over magenta lethal cells
        cv2.polylines(img, [pts], False, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.polylines(img, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
        if aim is not None and 0 <= aim < len(path):
            cv2.circle(img, tuple(pts[aim]), 5, (255, 255, 255), -1)
            cv2.circle(img, tuple(pts[aim]), 5, (0, 0, 0), 1)

    pitch_deg = np.rad2deg(cfg.cam_pitch)
    h_cm = cfg.cam_height * 100.0
    cv2.putText(img, f"pitch: {pitch_deg:.0f} deg | h: {h_cm:.0f}cm", (10, h - 14),
                0, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    if cmd is not None:
        v, omega = cmd
        if v <= 1e-6:
            txt, col = "STOP", (80, 80, 255)
        else:
            turn = "left" if omega > 0.05 else ("right" if omega < -0.05 else "straight")
            txt = f"v {v:.2f} m/s  w {omega:+.2f} rad/s  {turn}"
            col = (220, 220, 220) if reached else (0, 200, 255)
        cv2.putText(img, txt, (10, 20), 0, 0.45, col, 1, cv2.LINE_AA)
        if reached is False:
            cv2.putText(img, "goal blocked - partial plan", (10, 38), 0, 0.4,
                        (0, 160, 255), 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------
# 8. MAIN LOOP
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index (e.g. 0), video file path, or stream URL")
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--record", default=None, help="path to record raw camera stream (e.g. out.mp4)")
    ap.add_argument("--profile", action="store_true", help="print per-stage millisecond timing")
    ap.add_argument("--depth-res", type=int, default=336, help="depth model input resolution (default: 336)")
    ap.add_argument("--pitch", type=float, default=None, help="camera downward pitch in degrees (default: 12.0 for rig, set 0.0 for level)")
    ap.add_argument("--height", type=float, default=None, help="camera height above ground in metres (default: 0.60, set 0.10 for floor)")
    ap.add_argument("--goal", type=float, default=None, help="goal distance straight ahead in metres (default: 8.0)")
    ap.add_argument("--no-plan", action="store_true", help="perception only, skip the A* planner")
    a = ap.parse_args()

    cfg = Cfg()
    if a.pitch is not None:
        cfg.cam_pitch = np.deg2rad(a.pitch)
    if a.height is not None:
        cfg.cam_height = a.height
    if a.goal is not None:
        cfg.goal_x = float(np.clip(a.goal, cfg.x_min + cfg.res, cfg.x_max - cfg.res))

    dev = pick_device(); print("device:", dev)
    per = Perception(dev, depth_res=a.depth_res)
    cm  = Costmap(cfg)

    # Setup interactive runtime tuning panel on costmap window
    cv2.namedWindow("costmap")
    def _noop(x): pass
    cv2.createTrackbar("pitch (deg)", "costmap", int(round(np.rad2deg(cfg.cam_pitch))), 45, _noop)
    cv2.createTrackbar("cam_h (cm)", "costmap", int(round(cfg.cam_height * 100)), 150, _noop)
    cv2.createTrackbar("obs_h (cm)", "costmap", int(round(cfg.obstacle_h * 100)), 60, _noop)
    cv2.createTrackbar("ditch_h (cm)", "costmap", int(round(abs(cfg.ditch_h) * 100)), 60, _noop)
    if not a.no_plan:
        cv2.createTrackbar("goal (m)", "costmap", int(round(cfg.goal_x)),
                           int(cfg.x_max), _noop)

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {a.source}")

    writer = None
    if a.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(a.record, fourcc, 30.0, (cfg.proc_w, cfg.proc_h))
        print(f"Recording raw stream to: {a.record}")

    i, t0, fps = 0, time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % a.every:
                continue

            frame = cv2.resize(frame, (cfg.proc_w, cfg.proc_h))
            if writer is not None:
                writer.write(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t_stage = time.time()
            depth, disp = per.depth(rgb)
            dt_depth = (time.time() - t_stage) * 1000

            t_stage = time.time()
            labels = per.semantics(frame)   # YOLO26 semantics takes BGR directly
            dt_sem = (time.time() - t_stage) * 1000

            t_stage = time.time()
            sem_cost = per.cost_lut[labels].astype(np.float32)

            # Read live interactive tuning sliders from costmap window
            tb_p = cv2.getTrackbarPos("pitch (deg)", "costmap")
            tb_h = cv2.getTrackbarPos("cam_h (cm)", "costmap")
            tb_obs = cv2.getTrackbarPos("obs_h (cm)", "costmap")
            tb_ditch = cv2.getTrackbarPos("ditch_h (cm)", "costmap")
            if tb_p >= 0:
                cfg.cam_pitch = np.deg2rad(float(tb_p))
            if tb_h > 0:
                cfg.cam_height = max(0.04, float(tb_h) / 100.0)
            if tb_obs > 0:
                cfg.obstacle_h = max(0.05, float(tb_obs) / 100.0)
            if tb_ditch > 0:
                cfg.ditch_h = -max(0.05, float(tb_ditch) / 100.0)
            if not a.no_plan:
                tb_goal = cv2.getTrackbarPos("goal (m)", "costmap")
                if tb_goal > 0:
                    cfg.goal_x = float(np.clip(tb_goal, cfg.x_min + cfg.res,
                                               cfg.x_max - cfg.res))

            X, Y, Z, raw = backproject(depth, cfg)
            s = recover_scale(raw, sem_cost, cfg)
            depth = depth * s
            X, Y, Z, _ = backproject(depth, cfg)

            grid = cm.build(X, Y, Z, sem_cost, np.zeros((0, 4)), depth)
            dt_costmap = (time.time() - t_stage) * 1000

            # ---- plan ------------------------------------------------------
            t_stage = time.time()
            path, reached, aim, cmd, goal = [], None, None, None, None
            if not a.no_plan:
                goal = goal_cell(cfg, cm.nx, cm.ny)
                path, reached = astar(grid, cfg, goal=goal)
                v, omega, aim = drive_command(path, cfg)
                cmd = (v, omega)
            dt_plan = (time.time() - t_stage) * 1000

            t_stage = time.time()
            dt = time.time() - t0; t0 = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-3))

            # Render depth using v. fast depth pipeline (Turbo colormap: red=close, blue=far)
            dv = per.render_depth_colormap(disp)

            # Render camera feed with translucent semantics overlay
            frame_vis = per.render_semantics_overlay(frame, labels, alpha=0.30)
            cv2.putText(frame_vis, f"{fps:4.1f} fps   scale x{s:.2f}", (10, 30), 0, 0.8,
                        (255, 255, 255), 2)

            top = np.hstack([cv2.resize(frame_vis, (640, 360)), cv2.resize(dv, (640, 360))])
            cv2.imshow("camera | depth", top)
            cv2.imshow("costmap", render(grid, cfg, path=path, goal=goal,
                                         aim=aim, cmd=cmd, reached=reached))
            dt_render = (time.time() - t_stage) * 1000

            if a.profile:
                dt_total = dt_depth + dt_sem + dt_costmap + dt_plan + dt_render
                print(f"[PROFILE] depth: {dt_depth:5.1f}ms | sem: {dt_sem:5.1f}ms | "
                      f"costmap: {dt_costmap:5.1f}ms | plan: {dt_plan:5.1f}ms | "
                      f"render: {dt_render:5.1f}ms | total: {dt_total:5.1f}ms "
                      f"({1000/max(dt_total, 1e-3):4.1f} fps)")

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
