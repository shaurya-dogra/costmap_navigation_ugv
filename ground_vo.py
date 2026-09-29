#!/usr/bin/env python3
"""
ground_vo.py - metric visual odometry from the ground plane  (SIH PS 26126)
===========================================================================

    frame  ->  bird's-eye warp through the FITTED plane  ->  2D rigid fit  ->  (dx, dy, dtheta)

WHY THIS EXISTS
---------------
The global costmap needs a pose. The simulator cheats and is handed one; real hardware
has nothing - no GPS (the problem statement forbids it), no wheel encoders on this
chassis, and monocular SLAM is a heavier dependency than the job needs.

But a ground robot has a shortcut nobody else gets. `perception_core` already measures
the ground plane every frame, so the mapping from the ground to the image is a known
homography. Warp the ground region through it and you get a **metric top-down image**:
one pixel is a fixed number of centimetres, perspective removed. Between two such
images the robot's motion is a plain 2D rigid transform - a rotation and a
translation, already in metres.

That matters because it sidesteps the thing that makes monocular odometry hard. Ordinary
mono VO recovers translation only up to an unknown scale, and the usual fixes are an IMU,
a stereo baseline, or a loop closure. Here the scale comes from the plane distance, which
is the camera height - the same single ruler measurement the depth scale already rests on.
One assumption, used twice, rather than two independent ones.

WHAT IT IS NOT
--------------
This is ODOMETRY, not SLAM. Each frame is registered against a KEYFRAME (replaced every
~15 cm or ~8 deg of motion), so error accumulates per keyframe rather than per frame and
not at all while the rover stands still - but it still accumulates and never corrects: there is no loop closure and no global optimisation. Over a short traverse
it is good; over a long one it drifts. It is the right thing to have while ORB-SLAM3 is
not yet wired up, and it remains a useful cross-check afterwards.

Known failure modes, all reported rather than hidden via `Odometry.confidence`:
  * a smooth, textureless floor gives nothing to track (polished concrete, plain lino);
  * pure rotation is fine here - unlike mono SLAM, which it starves of parallax, because
    the BEV fit sees rotation directly;
  * a plane fit that is wrong makes the warp wrong, so the odometry inherits it. Motion is
    therefore only integrated while the plane's own confidence holds up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Odometry:
    """One frame-to-frame motion estimate, in the robot frame."""
    dx: float = 0.0              # metres forward
    dy: float = 0.0              # metres left
    dtheta: float = 0.0          # radians, CCW positive
    confidence: float = 0.0      # inlier ratio of the rigid fit
    n_tracked: int = 0
    ok: bool = False
    why: str = ""


class GroundVO:
    """
    Metric planar odometry by registering bird's-eye views of the ground against a keyframe.

    `bev_*` bounds the patch of ground used. Keep it NEAR: the far field is stretched by
    the warp (a pixel there covers far more ground, see the r^2/(fy*h) sampling law) and
    contributes noise, while the near field is dense and well conditioned. It should also
    stay inside the region the plane was actually fitted to.
    """

    def __init__(self,
                 bev_x: tuple = (0.25, 1.60),   # metres ahead
                 bev_y: tuple = (-0.70, 0.70),  # metres left/right
                 bev_res: float = 0.005,        # metres per BEV pixel
                 max_features: int = 400,
                 min_features: int = 25,
                 max_dx: float = 0.60,          # per-frame sanity gates
                 max_dtheta_deg: float = 30.0,
                 kf_dist: float = 0.15,         # new keyframe after this much travel ...
                 kf_angle_deg: float = 8.0,     # ... or this much turn
                 edge_margin: int = 12,         # BEV px kept clear of the view's edge
                 min_ncc: float = 0.5):         # median patch correlation of the matches

        self.bev_x, self.bev_y, self.bev_res = bev_x, bev_y, bev_res
        self.max_features, self.min_features = max_features, min_features
        self.max_dx, self.max_dtheta = max_dx, math.radians(max_dtheta_deg)
        self.w = int(round((bev_y[1] - bev_y[0]) / bev_res))
        self.h = int(round((bev_x[1] - bev_x[0]) / bev_res))
        self.kf_dist, self.kf_angle = kf_dist, math.radians(kf_angle_deg)
        self.edge_margin, self.min_ncc = edge_margin, min_ncc
        self._H_cache: dict = {}
        self.reset()

    def reset(self):
        self.prev = self.prev_mask = None
        self.kf = self.kf_mask = self.kf_pts = None
        self.rel = (0.0, 0.0, 0.0)             # robot pose at the last frame, in the keyframe

    # -- the ground <-> image homography --------------------------------------------
    def _bev_from_image(self, cfg, plane) -> np.ndarray:
        """
        3x3 taking an IMAGE pixel to a BEV pixel, built from the fitted plane.

        A ground point in the robot frame is `P_r = (X, Y, 0)`. `to_ground_frame` defines
        `P_r = R @ P_opt + (0, 0, d)`, so inverting for Z = 0:

            P_opt = R^T @ (X, Y, -d)

        and the pixel is `K @ P_opt`, homogeneous. Composing gives a homography straight
        from ground metres to pixels; the BEV axes are then just an affine relabelling of
        those metres. Both are exact - no approximation beyond the plane itself.
        """
        R = plane.R
        d = float(plane.d)
        K = np.array([[cfg.fx, 0.0, cfg.cx],
                      [0.0, cfg.fy, cfg.cy],
                      [0.0, 0.0, 1.0]], np.float64)
        # (X, Y, 1) -> P_opt, using Z = 0 on the plane
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, -d]], np.float64)
        H_ground_to_img = K @ R.T @ M

        # BEV pixel (u, v): u across +Y(left) reversed so the image reads like the costmap
        # (forward = up, left = left); v down from the far edge.
        x0, x1 = self.bev_x
        y0, y1 = self.bev_y
        # X = x1 - v*res ; Y = y1 - u*res
        # (u, v, 1) -> (X, Y, 1).  Row 0 produces X (forward), row 1 produces Y (left):
        #   X = x1 - v*res   (v runs down the image, away from the far edge)
        #   Y = y1 - u*res   (u runs right, so Y decreases)
        A = np.array([[0.0, -self.bev_res, x1],
                      [-self.bev_res, 0.0, y1],
                      [0.0, 0.0, 1.0]], np.float64)
        H_bev_to_img = H_ground_to_img @ A
        return np.linalg.inv(H_bev_to_img)

    def warp(self, gray: np.ndarray, cfg, plane) -> Optional[np.ndarray]:
        """Rectify the ground into a metric top-down image, or None if the plane is unusable."""
        if plane is None or not plane.ok or plane.d <= 1e-3:
            return None
        H = self._homography(cfg, plane)
        return cv2.warpPerspective(gray, H, (self.w, self.h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _homography(self, cfg, plane):
        key = (cfg.fx, cfg.fy, cfg.cx, cfg.cy, cfg.w, cfg.h, round(plane.d, 4),
               tuple(np.round(plane.n, 5)))
        hit = self._H_cache.get(key)
        if hit is None:
            H = self._bev_from_image(cfg, plane)
            # Where the BEV actually has image behind it, shrunk by a margin. Outside the
            # camera's view the warp is filled with black, and the edge between that fill
            # and the ground is a strong, perfectly trackable corner that NEVER MOVES in
            # the BEV - it is fixed by the camera geometry. Features there vote for zero
            # motion: on weak texture they win outright (a textureless floor then reports
            # confident standstill while the rover drives), and on good texture they bias
            # every fit toward it. Only features inside this mask are used.
            valid = cv2.warpPerspective(np.full((cfg.h, cfg.w), 255, np.uint8), H, (self.w, self.h),
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid = cv2.erode(valid, np.ones((2 * self.edge_margin + 1, 2 * self.edge_margin + 1), np.uint8))
            hit = (H, (valid > 0).astype(np.uint8))
            if len(self._H_cache) > 4:
                self._H_cache.clear()
            self._H_cache[key] = hit
        return hit[0]

    def valid_mask(self, cfg, plane) -> np.ndarray:
        """BEV pixels with real image behind them, eroded by `edge_margin`."""
        self._homography(cfg, plane)
        key = (cfg.fx, cfg.fy, cfg.cx, cfg.cy, cfg.w, cfg.h, round(plane.d, 4),
               tuple(np.round(plane.n, 5)))
        return self._H_cache[key][1]

    # -- keyframe bookkeeping -----------------------------------------------------------
    def _set_keyframe(self, bev: np.ndarray, bev_mask: Optional[np.ndarray]):
        self.kf, self.kf_mask = bev, bev_mask
        self.kf_pts = cv2.goodFeaturesToTrack(bev, self.max_features, qualityLevel=0.01,
                                              minDistance=8, mask=bev_mask, blockSize=7)
        self.rel = (0.0, 0.0, 0.0)

    def _predict(self, pts: np.ndarray, rel) -> np.ndarray:
        """Where keyframe BEV pixels `pts` should appear now if the robot sits at `rel`
        in the keyframe's robot frame. Seeds the optical flow so a keyframe that is
        several frames old is still tracked from a good starting guess."""
        x1, y1, res = self.bev_x[1], self.bev_y[1], self.bev_res
        u, v = pts[:, 0, 0], pts[:, 0, 1]
        X, Y = x1 - v * res, y1 - u * res                  # keyframe robot frame, metres
        c, s = math.cos(rel[2]), math.sin(rel[2])
        Xc = c * (X - rel[0]) + s * (Y - rel[1])           # same ground point, current frame
        Yc = -s * (X - rel[0]) + c * (Y - rel[1])
        return np.stack([(y1 - Yc) / res, (x1 - Xc) / res], -1).reshape(-1, 1, 2).astype(np.float32)

    # -- registration ------------------------------------------------------------------
    def _register(self, ref: np.ndarray, p0, bev: np.ndarray, guess=None):
        """
        Where the robot is NOW, as (dx, dy, dtheta) in the robot frame it had when `ref`
        was taken, or (None, n_tracked, confidence, why).
        """
        if p0 is None or len(p0) < self.min_features:
            return None, 0 if p0 is None else len(p0), 0.0, "too little ground texture to track"

        flags, p1 = 0, None
        if guess is not None:
            p1, flags = self._predict(p0, guess), cv2.OPTFLOW_USE_INITIAL_FLOW
        p1, st, _ = cv2.calcOpticalFlowPyrLK(ref, bev, p0, p1,
                                             winSize=(21, 21), maxLevel=3, flags=flags,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if p1 is None:
            return None, 0, 0.0, "optical flow failed"
        good = st.ravel() == 1
        a, b = p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)
        if len(a) < self.min_features:
            return None, len(a), 0.0, f"only {len(a)} tracked"

        # Rigid, NOT similarity: the BEV is already metric, so any scale change would be
        # fitting noise (or a bad plane) rather than motion.
        T, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0,
                                             maxIters=2000, refineIters=20)
        if T is None:
            return None, len(a), 0.0, "rigid fit failed"
        inliers = int(inl.sum()) if inl is not None else 0
        conf = inliers / max(len(a), 1)
        if conf < 0.30 or inliers < self.min_features:
            return None, len(a), conf, f"weak fit ({inliers} inliers)"

        # Are the matches real? On a featureless floor the only "texture" is sensor noise,
        # which is different in every frame: LK still converges (to roughly zero flow)
        # and RANSAC happily agrees, so the fit reports a confident STANDSTILL while the
        # rover drives. Texture magnitude cannot tell the two apart (noise and weak
        # gravel have the same contrast); whether the patches actually look alike can.
        ncc = self._match_ncc(ref, bev, a[inl.ravel() == 1], b[inl.ravel() == 1])
        if ncc < self.min_ncc:
            return None, len(a), conf, f"too little ground texture to track (patch NCC {ncc:.2f})"

        # T maps REFERENCE bev pixels to CURRENT ones: how the ground appeared to move.
        #
        # Rotation. BEV axes are u = (y1 - Y)/res and v = (x1 - X)/res, i.e. both are
        # NEGATED and swapped relative to (X, Y). Pushing a robot rotation of +dtheta
        # through that pair of negations leaves the image rotation with the SAME sign:
        #     u' = u*cos(dtheta) - v*sin(dtheta)
        #     v' = u*sin(dtheta) + v*cos(dtheta)
        # so the fitted angle IS dtheta. (The two sign flips cancel; taking the obvious
        # negation here inverts every turn, which is what the synthetic test caught.)
        dtheta = math.atan2(T[1, 0], T[0, 0])

        # Translation. NOT T's translation column: that is the motion of the BEV ORIGIN,
        # which under any rotation includes a lever-arm term about a point the robot is
        # not standing on. What we want is where the robot itself went, so evaluate the
        # INVERSE transform at the robot's own BEV coordinate - the robot sits at robot
        # frame (0, 0), which is bev (y1/res, x1/res) - and difference it.
        u0 = self.bev_y[1] / self.bev_res
        v0 = self.bev_x[1] / self.bev_res
        Tinv = cv2.invertAffineTransform(T)
        un = Tinv[0, 0] * u0 + Tinv[0, 1] * v0 + Tinv[0, 2]
        vn = Tinv[1, 0] * u0 + Tinv[1, 1] * v0 + Tinv[1, 2]
        # X = x1 - v*res and Y = y1 - u*res, so a step in bev is negated into metres.
        dx = float(-(vn - v0) * self.bev_res)
        dy = float(-(un - u0) * self.bev_res)
        return (dx, dy, dtheta), len(a), conf, inliers

    @staticmethod
    def _match_ncc(ref, cur, a, b, half: int = 5, max_pts: int = 40) -> float:
        """Median normalised cross-correlation of (2*half+1)^2 patches at matched points."""
        if len(a) == 0:
            return 0.0
        idx = np.linspace(0, len(a) - 1, min(len(a), max_pts)).astype(int)
        size = (2 * half + 1, 2 * half + 1)
        vals = []
        for i in idx:
            p = cv2.getRectSubPix(ref, size, (float(a[i, 0]), float(a[i, 1]))).astype(np.float32).ravel()
            q = cv2.getRectSubPix(cur, size, (float(b[i, 0]), float(b[i, 1]))).astype(np.float32).ravel()
            p -= p.mean(); q -= q.mean()
            den = math.sqrt(float(p @ p) * float(q @ q))
            vals.append(float(p @ q) / den if den > 1e-6 else 0.0)
        return float(np.median(vals))

    # -- one step --------------------------------------------------------------------
    def update(self, gray: np.ndarray, cfg, plane, mask: Optional[np.ndarray] = None) -> Odometry:
        """
        Register this frame against the current KEYFRAME and return the motion since the
        previous frame.

        Why a keyframe and not the previous frame: chaining frame-to-frame fits adds a
        fresh registration error every frame, so a rover standing still at 12 Hz random-
        walks away from where it is. Measuring every frame against one reference instead
        makes the per-frame steps TELESCOPE - their sum is exactly the latest keyframe-
        relative estimate - so error only accumulates once per keyframe, i.e. per
        `kf_dist` metres or `kf_angle` of real motion, and not at all while stationary.

        `mask` is an optional image-space boolean of pixels that are genuinely ground
        (the semantic cost map thresholded). Points on an obstacle move differently from
        the ground under a rotation, so excluding them measurably tightens the fit.
        """
        bev = self.warp(gray, cfg, plane)
        if bev is None:
            self.reset()
            return Odometry(why="no usable ground plane")

        bev_mask = self.valid_mask(cfg, plane).copy()
        if mask is not None:
            mw = self.warp(mask.astype(np.uint8) * 255, cfg, plane)
            if mw is not None:
                bev_mask &= (mw > 127).astype(np.uint8)

        if self.kf is None:
            self._set_keyframe(bev, bev_mask)
            self.prev, self.prev_mask = bev, bev_mask
            return Odometry(why="first frame")

        prev_rel = self.rel
        rel, n, conf, info = self._register(self.kf, self.kf_pts, bev, guess=prev_rel)
        if rel is not None:
            step = se2_between(prev_rel, rel)
        elif self.prev is not self.kf:
            # the keyframe has fallen out of reach (fast turn, occlusion): chain one
            # frame-to-frame step from the previous frame so the motion is not lost
            p0 = cv2.goodFeaturesToTrack(self.prev, self.max_features, qualityLevel=0.01,
                                         minDistance=8, mask=self.prev_mask, blockSize=7)
            step, n, conf, info = self._register(self.prev, p0, bev)
            rel = None
        else:
            step = None

        if step is None:
            self._set_keyframe(bev, bev_mask)
            self.prev, self.prev_mask = bev, bev_mask
            return Odometry(n_tracked=n, confidence=conf, why=info)

        dx, dy, dtheta = step
        if abs(dx) > self.max_dx or abs(dy) > self.max_dx or abs(dtheta) > self.max_dtheta:
            self._set_keyframe(bev, bev_mask)
            self.prev, self.prev_mask = bev, bev_mask
            return Odometry(n_tracked=n, confidence=conf,
                            why=f"implausible step dx={dx:.2f} dy={dy:.2f} dth={math.degrees(dtheta):.0f}")

        self.prev, self.prev_mask = bev, bev_mask
        if rel is None or math.hypot(rel[0], rel[1]) > self.kf_dist or abs(rel[2]) > self.kf_angle \
                or info < 2 * self.min_features:
            self._set_keyframe(bev, bev_mask)      # this frame becomes the new reference
        else:
            self.rel = rel
        return Odometry(dx=dx, dy=dy, dtheta=dtheta, confidence=conf, n_tracked=n, ok=True)


def se2_compose(a, b):
    """Pose b, expressed in the frame of pose a, taken into a's parent frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    th = a[2] + b[2]
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], math.atan2(math.sin(th), math.cos(th)))


def se2_between(a, b):
    """Pose b expressed in the frame of pose a (both given in the same parent frame)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    ex, ey = b[0] - a[0], b[1] - a[1]
    th = b[2] - a[2]
    return (c * ex + s * ey, -s * ex + c * ey, math.atan2(math.sin(th), math.cos(th)))
