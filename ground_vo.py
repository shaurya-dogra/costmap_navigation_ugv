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
one pixel is a fixed number of centimetres, perspective removed. Between two consecutive
such images the robot's motion is a plain 2D rigid transform - a rotation and a
translation, already in metres.

That matters because it sidesteps the thing that makes monocular odometry hard. Ordinary
mono VO recovers translation only up to an unknown scale, and the usual fixes are an IMU,
a stereo baseline, or a loop closure. Here the scale comes from the plane distance, which
is the camera height - the same single ruler measurement the depth scale already rests on.
One assumption, used twice, rather than two independent ones.

WHAT IT IS NOT
--------------
This is ODOMETRY, not SLAM. It integrates frame-to-frame motion, so error accumulates and
never corrects: there is no loop closure and no global optimisation. Over a short traverse
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
    Metric planar odometry by registering consecutive bird's-eye views of the ground.

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
                 max_dtheta_deg: float = 30.0):
        self.bev_x, self.bev_y, self.bev_res = bev_x, bev_y, bev_res
        self.max_features, self.min_features = max_features, min_features
        self.max_dx, self.max_dtheta = max_dx, math.radians(max_dtheta_deg)
        self.w = int(round((bev_y[1] - bev_y[0]) / bev_res))
        self.h = int(round((bev_x[1] - bev_x[0]) / bev_res))
        self.prev: Optional[np.ndarray] = None
        self._H_cache: dict = {}

    def reset(self):
        self.prev = None

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
        key = (cfg.fx, cfg.fy, cfg.cx, cfg.cy, round(plane.d, 4),
               tuple(np.round(plane.n, 5)))
        H = self._H_cache.get(key)
        if H is None:
            H = self._bev_from_image(cfg, plane)
            if len(self._H_cache) > 4:
                self._H_cache.clear()
            self._H_cache[key] = H
        return cv2.warpPerspective(gray, H, (self.w, self.h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # -- one step --------------------------------------------------------------------
    def update(self, gray: np.ndarray, cfg, plane, mask: Optional[np.ndarray] = None) -> Odometry:
        """
        Register this frame's BEV against the previous one.

        `mask` is an optional image-space boolean of pixels that are genuinely ground
        (the semantic cost map thresholded). Points on an obstacle move differently from
        the ground under a rotation, so excluding them measurably tightens the fit.
        """
        bev = self.warp(gray, cfg, plane)
        if bev is None:
            self.prev = None
            return Odometry(why="no usable ground plane")

        bev_mask = None
        if mask is not None:
            mw = self.warp(mask.astype(np.uint8) * 255, cfg, plane)
            if mw is not None:
                bev_mask = (mw > 127).astype(np.uint8)

        prev, self.prev = self.prev, bev
        if prev is None:
            return Odometry(why="first frame")

        p0 = cv2.goodFeaturesToTrack(prev, self.max_features, qualityLevel=0.01,
                                     minDistance=8, mask=bev_mask, blockSize=7)
        if p0 is None or len(p0) < self.min_features:
            return Odometry(n_tracked=0 if p0 is None else len(p0),
                            why="too little ground texture to track")

        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, bev, p0, None,
                                             winSize=(21, 21), maxLevel=3,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if p1 is None:
            return Odometry(why="optical flow failed")
        good = st.ravel() == 1
        a, b = p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)
        if len(a) < self.min_features:
            return Odometry(n_tracked=len(a), why=f"only {len(a)} tracked")

        # Rigid, NOT similarity: the BEV is already metric, so any scale change would be
        # fitting noise (or a bad plane) rather than motion.
        T, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0,
                                             maxIters=2000, refineIters=20)
        if T is None:
            return Odometry(n_tracked=len(a), why="rigid fit failed")
        inliers = int(inl.sum()) if inl is not None else 0
        conf = inliers / max(len(a), 1)
        if conf < 0.30 or inliers < self.min_features:
            return Odometry(n_tracked=len(a), confidence=conf, why=f"weak fit ({inliers} inliers)")

        # T maps PREVIOUS bev pixels to CURRENT ones: how the ground appeared to move.
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

        if abs(dx) > self.max_dx or abs(dy) > self.max_dx or abs(dtheta) > self.max_dtheta:
            return Odometry(n_tracked=len(a), confidence=conf,
                            why=f"implausible step dx={dx:.2f} dy={dy:.2f} dth={math.degrees(dtheta):.0f}")

        return Odometry(dx=dx, dy=dy, dtheta=dtheta, confidence=conf,
                        n_tracked=len(a), ok=True)
