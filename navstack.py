#!/usr/bin/env python3
"""
navstack.py - a Nav2-shaped navigation stack in plain Python  (SIH PS 26126)
============================================================================

Nav2 splits navigation into a GLOBAL costmap + planner (where to go, at coarse
resolution over the whole known map) and a LOCAL costmap + controller (how to
move right now, at fine resolution around the robot). This module keeps exactly
that split so a real Nav2 can be swapped in over rosbridge later; the message
shapes it emits are in `ros_msgs.py`.

    local grid (perception_core, single frame, robot frame)
        |  pose (PoseSource)
        v
    GlobalCostmap.fuse()      world frame, max-fusion, UNKNOWN never overwritten
        |
    plan_global()             A* on a coarse, boxed copy      -> world path
    (or DStarGlobalPlanner)   D* Lite: same grid/costs, search kept and repaired
        |
    carrot()                  first path point ~x_max ahead   -> local goal
        |
    local A* + pure pursuit   (costmap_prototype.astar / drive_command)
        |
    Navigator                 NO_GOAL / PLANNING / TURNING / DRIVING / BLOCKED / LOST / ARRIVED

Pose
----
`PoseSource` is the seam where the team's visual SLAM (part 2 of the problem
statement) plugs in. Today the simulator supplies ground truth through it; the
webcam rig has no pose and therefore no global map (the Navigator then runs in
local-only mode: drive toward a carrot in the robot frame).

Frames: world X east / Y north / theta CCW from +X; robot X forward / Y left.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from costmap_prototype import astar, drive_command, path_metres   # planner primitives
from perception_core import CoreCfg, LETHAL, UNKNOWN, inflate

#: global memory value for a cell seen lethal fewer than `lethal_confirm` times
UNCONFIRMED = 200


# ----------------------------------------------------------------------------
# 1. pose
# ----------------------------------------------------------------------------

@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def to_robot(self, wx, wy):
        """World point -> robot frame (forward, left)."""
        dx, dy = wx - self.x, wy - self.y
        c, s = math.cos(self.theta), math.sin(self.theta)
        return dx * c + dy * s, -dx * s + dy * c

    def to_world(self, rx, ry):
        """Robot-frame point -> world."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return self.x + rx * c - ry * s, self.y + rx * s + ry * c

    def as_dict(self):
        return dict(x=round(self.x, 3), y=round(self.y, 3), theta=round(self.theta, 4))


class PoseSource:
    """Interface: latest robot pose in the world frame, or None if unavailable."""
    def get(self) -> Optional[Pose]:
        raise NotImplementedError


class GroundTruthPose(PoseSource):
    """Pose pushed in from the simulator (stand-in for VSLAM)."""
    def __init__(self):
        self.pose: Optional[Pose] = None

    def set(self, x, y, theta):
        self.pose = Pose(float(x), float(y), float(theta))

    def get(self):
        return self.pose


class NoPose(PoseSource):
    """Webcam rig: nothing to localise against."""
    def get(self):
        return None


class VisualOdomPose(PoseSource):
    """
    Integrates frame-to-frame planar motion (see `ground_vo.GroundVO`) into a world pose.

    The world frame is simply wherever the rover was when this was created or last reset,
    facing along +X. That is enough for a global costmap and for world-frame goals: both
    only need a frame that is CONSISTENT over a run, not one that is absolute.

    This is dead reckoning over visual measurements, so error accumulates and is never
    corrected - there is no loop closure here. `dist_travelled` is exposed so a consumer
    can reason about how much drift has plausibly built up, and the pose is withheld
    entirely (`get()` returns None) once too many consecutive frames have failed, so the
    global map stops fusing rather than smearing itself against a stale pose.
    """

    def __init__(self, max_lost: int = 8):
        self.pose = Pose(0.0, 0.0, 0.0)
        self.max_lost = max_lost
        self.lost = 0
        self.started = False
        self.dist_travelled = 0.0
        self.confidence = 0.0

    def reset(self):
        self.pose = Pose(0.0, 0.0, 0.0)
        self.lost = 0
        self.started = False
        self.dist_travelled = 0.0
        self.confidence = 0.0

    def integrate(self, dx: float, dy: float, dtheta: float, confidence: float = 1.0) -> Pose:
        """Apply one robot-frame step: exact SE(2) composition. `(dx, dy)` is where the
        robot now is IN THE FRAME IT HAD BEFORE THE STEP (that is what `GroundVO` measures
        - the chord of the arc, not a velocity), so it is rotated by the PREVIOUS heading
        alone. A half-turn "midpoint" correction here would count the turn twice: the
        chord's own direction already carries it."""
        th = self.pose.theta
        c, s = math.cos(th), math.sin(th)
        self.pose.x += dx * c - dy * s
        self.pose.y += dx * s + dy * c
        self.pose.theta = math.atan2(math.sin(self.pose.theta + dtheta),
                                     math.cos(self.pose.theta + dtheta))
        self.dist_travelled += math.hypot(dx, dy)
        self.confidence = float(confidence)
        self.lost = 0
        self.started = True
        return self.pose

    def miss(self):
        """One frame produced no usable motion estimate."""
        self.lost += 1
        self.confidence *= 0.7

    def get(self):
        if not self.started or self.lost > self.max_lost:
            return None
        return self.pose

    def as_dict(self):
        return dict(x=round(self.pose.x, 3), y=round(self.pose.y, 3),
                    theta_deg=round(math.degrees(self.pose.theta), 1),
                    dist=round(self.dist_travelled, 2),
                    confidence=round(self.confidence, 2),
                    lost=self.lost, ok=self.get() is not None)


# ----------------------------------------------------------------------------
# 2. global costmap
# ----------------------------------------------------------------------------

class GlobalCostmap:
    """
    World-frame uint8 grid, `size_m` on a side, centred on the origin.

    fuse(): every MEASURED local cell is projected through the pose and written
    with np.maximum - the same "if any evidence says dangerous, it is dangerous"
    rule as the local map. UNKNOWN local cells never write, so unexplored ground
    stays UNKNOWN and explored ground is never forgotten. The local grid arrives
    already inflated; nothing inflates it again here.
    """

    def __init__(self, res: float = 0.25, size_m: float = 120.0, decay: float = 0.0,
                 clear_after: int = 0, lethal_confirm: int = 1):
        self.res = float(res)
        self.n = int(round(size_m / res))
        self.origin = -size_m / 2.0                 # world coordinate of cell (0, 0)
        self.grid = np.full((self.n, self.n), UNKNOWN, np.uint8)
        self.decay = decay                          # 0 = remember forever
        # Free-space clearing. 0 = off: pure max-fusion, an obstacle once seen stays
        # forever (the sim: a static course and a perfect pose). N > 0: a cell re-observed
        # as NOT an obstacle on N consecutive fusions takes the observed value, so an
        # obstacle that moved away, or a ghost smeared in by odometry drift, is forgotten
        # once the camera looks at that ground again. Asymmetric on purpose - one lethal
        # observation still marks a cell instantly, clearing needs sustained evidence.
        self.clear_after = int(clear_after)
        self.free_streak = np.zeros((self.n, self.n), np.uint8) if clear_after > 0 else None
        # Lethal confirmation (needs clear_after > 0). 1 = off: one lethal sighting is
        # remembered as lethal. N > 1: a cell must be SEEN lethal on N fusions before the
        # memory calls it lethal; until then it is stored as UNCONFIRMED (expensive, never
        # blocking). A monocular camera throws single-frame lethal specks - a puddle
        # labelled water for one frame, a depth spike - and in max-fusion memory each one
        # is permanent, gets inflated by a robot radius at plan time, and a 0.5 m trail
        # sprinkled with them plans as a wall. The live local map is untouched: a real
        # obstacle ahead is still lethal on the very first frame that sees it.
        self.lethal_confirm = int(lethal_confirm)
        self.lethal_hits = np.zeros((self.n, self.n), np.uint8) if (lethal_confirm > 1 and clear_after > 0) else None
        self.holds_raw = False                      # set by Navigator.step(raw=...)
        self.version = 0

    def reset(self):
        self.grid[:] = UNKNOWN
        if self.free_streak is not None:
            self.free_streak[:] = 0
        if self.lethal_hits is not None:
            self.lethal_hits[:] = 0
        self.version += 1

    # -- coordinates ---------------------------------------------------------
    def world_to_cell(self, wx, wy):
        ix = np.floor((np.asarray(wx) - self.origin) / self.res).astype(np.int64)
        iy = np.floor((np.asarray(wy) - self.origin) / self.res).astype(np.int64)
        return ix, iy

    def cell_to_world(self, ix, iy):
        return (self.origin + (np.asarray(ix) + 0.5) * self.res,
                self.origin + (np.asarray(iy) + 0.5) * self.res)

    def in_bounds(self, ix, iy):
        return (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)

    # -- fusion ----------------------------------------------------------------
    def fuse(self, local: np.ndarray, cfg: CoreCfg, pose: Pose):
        nx, ny = local.shape
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        rx = cfg.x_min + (ix + 0.5) * cfg.res
        ry = cfg.y_min + (iy + 0.5) * cfg.res
        measured = local != UNKNOWN
        if not measured.any():
            return
        rx, ry, val = rx[measured], ry[measured], local[measured]
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        wx = pose.x + rx * c - ry * s
        wy = pose.y + rx * s + ry * c
        gx, gy = self.world_to_cell(wx, wy)
        ok = self.in_bounds(gx, gy)
        gx, gy, val = gx[ok], gy[ok], val[ok]
        if self.decay > 0:
            # cells we are re-observing relax toward the new value first
            cur = self.grid[gx, gy].astype(np.float32)
            known = cur != UNKNOWN
            relaxed = np.where(known, cur * (1 - self.decay), 0.0)
            self.grid[gx, gy] = np.where(known, relaxed, 0).astype(np.uint8)
        cur = self.grid[gx, gy]
        cur = np.where(cur == UNKNOWN, 0, cur).astype(np.uint8)
        # several local cells can land in one global cell: resolve by max
        flat = gx * self.n + gy
        order = np.argsort(flat, kind="stable")
        if self.free_streak is None:
            flat_s, val_s = flat[order], np.maximum(cur[order], val[order])
            uniq, start = np.unique(flat_s, return_index=True)
            maxes = np.maximum.reduceat(val_s, start)
            self.grid.reshape(-1)[uniq] = maxes
        else:
            flat_s, obs_s = flat[order], val[order]
            uniq, start = np.unique(flat_s, return_index=True)
            obs = np.maximum.reduceat(obs_s, start)            # this frame, per global cell
            obs_seen = obs                                      # what the camera reported
            if self.lethal_hits is not None:
                hits = self.lethal_hits.reshape(-1)
                seen_lethal = (obs >= LETHAL) & (obs != UNKNOWN)
                hits[uniq] = np.where(seen_lethal, np.minimum(hits[uniq].astype(np.int32) + 1, 255), hits[uniq])
                obs = np.where(seen_lethal & (hits[uniq] < self.lethal_confirm), UNCONFIRMED, obs).astype(obs.dtype)
            prev = self.grid.reshape(-1)[uniq]
            prev = np.where(prev == UNKNOWN, 0, prev).astype(np.uint8)
            streak = self.free_streak.reshape(-1)
            free = obs_seen < LETHAL - 1                        # not lethal, not inscribed (as SEEN:
                                                                # an unconfirmed sighting is not free ground)
            streak[uniq] = np.where(free, np.minimum(streak[uniq].astype(np.int32) + 1, 255), 0)
            cleared = streak[uniq] >= self.clear_after
            if self.lethal_hits is not None:
                hits[uniq] = np.where(cleared, 0, hits[uniq])      # cleared ground starts over
            self.grid.reshape(-1)[uniq] = np.where(cleared, obs, np.maximum(prev, obs))
        self.version += 1

    # -- planning copies ------------------------------------------------------
    def pooled(self, factor: int = 2):
        """Max-pool by `factor` (UNKNOWN treated as 0 for pooling, restored after)."""
        n = (self.n // factor) * factor
        g = self.grid[:n, :n]
        unk = g == UNKNOWN
        v = np.where(unk, 0, g).reshape(n // factor, factor, n // factor, factor)
        m = v.max(axis=(1, 3))
        allunk = unk.reshape(n // factor, factor, n // factor, factor).all(axis=(1, 3))
        return np.where(allunk, UNKNOWN, m).astype(np.uint8)

    def to_png(self, path_world=None, pose: Optional[Pose] = None, goal=None,
               crop_m: Optional[float] = None, scale: int = 2) -> bytes:
        """Render for the dashboard: north up, east right."""
        g = self.grid
        x0 = y0 = 0
        if crop_m is not None and pose is not None:
            half = int(crop_m / self.res / 2)
            cx, cy = self.world_to_cell(pose.x, pose.y)
            x0, y0 = int(np.clip(cx - half, 0, self.n - 2 * half)), int(np.clip(cy - half, 0, self.n - 2 * half))
            g = g[x0:x0 + 2 * half, y0:y0 + 2 * half]
        img = _colourise(g)
        # image rows = -Y (north up), cols = X
        img = np.transpose(img, (1, 0, 2))[::-1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        H = img.shape[0]

        def px(wx, wy):
            ix, iy = self.world_to_cell(wx, wy)
            return int((ix - x0) * scale + scale / 2), int(H - ((iy - y0) * scale + scale / 2))

        if path_world:
            pts = np.array([px(x, y) for x, y in path_world], np.int32)
            cv2.polylines(img, [pts], False, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.polylines(img, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
        if goal is not None:
            cv2.drawMarker(img, px(goal[0], goal[1]), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        if pose is not None:
            p = px(pose.x, pose.y)
            q = px(pose.x + 1.5 * math.cos(pose.theta), pose.y + 1.5 * math.sin(pose.theta))
            cv2.circle(img, p, 5, (255, 255, 255), -1)
            cv2.arrowedLine(img, p, q, (255, 255, 255), 2, tipLength=0.4)
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return buf.tobytes()

    def crop_meta(self, pose: Optional[Pose], crop_m: Optional[float], scale: int = 2) -> dict:
        """Where the rendered image sits in the world, so a click can be inverted."""
        if crop_m is None or pose is None:
            return dict(origin_x=self.origin, origin_y=self.origin, res=self.res / 1.0,
                        w=self.n, h=self.n, scale=scale)
        half = int(crop_m / self.res / 2)
        cx, cy = self.world_to_cell(pose.x, pose.y)
        x0 = int(np.clip(cx - half, 0, self.n - 2 * half))
        y0 = int(np.clip(cy - half, 0, self.n - 2 * half))
        return dict(origin_x=self.origin + x0 * self.res, origin_y=self.origin + y0 * self.res,
                    res=self.res, w=2 * half, h=2 * half, scale=scale)


def _colourise(g: np.ndarray) -> np.ndarray:
    img = np.zeros((*g.shape, 3), np.uint8)
    unk = g == UNKNOWN
    v = g[~unk].astype(np.float32) / 253.0
    img[~unk] = np.clip(np.stack([(60 + 40 * v), (220 * (1 - v)), (60 + 180 * v)], -1), 0, 255).astype(np.uint8)
    img[unk] = (55, 55, 55)
    return img


# ----------------------------------------------------------------------------
# 3. global planner
# ----------------------------------------------------------------------------

@dataclass
class PlannerCfg:
    """Only the fields costmap_prototype.astar() reads, with global-scale values."""
    plan_unknown_cost: float = 40.0     # unexplored is not hazardous at map scale
    plan_cost_weight: float = 6.0
    LETHAL: int = LETHAL
    UNKNOWN: int = UNKNOWN
    pool: int = 2                       # plan at res * pool
    margin_m: float = 15.0              # box around start/goal
    robot_radius: float = 0.8
    algo: str = "astar"                 # "astar" (replan from scratch) | "dstar" (D* Lite, incremental)


class _InflateCfg:
    """The four fields perception_core.inflate() reads."""
    def __init__(self, res: float, robot_radius: float):
        self.res, self.robot_radius, self.LETHAL, self.UNKNOWN = res, robot_radius, LETHAL, UNKNOWN


def _planning_grid(gmap: GlobalCostmap, pose: Pose, goal, pcfg: PlannerCfg, box=None):
    """
    The coarse, boxed grid both global planners search, with the two fix-ups applied.
    `box` = (x0, x1, y0, y1) in pooled cells; None boxes start and goal with the margin.
    Returns (sub, (x0, y0), res, start, goal) with start/goal in sub-grid cells.
    """
    res = gmap.res * pcfg.pool
    g = gmap.pooled(pcfg.pool)
    n = g.shape[0]

    def cell(wx, wy):
        return (int(np.clip(np.floor((wx - gmap.origin) / res), 0, n - 1)),
                int(np.clip(np.floor((wy - gmap.origin) / res), 0, n - 1)))

    sx, sy = cell(pose.x, pose.y)
    gx, gy = cell(goal[0], goal[1])
    if box is None:
        m = int(math.ceil(pcfg.margin_m / res))
        x0, x1 = max(0, min(sx, gx) - m), min(n, max(sx, gx) + m + 1)
        y0, y1 = max(0, min(sy, gy) - m), min(n, max(sy, gy) + m + 1)
    else:
        x0, x1, y0, y1 = box
    sub = g[x0:x1, y0:y1].copy()
    if getattr(gmap, "holds_raw", False):
        # the memory holds raw obstacles (see Navigator.step): inflate the planning copy,
        # at the planning resolution, so the global path keeps a robot radius of clearance
        sub = inflate(sub, _InflateCfg(res, pcfg.robot_radius))

    # the robot is standing here, so here is drivable whatever the fusion smear says
    r = int(math.ceil(pcfg.robot_radius / res))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disc = (xx * xx + yy * yy) <= r * r
    lx, ly = sx - x0, sy - y0
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if disc[dy + r, dx + r] and 0 <= lx + dx < sub.shape[0] and 0 <= ly + dy < sub.shape[1]:
                if sub[lx + dx, ly + dy] >= 253 and sub[lx + dx, ly + dy] != UNKNOWN:
                    sub[lx + dx, ly + dy] = 100
    # a goal placed on an obstacle is still a direction to head in
    if 0 <= gx - x0 < sub.shape[0] and 0 <= gy - y0 < sub.shape[1] and sub[gx - x0, gy - y0] == LETHAL:
        sub[gx - x0, gy - y0] = 253
    return sub, (x0, y0), res, (lx, ly), (gx - x0, gy - y0)


def plan_global(gmap: GlobalCostmap, pose: Pose, goal, pcfg: PlannerCfg = PlannerCfg()):
    """
    A* on a coarse, boxed copy of the global map. Returns (path_world, reached).

    The full 480x480 map takes ~0.5 s in pure Python; a 2x pooled copy boxed
    around start and goal with a 15 m margin is <= 200x200 and ~80 ms worst case.
    """
    sub, (x0, y0), res, start, goal_c = _planning_grid(gmap, pose, goal, pcfg)
    path, reached = astar(sub, pcfg, start=start, goal=goal_c)
    world = [(gmap.origin + (ix + x0 + 0.5) * res, gmap.origin + (iy + y0 + 0.5) * res) for ix, iy in path]
    return world, reached


class DStarGlobalPlanner:
    """
    Global planning with D* Lite (see `dstar_lite.py`): same grid, same costs and same
    return value as `plan_global`, but the search is KEPT between replans and repaired
    where the map changed, instead of redone from scratch.

    D* Lite needs a fixed graph, so the box is frozen when a goal is set (start and goal
    plus the margin, like A*'s) and rebuilt only if the robot leaves it or the goal
    changes. An unreachable goal falls back to A* for that replan, which returns the
    path to the closest reachable cell - D* Lite has no equivalent answer.
    """

    def __init__(self, pcfg: PlannerCfg):
        self.pcfg = pcfg
        self.reset()

    def reset(self):
        self.ds = None
        self.goal = None
        self.box = None
        self.origin_cell = (0, 0)
        self.last_changed = 0
        self.fallbacks = 0

    def _cost(self, sub):
        from costmap_prototype import traversal_cost
        return traversal_cost(sub, self.pcfg)

    def plan(self, gmap: GlobalCostmap, pose: Pose, goal):
        from dstar_lite import DStarLite
        pcfg = self.pcfg
        res = gmap.res * pcfg.pool
        n = gmap.n // pcfg.pool
        sx = int(np.clip(np.floor((pose.x - gmap.origin) / res), 0, n - 1))
        sy = int(np.clip(np.floor((pose.y - gmap.origin) / res), 0, n - 1))
        inside = (self.box is not None and self.box[0] <= sx < self.box[1]
                  and self.box[2] <= sy < self.box[3])
        if self.ds is None or goal != self.goal or not inside:
            sub, (x0, y0), res, start, goal_c = _planning_grid(gmap, pose, goal, pcfg)
            self.box = (x0, x0 + sub.shape[0], y0, y0 + sub.shape[1])
            self.origin_cell, self.goal = (x0, y0), goal
            cost, blocked = self._cost(sub)
            self.ds = DStarLite(sub.shape, goal_c, cost, blocked, pcfg.plan_cost_weight)
            self.last_changed = sub.size
        else:
            sub, (x0, y0), res, start, goal_c = _planning_grid(gmap, pose, goal, pcfg, box=self.box)
            cost, blocked = self._cost(sub)
            self.last_changed = self.ds.update_costs(cost, blocked, start)

        path = self.ds.plan(start)
        reached = True
        if path is None:
            self.fallbacks += 1
            path, reached = astar(sub, pcfg, start=start, goal=goal_c)
        world = [(gmap.origin + (ix + x0 + 0.5) * res, gmap.origin + (iy + y0 + 0.5) * res) for ix, iy in path]
        return world, reached


def path_blocked(gmap: GlobalCostmap, path_world, pcfg: PlannerCfg = PlannerCfg()) -> bool:
    """True if any point of the world path now sits on a LETHAL cell."""
    if not path_world:
        return False
    xs = np.array([p[0] for p in path_world]); ys = np.array([p[1] for p in path_world])
    ix, iy = gmap.world_to_cell(xs, ys)
    ok = gmap.in_bounds(ix, iy)
    return bool((gmap.grid[ix[ok], iy[ok]] == LETHAL).any())


def carrot(path_world: Sequence, pose: Pose, cfg: CoreCfg, goal, ahead: Optional[float] = None):
    """
    Local goal for the fine planner: the first global-path point at least `ahead`
    metres from the robot (default x_max - 1.5), or the goal itself when it is
    already inside the local window. Returned in the ROBOT frame, unclamped -
    the caller decides what to do when it is behind or beside the robot.
    """
    ahead = ahead if ahead is not None else max(cfg.x_max - 1.5, cfg.x_min + 1.0)
    gx, gy = pose.to_robot(goal[0], goal[1])
    if math.hypot(gx, gy) <= ahead:
        return gx, gy
    if not path_world:
        return gx, gy
    for wx, wy in path_world:
        rx, ry = pose.to_robot(wx, wy)
        if math.hypot(rx, ry) >= ahead:
            return rx, ry
    return pose.to_robot(*path_world[-1])


def fill_unknown_from_global(local: np.ndarray, gmap: GlobalCostmap, cfg: CoreCfg, pose: Pose) -> np.ndarray:
    """
    Local planning grid = this frame's grid, with UNKNOWN cells backfilled from
    the global memory. The camera only sees a wedge; without memory the planner
    routes through never-observed cells beside the robot, and a trench edge that
    was lethal a second ago disappears as soon as the rover turns toward it.
    """
    nx, ny = local.shape
    out = local.copy()
    unk = local == UNKNOWN
    if not unk.any():
        return out
    ix, iy = np.nonzero(unk)
    rx = cfg.x_min + (ix + 0.5) * cfg.res
    ry = cfg.y_min + (iy + 0.5) * cfg.res
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    wx = pose.x + rx * c - ry * s
    wy = pose.y + rx * s + ry * c
    gx, gy = gmap.world_to_cell(wx, wy)
    ok = gmap.in_bounds(gx, gy)
    vals = gmap.grid[gx[ok], gy[ok]]
    out[ix[ok], iy[ok]] = vals
    return out


def inside_local(rx, ry, cfg: CoreCfg, pad: float = 0.0) -> bool:
    return (cfg.x_min + pad <= rx <= cfg.x_max - pad) and (cfg.y_min + pad <= ry <= cfg.y_max - pad)


# ----------------------------------------------------------------------------
# 4. the navigator
# ----------------------------------------------------------------------------

@dataclass
class NavCfg:
    goal_tol: float = 1.0          # metres: must exceed cfg.x_min or the goal hides behind the grid
    turn_gain: float = 1.5
    turn_enter_deg: float = 60.0   # carrot further off-axis than this -> turn in place
    turn_exit_deg: float = 15.0
    turn_min_x: float = 1.0        # carrot closer ahead than this -> turn in place
    v_max: float = 1.0
    w_max: float = 1.0
    slow_dist: float = 3.0         # start slowing this far from the goal
    blocked_frames: int = 3        # stop this long, then spin recovery
    recovery_frames: int = 20
    replan_period: float = 1.0     # seconds between global replans
    watchdog: float = 1.0          # seconds without a frame -> STOP
    plan_unknown_cost: float = 200.0   # local planner: UNKNOWN expensive, never blocked
    plan_cost_weight: float = 6.0
    lookahead: float = 1.5
    stop_dist: float = 0.7
    turn_slow: float = 0.6
    cmd_smooth: float = 0.5        # EMA on omega between frames (0 = off)
    inscribed_blocks: bool = True  # local planner: the 253 skirt is a collision, not a squeeze
    accel_max: float = 1.5         # m/s per second, ramps v instead of stepping it
    # Unknown-path gate. UNKNOWN is passable for the planner (it must be, or a monocular
    # robot never moves), which means a frame whose map came out empty - a lost plane,
    # a depth model off its scale - produced a full-speed plan straight through it.
    # Look at the first `unknown_gate_m` of the chosen path: this fraction UNKNOWN or
    # more -> half speed, `unknown_stop` or more -> BLOCKED. Off by default: the sim's
    # true depth never needs it, and its tuned runs must not change.
    unknown_gate: bool = False
    unknown_gate_m: float = 1.5
    unknown_slow: float = 0.3
    unknown_stop: float = 0.6
    # Starting inside the 253 skirt means the body is already within a radius of an
    # obstacle; astar only opens a way OUT, but leaving at cruise speed is how a noisy
    # frame turns into a scrape. None = no cap (the sim's tuned default).
    escape_v: Optional[float] = None
    LETHAL: int = LETHAL
    UNKNOWN: int = UNKNOWN


class _LocalCfg:
    """Adapter: what costmap_prototype.astar / drive_command read, from CoreCfg + NavCfg."""
    def __init__(self, cfg: CoreCfg, ncfg: NavCfg, goal_x: float, goal_y: float):
        self.x_min, self.x_max, self.y_min, self.y_max, self.res = cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max, cfg.res
        self.goal_x, self.goal_y = goal_x, goal_y
        self.plan_unknown_cost, self.plan_cost_weight = ncfg.plan_unknown_cost, ncfg.plan_cost_weight
        self.lookahead, self.stop_dist, self.v_max, self.w_max, self.turn_slow = (
            ncfg.lookahead, ncfg.stop_dist, ncfg.v_max, ncfg.w_max, ncfg.turn_slow)
        self.LETHAL, self.UNKNOWN = LETHAL, UNKNOWN
        self.inscribed_blocks, self.robot_radius = ncfg.inscribed_blocks, cfg.robot_radius


def local_goal_cell(cfg: CoreCfg, rx, ry):
    ix = int(round((rx - cfg.x_min) / cfg.res))
    iy = int(round((ry - cfg.y_min) / cfg.res))
    return int(np.clip(ix, 0, cfg.nx - 1)), int(np.clip(iy, 0, cfg.ny - 1))


@dataclass
class NavOutput:
    status: str
    v: float
    omega: float
    local_path: list = field(default_factory=list)       # [(ix, iy)]
    local_path_m: list = field(default_factory=list)     # [(x, y)] robot frame
    local_goal: Optional[tuple] = None                   # (ix, iy)
    aim: Optional[int] = None
    reached: Optional[bool] = None
    global_path: list = field(default_factory=list)      # [(wx, wy)]
    dist_to_goal: Optional[float] = None
    note: str = ""


class Navigator:
    """
    State machine tying the layers together. Call step() once per perceived frame.

    NO_GOAL   : nothing to do, stop
    PLANNING  : goal set, first global plan pending
    TURNING   : carrot is behind/beside -> rotate in place toward it
    DRIVING   : local A* to the carrot + pure pursuit
    BLOCKED   : local planner found no safe first step -> stop, then spin recovery
    LOST      : a global map is in use but the pose is unavailable -> stop and wait
    ARRIVED   : within goal_tol of the goal -> stop until a new goal
    """

    def __init__(self, cfg: CoreCfg, ncfg: NavCfg = NavCfg(), gmap: Optional[GlobalCostmap] = None,
                 pcfg: PlannerCfg = PlannerCfg()):
        self.cfg, self.ncfg, self.pcfg = cfg, ncfg, pcfg
        self.gmap = gmap
        self.dstar = DStarGlobalPlanner(pcfg) if pcfg.algo == "dstar" else None
        self.goal = None
        self.state = "NO_GOAL"
        self.global_path: list = []
        self.global_reached = None
        self._last_plan_t = -1e9
        self._blocked = 0
        self._recover = 0
        self._last_frame_t = time.monotonic()
        self._turning = False
        self._v_prev = 0.0
        self._w_prev = 0.0
        self._t_prev = None

    # -- goal management ------------------------------------------------------
    def set_goal(self, x, y):
        self.goal = (float(x), float(y))
        if self.dstar is not None:
            self.dstar.reset()
        self.state = "PLANNING"
        self.global_path, self.global_reached = [], None
        self._last_plan_t = -1e9
        self._blocked = self._recover = 0
        self._turning = False

    def clear_goal(self):
        self.goal = None
        self.state = "NO_GOAL"
        self.global_path, self.global_reached = [], None

    def reset(self):
        self.clear_goal()
        if self.gmap is not None:
            self.gmap.reset()
        if self.dstar is not None:
            self.dstar.reset()

    def watchdog(self, now: Optional[float] = None) -> bool:
        """True if no frame has arrived within ncfg.watchdog seconds."""
        now = time.monotonic() if now is None else now
        return (now - self._last_frame_t) > self.ncfg.watchdog

    # -- one cycle -----------------------------------------------------------
    def step(self, local_grid: np.ndarray, pose: Optional[Pose], now: Optional[float] = None,
             raw: Optional[np.ndarray] = None) -> NavOutput:
        """
        `local_grid` is this frame's INFLATED costmap. Pass `raw` (the same grid before
        inflation, `CoreResult.raw`) and the memory stores raw obstacles instead: fused
        and backfilled raw, then the combined planning grid is inflated once. Fusing
        inflated grids makes every noisy detection's 1-robot-radius skirt permanent, and
        re-observing it from a slightly different place widens it - after a while the
        map around a rock field is all skirt and nothing passes.
        """
        now = time.monotonic() if now is None else now
        self._last_frame_t = now
        cfg, ncfg = self.cfg, self.ncfg

        if self.gmap is not None and pose is not None:
            if raw is not None:
                self.gmap.holds_raw = True
                self.gmap.fuse(raw, cfg, pose)
                local_grid = inflate(fill_unknown_from_global(raw, self.gmap, cfg, pose), cfg)
            else:
                self.gmap.fuse(local_grid, cfg, pose)
                local_grid = fill_unknown_from_global(local_grid, self.gmap, cfg, pose)
        self.last_plan_grid = local_grid          # exactly what the local planner sees (recorder)

        if self.goal is None:
            self.state = "NO_GOAL"
            return NavOutput("NO_GOAL", 0.0, 0.0)

        # A navigator WITH a global map was given a world-frame goal. Without a pose that
        # goal cannot be put into the robot frame, and reading its coordinates as if they
        # were robot-relative (the local-only branch below) sends the rover toward a
        # point that has nothing to do with the goal - the wrong way entirely once it has
        # turned. The only safe answer is to stop and wait for the pose to come back.
        if self.gmap is not None and pose is None:
            self.state = "LOST"
            self._turning = False
            self._v_prev, self._w_prev = 0.0, 0.0
            self._blocked = self._recover = 0
            return NavOutput("LOST", 0.0, 0.0, global_path=self.global_path,
                             note="no pose - holding until odometry recovers")

        # ---- where is the goal, in the robot frame? ---------------------------
        if pose is not None:
            gx, gy = pose.to_robot(*self.goal)
        else:
            gx, gy = self.goal            # local-only mode: the goal IS robot-frame
        dist = math.hypot(gx, gy)
        if dist <= ncfg.goal_tol:
            self.state = "ARRIVED"
            return NavOutput("ARRIVED", 0.0, 0.0, dist_to_goal=dist, global_path=self.global_path,
                             local_goal=local_goal_cell(cfg, float(np.clip(gx, cfg.x_min + cfg.res, cfg.x_max - cfg.res)),
                                                        float(np.clip(gy, cfg.y_min + cfg.res, cfg.y_max - cfg.res))))

        # ---- global layer -----------------------------------------------------
        if self.gmap is not None and pose is not None:
            due = (now - self._last_plan_t) >= ncfg.replan_period
            if due or not self.global_path or path_blocked(self.gmap, self.global_path, self.pcfg):
                if self.dstar is not None:
                    self.global_path, self.global_reached = self.dstar.plan(self.gmap, pose, self.goal)
                else:
                    self.global_path, self.global_reached = plan_global(self.gmap, pose, self.goal, self.pcfg)
                self._last_plan_t = now
            cx, cy = carrot(self.global_path, pose, cfg, self.goal)
        else:
            cx, cy = gx, gy

        # Where the carrot lands on the local grid. Computed HERE, before the turn-in-place
        # branch, purely so every exit path can report it: the operator clicks a goal, the
        # rover turns towards it, and if this were computed later the marker would vanish
        # from the costmap for exactly as long as the turn lasts - which reads as "my
        # click did nothing".
        lx = float(np.clip(cx, cfg.x_min + cfg.res, cfg.x_max - cfg.res))
        ly = float(np.clip(cy, cfg.y_min + cfg.res, cfg.y_max - cfg.res))
        gcell = local_goal_cell(cfg, lx, ly)

        # ---- turn in place when the carrot is not in front of us ---------------
        bearing = math.atan2(cy, cx)
        enter, exit_ = math.radians(ncfg.turn_enter_deg), math.radians(ncfg.turn_exit_deg)
        if self._turning:
            if abs(bearing) < exit_:
                self._turning = False
        elif (cx < ncfg.turn_min_x and abs(bearing) > exit_) or abs(bearing) > enter:
            # (a close carrot DEAD AHEAD is driven to, not turned toward: turning in place
            # at omega = gain * 0 is a stall that never ends)
            self._turning = True
        if self._turning:
            self.state = "TURNING"
            self._v_prev, self._w_prev = 0.0, 0.0
            w = float(np.clip(ncfg.turn_gain * bearing, -ncfg.w_max, ncfg.w_max))
            return NavOutput("TURNING", 0.0, w, local_goal=gcell, global_path=self.global_path,
                             dist_to_goal=dist, note=f"bearing {math.degrees(bearing):+.0f} deg")

        # ---- local layer: A* to the carrot, pure pursuit ------------------------
        lcfg = _LocalCfg(cfg, ncfg, lx, ly)
        path, reached = astar(local_grid, lcfg, goal=gcell)
        v = w = 0.0; aim = None
        if len(path) >= 2:
            v, w, aim = drive_command(path, lcfg)
        if len(path) < 2 or (v == 0.0 and w == 0.0):
            # No step to take: [] = lethal dead ahead; [start] = nothing reachable is any
            # closer to the carrot; (0, 0) from pure pursuit = the reachable part of the
            # plan ends inside stop_dist. All three used to fall through to DRIVING at
            # v = 0, so recovery never ran and the rover sat still indefinitely while
            # reporting that it was driving.
            path = []
            # The goal itself is inside a hazard's clearance (clicked next to a trench,
            # on a rock): no safe cell is closer to it than where we already are, so this
            # is as close as it gets. Arrive here rather than spin-recovering forever.
            if inside_local(gx, gy, cfg) and local_grid[local_goal_cell(cfg, gx, gy)] in (LETHAL - 1, LETHAL):
                self.state = "ARRIVED"
                self._blocked = self._recover = 0
                self._v_prev, self._w_prev = 0.0, 0.0
                return NavOutput("ARRIVED", 0.0, 0.0, local_goal=gcell, global_path=self.global_path,
                                 dist_to_goal=dist, note="goal is inside a hazard's clearance - closest safe point")
            self._blocked += 1
            if self._blocked > ncfg.blocked_frames:
                self._recover += 1
                if self._recover <= ncfg.recovery_frames:
                    self.state = "BLOCKED"
                    w = ncfg.w_max * 0.5 * (1 if bearing >= 0 else -1)
                    return NavOutput("BLOCKED", 0.0, w, local_goal=gcell, global_path=self.global_path,
                                     dist_to_goal=dist, note="spin recovery")
                self._recover = 0
                self._blocked = 0
            self.state = "BLOCKED"
            return NavOutput("BLOCKED", 0.0, 0.0, local_goal=gcell, global_path=self.global_path,
                             dist_to_goal=dist, note="no safe first step")
        self._blocked = 0
        self._recover = 0

        # Safety caps are applied AFTER the accel ramp below: the ramp limits how fast the
        # command may fall, and a cap it could lift is not a cap.
        note, v_cap = "", None
        if ncfg.unknown_gate:
            n_gate = max(1, int(ncfg.unknown_gate_m / cfg.res))
            ahead = [local_grid[c] for c in path[:n_gate]]
            unk = sum(1 for c in ahead if c == UNKNOWN) / len(ahead)
            if unk >= ncfg.unknown_stop:
                self.state = "BLOCKED"
                self._v_prev, self._w_prev = 0.0, 0.0
                return NavOutput("BLOCKED", 0.0, 0.0, local_goal=gcell, global_path=self.global_path,
                                 dist_to_goal=dist, note=f"path ahead {unk:.0%} unknown")
            if unk >= ncfg.unknown_slow:
                v_cap = 0.5 * v
                note = f"slow: path ahead {unk:.0%} unknown"
        if ncfg.escape_v is not None and local_grid[path[0]] == LETHAL - 1:
            v_cap = ncfg.escape_v if v_cap is None else min(v_cap, ncfg.escape_v)
            note = "crawl: leaving an obstacle's clearance"

        # slow into the goal
        if dist < ncfg.slow_dist:
            v *= max(0.25, dist / ncfg.slow_dist)
        # smooth: the local path is re-planned from scratch every frame on a 0.1 m
        # grid, so the raw pure-pursuit omega steps by ~0.2 rad/s frame to frame.
        dt = 0.2 if self._t_prev is None else max(0.05, min(1.0, now - self._t_prev))
        self._t_prev = now
        if self.state == "DRIVING" and ncfg.cmd_smooth > 0:
            w = ncfg.cmd_smooth * self._w_prev + (1 - ncfg.cmd_smooth) * w
        dv = ncfg.accel_max * dt
        v = float(np.clip(v, self._v_prev - 2 * dv, self._v_prev + dv))
        if v_cap is not None:
            v = min(v, v_cap)
        self._v_prev, self._w_prev = v, w
        self.state = "DRIVING"
        return NavOutput("DRIVING", float(v), float(w), local_path=path,
                         local_path_m=path_metres(path, lcfg), local_goal=gcell, aim=aim,
                         reached=reached, global_path=self.global_path, dist_to_goal=dist, note=note)
