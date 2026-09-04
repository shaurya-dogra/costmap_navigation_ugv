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
        |
    carrot()                  first path point ~x_max ahead   -> local goal
        |
    local A* + pure pursuit   (costmap_prototype.astar / drive_command)
        |
    Navigator                 NO_GOAL / PLANNING / TURNING / DRIVING / BLOCKED / ARRIVED

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
from perception_core import CoreCfg, LETHAL, UNKNOWN


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

    def __init__(self, res: float = 0.25, size_m: float = 120.0, decay: float = 0.0):
        self.res = float(res)
        self.n = int(round(size_m / res))
        self.origin = -size_m / 2.0                 # world coordinate of cell (0, 0)
        self.grid = np.full((self.n, self.n), UNKNOWN, np.uint8)
        self.decay = decay                          # 0 = remember forever
        self.version = 0

    def reset(self):
        self.grid[:] = UNKNOWN
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
        flat_s, val_s = flat[order], np.maximum(cur[order], val[order])
        uniq, start = np.unique(flat_s, return_index=True)
        maxes = np.maximum.reduceat(val_s, start)
        self.grid.reshape(-1)[uniq] = maxes
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


def plan_global(gmap: GlobalCostmap, pose: Pose, goal, pcfg: PlannerCfg = PlannerCfg()):
    """
    A* on a coarse, boxed copy of the global map. Returns (path_world, reached).

    The full 480x480 map takes ~0.5 s in pure Python; a 2x pooled copy boxed
    around start and goal with a 15 m margin is <= 200x200 and ~80 ms worst case.
    """
    res = gmap.res * pcfg.pool
    g = gmap.pooled(pcfg.pool)
    n = g.shape[0]

    def cell(wx, wy):
        return (int(np.clip(np.floor((wx - gmap.origin) / res), 0, n - 1)),
                int(np.clip(np.floor((wy - gmap.origin) / res), 0, n - 1)))

    sx, sy = cell(pose.x, pose.y)
    gx, gy = cell(goal[0], goal[1])
    m = int(math.ceil(pcfg.margin_m / res))
    x0, x1 = max(0, min(sx, gx) - m), min(n, max(sx, gx) + m + 1)
    y0, y1 = max(0, min(sy, gy) - m), min(n, max(sy, gy) + m + 1)
    sub = g[x0:x1, y0:y1].copy()

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
    if sub[gx - x0, gy - y0] == LETHAL:
        sub[gx - x0, gy - y0] = 253

    path, reached = astar(sub, pcfg, start=(lx, ly), goal=(gx - x0, gy - y0))
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
    accel_max: float = 1.5         # m/s per second, ramps v instead of stepping it
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
    ARRIVED   : within goal_tol of the goal -> stop until a new goal
    """

    def __init__(self, cfg: CoreCfg, ncfg: NavCfg = NavCfg(), gmap: Optional[GlobalCostmap] = None,
                 pcfg: PlannerCfg = PlannerCfg()):
        self.cfg, self.ncfg, self.pcfg = cfg, ncfg, pcfg
        self.gmap = gmap
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

    def watchdog(self, now: Optional[float] = None) -> bool:
        """True if no frame has arrived within ncfg.watchdog seconds."""
        now = time.monotonic() if now is None else now
        return (now - self._last_frame_t) > self.ncfg.watchdog

    # -- one cycle -----------------------------------------------------------
    def step(self, local_grid: np.ndarray, pose: Optional[Pose], now: Optional[float] = None) -> NavOutput:
        now = time.monotonic() if now is None else now
        self._last_frame_t = now
        cfg, ncfg = self.cfg, self.ncfg

        if self.gmap is not None and pose is not None:
            self.gmap.fuse(local_grid, cfg, pose)
            local_grid = fill_unknown_from_global(local_grid, self.gmap, cfg, pose)

        if self.goal is None:
            self.state = "NO_GOAL"
            return NavOutput("NO_GOAL", 0.0, 0.0)

        # ---- where is the goal, in the robot frame? ---------------------------
        if pose is not None:
            gx, gy = pose.to_robot(*self.goal)
        else:
            gx, gy = self.goal            # local-only mode: the goal IS robot-frame
        dist = math.hypot(gx, gy)
        if dist <= ncfg.goal_tol:
            self.state = "ARRIVED"
            return NavOutput("ARRIVED", 0.0, 0.0, dist_to_goal=dist, global_path=self.global_path)

        # ---- global layer -----------------------------------------------------
        if self.gmap is not None and pose is not None:
            due = (now - self._last_plan_t) >= ncfg.replan_period
            if due or not self.global_path or path_blocked(self.gmap, self.global_path, self.pcfg):
                self.global_path, self.global_reached = plan_global(self.gmap, pose, self.goal, self.pcfg)
                self._last_plan_t = now
            cx, cy = carrot(self.global_path, pose, cfg, self.goal)
        else:
            cx, cy = gx, gy

        # ---- turn in place when the carrot is not in front of us ---------------
        bearing = math.atan2(cy, cx)
        enter, exit_ = math.radians(ncfg.turn_enter_deg), math.radians(ncfg.turn_exit_deg)
        if self._turning:
            if abs(bearing) < exit_:
                self._turning = False
        elif cx < ncfg.turn_min_x or abs(bearing) > enter:
            self._turning = True
        if self._turning:
            self.state = "TURNING"
            self._v_prev, self._w_prev = 0.0, 0.0
            w = float(np.clip(ncfg.turn_gain * bearing, -ncfg.w_max, ncfg.w_max))
            return NavOutput("TURNING", 0.0, w, global_path=self.global_path, dist_to_goal=dist,
                             note=f"bearing {math.degrees(bearing):+.0f} deg")

        # ---- local layer: A* to the carrot, pure pursuit ------------------------
        lx = float(np.clip(cx, cfg.x_min + cfg.res, cfg.x_max - cfg.res))
        ly = float(np.clip(cy, cfg.y_min + cfg.res, cfg.y_max - cfg.res))
        lcfg = _LocalCfg(cfg, ncfg, lx, ly)
        gcell = local_goal_cell(cfg, lx, ly)
        path, reached = astar(local_grid, lcfg, goal=gcell)
        if not path:
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

        v, w, aim = drive_command(path, lcfg)
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
        self._v_prev, self._w_prev = v, w
        self.state = "DRIVING"
        return NavOutput("DRIVING", float(v), float(w), local_path=path,
                         local_path_m=path_metres(path, lcfg), local_goal=gcell, aim=aim,
                         reached=reached, global_path=self.global_path, dist_to_goal=dist)
