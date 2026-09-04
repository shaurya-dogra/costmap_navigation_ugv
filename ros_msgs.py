"""
ros_msgs.py - Nav2-shaped message dictionaries (no ROS dependency)
===================================================================

Pure functions that turn this project's grids, poses and commands into dicts
laid out exactly like the ROS 2 messages Nav2 consumes or produces:

    nav_msgs/OccupancyGrid   occupancy_grid(grid, res, origin_xy, frame_id)
    nav_msgs/Odometry        odometry(pose, v, omega)
    nav_msgs/Path            path_msg(points_world, frame_id)
    geometry_msgs/Twist      twist(v, omega)

They serialise to JSON directly, which is what rosbridge_suite expects on the
wire, so publishing them to a real Nav2 later is `roslibpy.Topic.publish(msg)`.

Cost conversion: this project's 0..253 / 254 LETHAL / 255 UNKNOWN byte becomes
OccupancyGrid's -1 (unknown) / 0..100. LETHAL maps to 100; the inflation skirt
(253) maps to 99 so Nav2's own inflation layer still sees it as near-lethal.
"""

from __future__ import annotations

import math
import time

import numpy as np

LETHAL, UNKNOWN = 254, 255


def _stamp(t=None):
    t = time.time() if t is None else t
    sec = int(t)
    return {"sec": sec, "nanosec": int((t - sec) * 1e9)}


def _quat_yaw(theta):
    return {"x": 0.0, "y": 0.0, "z": math.sin(theta / 2.0), "w": math.cos(theta / 2.0)}


def grid_to_occupancy(grid: np.ndarray) -> np.ndarray:
    """uint8 cost grid -> int8 occupancy values (-1 unknown, 0..100)."""
    g = grid.astype(np.int16)
    occ = np.where(g == UNKNOWN, -1,
                   np.where(g >= LETHAL, 100,
                            np.where(g >= 253, 99, np.round(g * 98.0 / 252.0)))).astype(np.int8)
    return occ


def occupancy_grid(grid: np.ndarray, res: float, origin_xy, frame_id: str = "map",
                   layout: str = "xy", t=None) -> dict:
    """
    grid with axis 0 = X, axis 1 = Y (this project's layout) -> OccupancyGrid.

    OccupancyGrid.data is row-major with X varying fastest (index = y*width + x),
    so the grid is transposed before flattening. `origin_xy` is the world
    position of the grid's (0, 0) corner.
    """
    if layout != "xy":
        raise ValueError("only axis0=X, axis1=Y grids are supported")
    nx, ny = grid.shape
    occ = grid_to_occupancy(grid).T                       # (ny, nx): rows = y
    return {
        "header": {"stamp": _stamp(t), "frame_id": frame_id},
        "info": {
            "map_load_time": _stamp(t),
            "resolution": float(res),
            "width": int(nx),
            "height": int(ny),
            "origin": {"position": {"x": float(origin_xy[0]), "y": float(origin_xy[1]), "z": 0.0},
                       "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        },
        "data": occ.ravel().tolist(),
    }


def odometry(pose, v: float, omega: float, frame_id: str = "odom", child: str = "base_link", t=None) -> dict:
    return {
        "header": {"stamp": _stamp(t), "frame_id": frame_id},
        "child_frame_id": child,
        "pose": {"pose": {"position": {"x": float(pose.x), "y": float(pose.y), "z": 0.0},
                          "orientation": _quat_yaw(float(pose.theta))},
                 "covariance": [0.0] * 36},
        "twist": {"twist": twist(v, omega), "covariance": [0.0] * 36},
    }


def twist(v: float, omega: float) -> dict:
    return {"linear": {"x": float(v), "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": float(omega)}}


def path_msg(points, frame_id: str = "map", t=None) -> dict:
    stamp = _stamp(t)
    poses = []
    for i, (x, y) in enumerate(points):
        if i + 1 < len(points):
            yaw = math.atan2(points[i + 1][1] - y, points[i + 1][0] - x)
        elif i > 0:
            yaw = math.atan2(y - points[i - 1][1], x - points[i - 1][0])
        else:
            yaw = 0.0
        poses.append({"header": {"stamp": stamp, "frame_id": frame_id},
                      "pose": {"position": {"x": float(x), "y": float(y), "z": 0.0},
                               "orientation": _quat_yaw(yaw)}})
    return {"header": {"stamp": stamp, "frame_id": frame_id}, "poses": poses}
