#!/usr/bin/env python3
"""
perception_server.py - camera in, costmap + drive command out  (SIH PS 26126)
=============================================================================

One process serves every demo mode:

  --source sim          frames are PUSHED by the Three.js rover over WebSocket
                        (RGB JPEG + optional true depth); commands go back.
  --source rover        frames are PUSHED by the real Pi (rover_agent.py) over the same
                        socket: RGB JPEG + intrinsics + lens distortion, no depth.
  --source 0            MacBook / phone webcam captured here, dashboard only.
  --source clip.mp4     recorded footage, looped.

Behaviour is driven by CAPABILITY FLAGS set once in __init__ (pushes_frames,
has_true_depth, has_pose, is_vehicle), never by the source name - so adding a source
means declaring what it can do rather than editing every branch that used to test it.

Pipeline per frame (worker thread, latest frame wins):

  undist (lens rectification; a no-op when the source sends no coefficients)
  depth  (Depth Anything V2 metric | metric-indoor | relative | affine | sim truth)
  sem    (YOLO26 ADE20K -> per-pixel cost)
  core   (perception_core: self-calibrating ground plane -> local costmap)
  nav    (navstack: global fusion + global A* + carrot + local A* + pure pursuit)
  render (panels -> JPEG/PNG -> one JSON `nav` message, see PROTOCOL.md)

    python perception_server.py --source 0 --rig macbook
    python perception_server.py --source sim --depth sim
    python perception_server.py --source 3d_sim_video.mp4 --depth metric --windows
    python perception_server.py --source rover --depth metric-indoor --profile

then open http://localhost:8790 (dashboard) and/or the SLAM3D page.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import asyncio
import base64
import json
import math
import struct
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from aiohttp import web, WSMsgType

import perception_core as pc
import navstack as ns
import ground_vo as gvo
import ros_msgs as rm

HERE = Path(__file__).resolve().parent
# Semantic weights per source. The sim keeps the higher-capacity "s" model (its
# frames are synthetic and it has GPU headroom); camera sources default to the
# nano model, 3x faster, so the live feed stays real-time. --sem-weights overrides.
SEM_WEIGHTS = {
    "sim": [HERE / ".." / "object segmentation" / "yolo26s-sem-ade20k.pt",
            HERE / "yolo26s-sem-ade20k.pt",
            HERE / ".." / "object segmentation" / "yolo26n-sem-ade20k.pt",
            HERE / "yolo26n-sem-ade20k.pt"],
    "camera": [HERE / "yolo26n-sem-ade20k.pt",
               HERE / ".." / "object segmentation" / "yolo26n-sem-ade20k.pt",
               HERE / ".." / "object segmentation" / "yolo26s-sem-ade20k.pt",
               HERE / "yolo26s-sem-ade20k.pt"],
    # The rover gets the "s" model, NOT the nano one, and the reason is worth recording.
    # On a polished indoor floor the nano model labelled 91 % of the ground "water" -
    # cost 254, which build_costmap treats as UNCONDITIONALLY lethal because water has no
    # height for the geometry channel to argue with. The rover was correctly refusing to
    # drive across a dry floor. The "s" model labelled the same frame 100 % "floor".
    # Compute is off-board here, so the extra tens of milliseconds are worth paying;
    # --sem-weights still overrides.
    "rover":  [HERE / ".." / "object segmentation" / "yolo26s-sem-ade20k.pt",
               HERE / "yolo26s-sem-ade20k.pt",
               HERE / ".." / "object segmentation" / "yolo26n-sem-ade20k.pt",
               HERE / "yolo26n-sem-ade20k.pt"],
}

TUNABLE = ("obstacle_h", "ditch_h", "robot_radius", "sem_lethal_frac", "min_cell_pts",
           "plane_gate", "plane_near_range", "max_depth")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def b64img(img: np.ndarray, kind: str = "jpeg", quality: int = 80) -> str:
    if kind == "png":
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        mime = "image/png"
    else:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def b64png_bytes(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def parse_frame(data: bytes):
    """Binary frame -> (header, bgr, depth_m or None). See PROTOCOL.md §2."""
    (hl,) = struct.unpack_from("<I", data, 0)
    header = json.loads(data[4:4 + hl].decode("utf-8"))
    off = 4 + hl
    jl = int(header["jpeg_len"])
    jpeg = np.frombuffer(data, np.uint8, count=jl, offset=off)
    bgr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    off += jl
    depth = None
    d = header.get("depth")
    if d:
        n = int(d["w"]) * int(d["h"])
        if len(data) >= off + 2 * n:
            u16 = np.frombuffer(data, "<u2", count=n, offset=off).reshape(int(d["h"]), int(d["w"]))
            depth = u16.astype(np.float32) / 1000.0
    return header, bgr, depth


class Profile:
    def __init__(self):
        self.t = time.perf_counter()
        self.d = {}

    def lap(self, name):
        now = time.perf_counter()
        self.d[name] = round((now - self.t) * 1000, 1)
        self.t = now
        return self


# ----------------------------------------------------------------------------
# frame sources
# ----------------------------------------------------------------------------

class FrameSlot:
    """Latest-frame-wins handoff between producer and the worker thread."""

    def __init__(self):
        self.cv = threading.Condition()
        self.item = None
        self.dropped = 0

    def put(self, item):
        with self.cv:
            if self.item is not None:
                self.dropped += 1
            self.item = item
            self.cv.notify()

    def get(self, timeout=0.5):
        with self.cv:
            if self.item is None:
                self.cv.wait(timeout)
            item, self.item = self.item, None
            return item


class CaptureSource(threading.Thread):
    """cv2.VideoCapture reader: webcam (zero-lag, latest frame) or looped video file."""

    def __init__(self, src, slot: FrameSlot, size=(1280, 720), every=1, fps_cap=None):
        super().__init__(daemon=True)
        self.src = int(src) if str(src).isdigit() else src
        self.is_file = not isinstance(self.src, int) and Path(str(self.src)).exists()
        self.slot, self.size, self.every, self.fps_cap = slot, size, every, fps_cap
        self.stop_flag = False
        self.seq = 0

    def run(self):
        if isinstance(self.src, int) and sys.platform == "darwin":
            cap = cv2.VideoCapture(self.src, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.src)
        else:
            cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            print(f"[capture] cannot open source {self.src}", file=sys.stderr)
            os._exit(2)
        if not self.is_file:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, max(self.size[0], 1280))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max(self.size[1], 720))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        period = 1.0 / self.fps_cap if self.fps_cap else 0.0
        i = 0
        while not self.stop_flag:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if self.is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.02)
                continue
            i += 1
            if i % self.every:
                continue
            if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
                frame = cv2.resize(frame, self.size)
            self.seq += 1
            self.slot.put(dict(header=dict(seq=self.seq, t=time.time() * 1000, w=self.size[0], h=self.size[1],
                                           mode="manual", pose=None), bgr=frame, depth=None))
            if self.is_file:
                # a file has no natural pacing: wait for the worker to consume
                while self.slot.item is not None and not self.stop_flag:
                    time.sleep(0.005)
            if period:
                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)
        cap.release()


# ----------------------------------------------------------------------------
# the server
# ----------------------------------------------------------------------------

class PerceptionServer:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.clients: dict = {}                 # ws -> role
        self.slot = FrameSlot()
        if a.source in ("sim", "rover"):
            self.source_kind = a.source
        else:
            self.source_kind = "video" if Path(a.source).exists() else "webcam"

        # ---- capabilities ------------------------------------------------------
        # Everything below branches on THESE, never on source_kind again. Adding a
        # source then means declaring what it can do, instead of editing ten call
        # sites and hoping none was missed - which is how the sim path would break.
        self.pushes_frames = self.source_kind in ("sim", "rover")   # frames arrive over the socket
        self.has_true_depth = self.source_kind == "sim"             # renderer depth rides in the frame
        self.has_pose = self.source_kind in ("sim", "rover")        # sim: ground truth; rover: ground-plane VO
        self.is_vehicle = self.source_kind in ("sim", "rover")      # cmd_vel reaches something real
        self.mode_auto = a.auto
        self.depth_mode = a.depth
        self.lock = threading.Lock()
        self.last_nav_msg: Optional[dict] = None
        self.last_grid: Optional[np.ndarray] = None
        self.last_pose: Optional[ns.Pose] = None
        self.last_cmd = (0.0, 0.0)
        self.last_seq = -1
        self.fps = 0.0
        self.windows = a.windows
        self._win_frames: dict = {}
        self.vo: Optional[gvo.GroundVO] = None   # rover only; None elsewhere
        self._streams: dict = {}          # name -> latest full-size JPEG bytes
        self._stream_seq = -1             # bumped per processed frame, so viewers can wait

        # ---- config per source -------------------------------------------------
        if self.source_kind == "sim":
            self.cfg = pc.CoreCfg(w=640, h=360, x_min=0.5, x_max=12.0, y_min=-5.0, y_max=5.0, res=0.1,
                                  robot_radius=a.robot_radius or 1.0, max_depth=20.0, plane_near_range=6.0,
                                  stride=2, ditch_max_range=9.0, hole_max_range=9.0)
            self.ncfg = ns.NavCfg(v_max=a.v_max or 2.0, w_max=a.w_max or 0.8, goal_tol=1.2, slow_dist=4.0,
                                  lookahead=2.5, stop_dist=1.2, turn_gain=1.0, turn_enter_deg=70.0)
            self.gmap = ns.GlobalCostmap(res=0.25, size_m=160.0)
            self.pose_src = ns.GroundTruthPose()
        elif self.source_kind == "rover":
            # Measured rig: IMX219 at 640x480 from the FULL-FOV 1640x1232 sensor mode.
            # The camera sits ~0.17 m up, so the honest sensing horizon is 2.6 m and the
            # speeds are scaled to it: at 0.5 m/s with ~150 ms of link latency the rover
            # stops well inside the map. See pc.rover_cfg for the derivation.
            self.cfg = pc.rover_cfg(cam_height=a.nominal_height or 0.17,
                                    robot_radius=a.robot_radius or 0.20)
            if a.hfov:
                self.cfg.fx, self.cfg.fy, self.cfg.cx, self.cfg.cy = pc.intrinsics_from_hfov(
                    self.cfg.w, self.cfg.h, a.hfov)
            self.ncfg = ns.NavCfg(v_max=a.v_max or 0.5, w_max=a.w_max or 1.0, goal_tol=0.35,
                                  slow_dist=1.0, lookahead=0.6, stop_dist=0.35,
                                  turn_min_x=0.4, accel_max=0.6)
            # Ground-plane visual odometry supplies the pose, so the global map works.
            # 0.05 m to match the local grid, 16 m across - a rover with a 2.6 m horizon
            # is not going to outrun that, and it keeps fuse() cheap.
            self.gmap = ns.GlobalCostmap(res=0.05, size_m=16.0)
            self.pose_src = ns.VisualOdomPose()
            self.vo = gvo.GroundVO()
            if a.map_view == 60.0:      # the sim-scaled default dwarfs a 16 m map
                a.map_view = 8.0
        else:
            W, H = a.proc_width, int(round(a.proc_width * 9 / 16))
            preset = pc.RIG_PRESETS.get(a.rig, {})
            hfov = a.hfov or preset.get("hfov") or 78.0
            fx, fy, cx, cy = pc.intrinsics_from_hfov(W, H, hfov)
            self.cfg = pc.CoreCfg(w=W, h=H, fx=fx, fy=fy, cx=cx, cy=cy, x_min=0.3, x_max=8.0,
                                  y_min=-4.0, y_max=4.0, res=0.1, robot_radius=a.robot_radius or 0.35)
            self.ncfg = ns.NavCfg(v_max=a.v_max or 1.0, w_max=a.w_max or 1.0, goal_tol=1.0)
            self.gmap = None
            self.pose_src = ns.NoPose()
        if a.height is not None:
            self.cfg.lock_height = a.height
        if a.pitch is not None:
            self.cfg.lock_pitch = math.radians(a.pitch)
        if a.roll is not None:
            self.cfg.lock_roll = math.radians(a.roll)
        if a.nominal_height is not None:
            self.cfg.nominal_height = a.nominal_height
        if a.dist:                       # camera/video rigs; the rover sends its own per frame
            self.cfg.dist = tuple(float(v) for v in a.dist.split(","))
        self.core = pc.PerceptionCore(self.cfg)
        self.nav = ns.Navigator(self.cfg, self.ncfg, self.gmap,
                                ns.PlannerCfg(robot_radius=self.cfg.robot_radius))
        if a.goal:
            gx, gy = (float(v) for v in a.goal.split(","))
            self.nav.set_goal(gx, gy)
        elif not self.has_pose:
            # no pose source: the goal is a carrot in the robot frame, so a plan
            # (and a cmd_vel) is always produced for the feed in front of the camera
            self.nav.set_goal(self.cfg.x_max - 1.0, 0.0)

        # ---- models -----------------------------------------------------------
        self.device = pc.pick_device()
        print(f"[server] device {self.device}, source {self.source_kind}, depth {self.depth_mode}")
        self.depth_models: dict = {}
        if self.depth_mode != "sim" or not self.has_true_depth:
            self._depth_model("metric" if self.depth_mode == "sim" else self.depth_mode)
        candidates = SEM_WEIGHTS.get(self.source_kind, SEM_WEIGHTS["camera"])
        weights = next((p for p in candidates if p.exists()), None)
        if a.sem_weights:
            weights = Path(a.sem_weights)
        if weights is None:
            raise SystemExit("no YOLO26 ADE20K semantic weights found; pass --sem-weights")
        print(f"[server] semantics: {weights.name}")
        self.sem = pc.SemanticModel(str(weights), self.device, imgsz=a.sem_imgsz)

        self.worker = threading.Thread(target=self._work, daemon=True)

    # -- models --------------------------------------------------------------
    def _depth_model(self, kind: str) -> pc.DepthModel:
        if kind not in self.depth_models:
            print(f"[server] loading depth model: {kind}")
            self.depth_models[kind] = pc.DepthModel(kind, self.device, res=self.a.depth_res)
        return self.depth_models[kind]

    # -- config message -------------------------------------------------------
    def config_msg(self) -> dict:
        c = self.cfg
        return dict(type="config", source=self.source_kind, has_pose=self.has_pose, depth_mode=self.depth_mode,
                    depth_modes=["metric", "metric-indoor", "relative", "affine"] + (["sim"] if self.has_true_depth else []),
                    v_max=self.ncfg.v_max, w_max=self.ncfg.w_max, robot_radius=c.robot_radius,
                    grid=dict(x_min=c.x_min, x_max=c.x_max, y_min=c.y_min, y_max=c.y_max, res=c.res),
                    goal_frame="world" if self.has_pose else "robot", mode="auto" if self.mode_auto else "manual",
                    tunables={k: getattr(c, k) for k in TUNABLE})

    # -- worker ------------------------------------------------------------------
    def _work(self):
        t_last = time.perf_counter()
        while True:
            item = self.slot.get(timeout=0.25)
            if item is None:
                if self.nav.watchdog() and self.nav.goal is not None and self.last_nav_msg:
                    self._broadcast(dict(type="nav", status="STOPPED", cmd=dict(v=0.0, omega=0.0),
                                         twist=rm.twist(0.0, 0.0), seq=self.last_seq, t=time.time() * 1000,
                                         note="watchdog: no frames", source=self.source_kind, has_pose=self.has_pose,
                                         mode="auto" if self.mode_auto else "manual", goal=self._goal_dict()))
                continue
            try:
                msg = self._process(item)
            except Exception as e:      # keep serving; report the failure
                import traceback
                traceback.print_exc()
                msg = dict(type="nav", status="ERROR", cmd=dict(v=0.0, omega=0.0), twist=rm.twist(0, 0),
                           note=f"{type(e).__name__}: {e}", seq=item["header"].get("seq"), t=time.time() * 1000,
                           source=self.source_kind, has_pose=self.has_pose)
            now = time.perf_counter()
            self.fps = 0.8 * self.fps + 0.2 * (1.0 / max(now - t_last, 1e-3))
            t_last = now
            msg["fps"] = round(self.fps, 1)
            self.last_nav_msg = msg
            self._broadcast(msg)

    def _goal_dict(self):
        return None if self.nav.goal is None else dict(x=self.nav.goal[0], y=self.nav.goal[1])

    def _process(self, item) -> dict:
        a = self.a
        prof = Profile()
        hdr, bgr, sim_depth = item["header"], item["bgr"], item["depth"]
        seq = int(hdr.get("seq", 0))
        if seq < self.last_seq:
            print("[server] frame sequence regressed: resetting map, goal and plane")
            self._reset()
        self.last_seq = seq

        # ---- per-frame config (sim sends exact intrinsics) ----------------------
        cfg = self.cfg
        H, W = bgr.shape[:2]
        if "fx" in hdr:
            cfg.w, cfg.h, cfg.fx, cfg.fy, cfg.cx, cfg.cy = W, H, float(hdr["fx"]), float(hdr["fy"]), float(hdr["cx"]), float(hdr["cy"])
        elif (cfg.w, cfg.h) != (W, H):
            sx = W / cfg.w
            cfg.fx *= sx; cfg.fy *= sx; cfg.cx = W / 2; cfg.cy = H / 2; cfg.w, cfg.h = W, H
        if hdr.get("dist"):
            cfg.dist = tuple(float(v) for v in hdr["dist"])
        # Rectify BEFORE anything reads geometry: backproject_optical is a pure pinhole
        # model, so uncorrected barrel distortion bows the ground plane upward at the
        # image edges and invents LETHAL cells along both sides of the path. A source
        # that sends no coefficients (the simulator) is returned untouched.
        bgr = pc.undistort(bgr, cfg)
        pose = None
        if self.has_pose and hdr.get("pose"):
            p = hdr["pose"]
            self.pose_src.set(p["x"], p["y"], p["theta"])
            pose = self.pose_src.get()
        if hdr.get("mode") in ("auto", "manual") and self.source_kind == "sim":
            self.mode_auto = hdr["mode"] == "auto"
        prof.lap("decode")

        # ---- depth ------------------------------------------------------------------
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # bgr is already rectified above
        warnings = []
        depth_mode = self.depth_mode
        if depth_mode == "sim":
            if hdr.get("depth_valid") is not None and hdr["depth_valid"] < 0.2:
                warnings.append(f"sim depth mostly empty ({hdr['depth_valid']:.0%} valid)")
            if sim_depth is None:
                warnings.append("no sim depth in frame; using metric model")
                depth_mode = "metric"
            else:
                depth = sim_depth
                if depth.shape != (H, W):
                    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
                metric, depth_arr_kind = True, "metric"
        if depth_mode != "sim":
            dm = self._depth_model(depth_mode)
            depth = dm(rgb, smooth=a.depth_smooth)
            metric, depth_arr_kind = dm.is_metric, dm.depth_kind
        prof.lap("depth")

        # ---- semantics --------------------------------------------------------------
        labels = self.sem(bgr)
        sem_cost = self.sem.cost(labels)
        prof.lap("sem")

        # ---- costmap (self-calibrating) ---------------------------------------------
        res = self.core.process(depth, sem_cost, depth_is_metric=metric, depth_kind=depth_arr_kind)
        warnings += res.warnings
        grid = res.grid
        prof.lap("core")

        # ---- visual odometry ----------------------------------------------------------
        # Placed AFTER core.process because it needs this frame's fitted plane: the whole
        # method is a bird's-eye warp through that plane, which is what makes the result
        # metric without a stereo baseline or an IMU.
        odo = None
        if self.vo is not None:
            if res.plane.ok:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                ground = (sem_cost >= 0) & (sem_cost <= pc.GROUND_COST_MAX)
                odo = self.vo.update(gray, cfg, res.plane, ground)
                if odo.ok:
                    self.pose_src.integrate(odo.dx, odo.dy, odo.dtheta, odo.confidence)
                else:
                    self.pose_src.miss()
            else:
                self.vo.reset()
                self.pose_src.miss()
            pose = self.pose_src.get()
            if pose is None and odo is not None and not odo.ok and odo.why:
                warnings.append(f"odometry: {odo.why}")
        prof.lap("vo")

        # ---- navigation -------------------------------------------------------------
        out = self.nav.step(grid, pose)
        v, w = out.v, out.omega
        if not self.mode_auto and self.source_kind == "sim":
            v, w = 0.0, 0.0             # manual: the human drives; still report the plan
        prof.lap("nav")

        # ---- render -----------------------------------------------------------------
        plane = res.plane.as_dict()
        if hdr.get("cam_height") is not None and res.plane.ok:
            plane["mount_err"] = dict(height=round(res.plane.height - float(hdr["cam_height"]), 3),
                                      pitch_deg=round(math.degrees(res.plane.pitch) - math.degrees(float(hdr.get("cam_pitch", 0.0))), 2))
        status_txt = f"{out.status}" + (f"  {out.note}" if out.note else "")
        extra = [f"{k}: {v_}" for k, v_ in prof.d.items()] if a.profile else []
        cm_img = pc.render_costmap(grid, cfg, scale=3, path=out.local_path, goal=out.local_goal, aim=out.aim,
                                   cmd=(out.v, out.omega), status=status_txt, plane=res.plane, extra_lines=warnings)
        sw = 320
        if seq % 2 == 0 or not getattr(self, "_thumbs", None):
            cam_small = cv2.resize(self.sem.overlay(bgr, labels, 0.35), (sw, int(sw * H / W)))
            depth_small = cv2.resize(pc.render_depth(depth, cfg.max_depth), (sw, int(sw * H / W)))
            self._thumbs = (b64img(cam_small, "jpeg", 70), b64img(depth_small, "jpeg", 65))
        # Full-resolution views for the standalone MJPEG endpoints. These are the same
        # arrays the dashboard gets as base64 thumbnails - encoded once more here at full
        # size so /camera and /costmap can be opened in a bare browser tab, or pulled into
        # OBS or a projector, with no dashboard and no WebSocket client.
        def _jpg(img, q=80):
            ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            return b.tobytes() if ok else None
        streams = {
            "camera":  _jpg(bgr, 82),                                   # rectified, no overlay
            "overlay": _jpg(self.sem.overlay(bgr, labels, 0.35), 80),    # + semantic classes
            "depth":   _jpg(pc.render_depth(depth, cfg.max_depth), 78),
            "costmap": _jpg(cm_img, 88),                                 # the final output
        }
        images = dict(costmap=b64img(cm_img, "png"), camera=self._thumbs[0], depth=self._thumbs[1])
        glob = None
        if self.gmap is not None and pose is not None:
            # the global map changes slowly: render it every other frame
            if seq % 2 == 0 or not getattr(self, "_glob_png", None):
                self._glob_png = b64png_bytes(self.gmap.to_png(path_world=out.global_path, pose=pose,
                                                               goal=self.nav.goal, crop_m=a.map_view, scale=2))
            images["global"] = self._glob_png
            glob = dict(path_world=[[round(x, 2), round(y, 2)] for x, y in out.global_path[::2]],
                        meta=self.gmap.crop_meta(pose, a.map_view, 2))
        prof.lap("render")
        prof.d["total"] = round(sum(prof.d.values()), 1)

        with self.lock:
            self._streams = {k: v for k, v in streams.items() if v}
            self._stream_seq = seq
            self.last_grid = grid
            self.last_pose = pose
            self.last_cmd = (v, w)
            self.last_local_path = out.local_path_m
            self.last_global_path = out.global_path
        if self.windows:
            self._win_frames = dict(costmap=cm_img, camera=np.hstack([cam_small, depth_small]))
        if a.profile:
            print("[PROFILE] " + " | ".join(f"{k} {v_:6.1f}" for k, v_ in prof.d.items()) + f" ms  {self.fps:4.1f} fps  {out.status}")

        return dict(type="nav", seq=seq, t=time.time() * 1000, source=self.source_kind, has_pose=self.has_pose,
                    status=out.status, mode="auto" if self.mode_auto else "manual",
                    cmd=dict(v=round(v, 3), omega=round(w, 3)), twist=rm.twist(v, w),
                    goal=self._goal_dict(), pose=None if pose is None else pose.as_dict(),
                    dist_to_goal=None if out.dist_to_goal is None else round(out.dist_to_goal, 2),
                    plane=plane, depth_mode=depth_mode, scale=round(res.scale, 3),
                    odom=None if self.vo is None else dict(
                        self.pose_src.as_dict(),
                        tracked=0 if odo is None else odo.n_tracked,
                        why="" if odo is None or odo.ok else odo.why),
                    local=dict(path_m=[[round(x, 2), round(y, 2)] for x, y in out.local_path_m[::2]], reached=out.reached,
                               grid=dict(x_min=cfg.x_min, x_max=cfg.x_max, y_min=cfg.y_min, y_max=cfg.y_max, res=cfg.res)),
                    **{"global": glob}, images=images, profile=prof.d, warnings=warnings, note=out.note,
                    dropped=self.slot.dropped)

    def _reset(self):
        self.nav.reset()
        self.core.reset()
        for dm in self.depth_models.values():
            dm.prev = None
        if self.vo is not None:
            self.vo.reset()
            self.pose_src.reset()
        if self.source_kind == "rover" and not self.a.goal:
            # pose starts at the origin facing +X, so this is "1.5 m straight ahead of
            # wherever you were switched on" - a world goal that STAYS PUT as the rover
            # drives, unlike the robot-frame carrot a pose-less source needs.
            self.nav.set_goal(1.5, 0.0)
        if not self.has_pose and not self.a.goal:
            # A pose-less source plans towards a carrot in its OWN frame. nav.reset()
            # clears it, so without this a rover that reconnects sits in NO_GOAL for
            # ever, looking like a perception failure when nothing is actually wrong.
            self.nav.set_goal(self.cfg.x_max - 1.0, 0.0)

    # -- websocket ------------------------------------------------------------------
    def _broadcast(self, msg: dict):
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_all(msg), self.loop)

    async def _send_all(self, msg: dict):
        if not self.clients:
            return
        data = json.dumps(msg)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.pop(ws, None)

    async def ws_handler(self, request):
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=10.0)
        await ws.prepare(request)
        self.clients[ws] = "viewer"
        try:
            async for m in ws:
                if m.type == WSMsgType.BINARY:
                    if not self.pushes_frames:
                        continue
                    try:
                        header, bgr, depth = parse_frame(m.data)
                    except Exception as e:
                        await ws.send_str(json.dumps(dict(type="error", error=f"bad frame: {e}")))
                        continue
                    if bgr is None:
                        continue
                    self.slot.put(dict(header=header, bgr=bgr, depth=depth))
                elif m.type == WSMsgType.TEXT:
                    try:
                        cmd = json.loads(m.data)
                    except json.JSONDecodeError:
                        continue
                    await self._command(ws, cmd)
                elif m.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self.clients.pop(ws, None)
        return ws

    async def _command(self, ws, cmd: dict):
        t = cmd.get("type")
        if t == "hello":
            role = cmd.get("role", "viewer")
            self.clients[ws] = role
            if role in ("sim", "rover"):
                print(f"[server] {role} connected ({cmd.get('client', '?')}): resetting map and goal")
                self._reset()
            await ws.send_str(json.dumps(self.config_msg()))
            if self.last_nav_msg:
                await ws.send_str(json.dumps(self.last_nav_msg))
        elif t == "set_goal":
            self.nav.set_goal(float(cmd["x"]), float(cmd["y"]))
            await self._send_all(dict(type="goal", goal=self._goal_dict()))
        elif t == "clear_goal":
            self.nav.clear_goal()
            await self._send_all(dict(type="goal", goal=None))
        elif t == "set_mode":
            self.mode_auto = bool(cmd.get("auto", True))
            await self._send_all(dict(type="mode", mode="auto" if self.mode_auto else "manual"))
        elif t == "reset":
            self._reset()
            await self._send_all(dict(type="goal", goal=None))
        elif t == "set_depth":
            mode = cmd.get("mode")
            if mode in ("metric", "metric-indoor", "relative", "affine") or (mode == "sim" and self.has_true_depth):
                self.depth_mode = mode
                self.core.reset()
                await self._send_all(self.config_msg())
        elif t == "set_param":
            name, val = cmd.get("name"), cmd.get("value")
            if name in TUNABLE:
                cur = getattr(self.cfg, name)
                setattr(self.cfg, name, type(cur)(val))
                await self._send_all(self.config_msg())

    # -- http -----------------------------------------------------------------------
    async def index(self, request):
        return web.FileResponse(HERE / "dashboard" / "index.html")

    async def ros(self, request):
        what = request.match_info["what"]
        with self.lock:
            grid, pose, (v, w) = self.last_grid, self.last_pose, self.last_cmd
            lpath = getattr(self, "last_local_path", [])
            gpath = getattr(self, "last_global_path", [])
        if what == "occupancy_grid" and grid is not None:
            return web.json_response(rm.occupancy_grid(grid, self.cfg.res, (self.cfg.x_min, self.cfg.y_min), "base_link"))
        if what == "global_grid" and self.gmap is not None:
            return web.json_response(rm.occupancy_grid(self.gmap.grid, self.gmap.res, (self.gmap.origin, self.gmap.origin), "map"))
        if what == "odometry":
            return web.json_response(rm.odometry(pose or ns.Pose(), v, w))
        if what == "cmd_vel":
            return web.json_response(rm.twist(v, w))
        if what == "path":
            pts = gpath if gpath else lpath
            return web.json_response(rm.path_msg(pts, "map" if gpath else "base_link"))
        return web.json_response(dict(error=f"unknown or not ready: {what}"), status=404)

    STREAMS = {
        "costmap": "the final output - traversability grid, plan and drive command",
        "camera":  "the rover's live camera, lens-rectified, no overlay",
        "overlay": "camera + semantic segmentation classes",
        "depth":   "depth, colour-mapped to max_depth",
    }

    async def mjpeg(self, request):
        """One view as multipart/x-mixed-replace - a plain <img src> in any browser.

        Deliberately NOT the dashboard: no WebSocket, no JavaScript, nothing to keep in
        sync. Opening http://<host>:<port>/camera in a tab is enough, which also makes
        these usable as a projector view or an OBS browser source during a demo.
        Each client is served the newest processed frame and never a backlog, so a slow
        viewer falls behind in quality of experience, never in latency.
        """
        what = request.match_info["what"]
        if what not in self.STREAMS:
            return web.json_response(dict(error=f"unknown stream {what}",
                                          available=list(self.STREAMS)), status=404)
        resp = web.StreamResponse(status=200, headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Connection": "close",
        })
        await resp.prepare(request)
        last = -1
        try:
            while True:
                with self.lock:
                    seq, buf = self._stream_seq, self._streams.get(what)
                if buf is not None and seq != last:
                    last = seq
                    await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(buf)).encode() +
                                     b"\r\n\r\n" + buf + b"\r\n")
                else:
                    await asyncio.sleep(0.01)
        except (ConnectionResetError, asyncio.CancelledError, RuntimeError):
            pass            # viewer closed the tab; nothing to clean up
        return resp

    async def streams_index(self, request):
        rows = "".join(
            f'<li><a href="/{k}">/{k}</a> &mdash; {v}</li>' for k, v in self.STREAMS.items())
        return web.Response(content_type="text/html", text=(
            "<!doctype html><meta charset=utf-8><title>streams</title>"
            "<style>body{font:15px/1.6 system-ui;margin:2rem;max-width:46rem}"
            "a{color:#06c}code{background:#eee;padding:.1em .3em;border-radius:3px}</style>"
            "<h2>Perception streams</h2><ul>" + rows + "</ul>"
            "<p>Each is a plain MJPEG stream &mdash; open it in a tab, embed it with "
            "<code>&lt;img src=\"/camera\"&gt;</code>, or point OBS at it. "
            "The full dashboard is at <a href=\"/\">/</a>.</p>"))

    async def status(self, request):
        return web.json_response(dict(config=self.config_msg(), fps=self.fps, clients=len(self.clients),
                                      last=None if not self.last_nav_msg else {k: v for k, v in self.last_nav_msg.items() if k != "images"}))

    # -- OpenCV windows must run on the main thread on macOS -----------------------------
    def _window_tick(self):
        if self._win_frames:
            for name, img in self._win_frames.items():
                cv2.imshow(name, img)
            self._win_frames = {}
        if (cv2.waitKey(1) & 0xFF) == 27:
            print("[server] ESC in window: exiting")
            os._exit(0)
        self.loop.call_later(0.03, self._window_tick)

    # -- run ----------------------------------------------------------------------------
    def run(self):
        app = web.Application(client_max_size=32 * 1024 * 1024)
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/status", self.status)
        app.router.add_get("/streams", self.streams_index)
        app.router.add_get("/{what:costmap|camera|overlay|depth}", self.mjpeg)
        app.router.add_get("/ros/{what}", self.ros)
        app.router.add_static("/dashboard", HERE / "dashboard")

        async def on_startup(app_):
            self.loop = asyncio.get_running_loop()
            self.worker.start()
            if not self.pushes_frames:
                self.capture = CaptureSource(self.a.source, self.slot, size=(self.cfg.w, self.cfg.h),
                                             every=self.a.every, fps_cap=self.a.fps_cap)
                self.capture.start()
            if self.windows:
                self.loop.call_later(0.5, self._window_tick)
            print(f"[server] dashboard  http://localhost:{self.a.port}   ws://localhost:{self.a.port}/ws")

        app.on_startup.append(on_startup)
        web.run_app(app, host=self.a.host, port=self.a.port, print=None)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="'sim' (SLAM3D pushes frames), 'rover' (the Pi pushes frames), camera index, or video path")
    ap.add_argument("--depth", default="metric", choices=["metric", "metric-indoor", "relative", "affine", "sim"],
                    help="metric: Depth Anything V2 metric-outdoor (default); relative: 1/disp + nominal-height scale; "
                         "affine: solve BOTH affine unknowns from ground planarity (best on a fixed low rig); sim: renderer depth")
    ap.add_argument("--rig", default="macbook", choices=list(pc.RIG_PRESETS), help="intrinsics preset for camera/video sources")
    ap.add_argument("--dist", default=None, help="lens distortion 'k1,k2,p1,p2,k3' (camera/video sources; the rover sends its own)")
    ap.add_argument("--hfov", type=float, default=None, help="horizontal field of view in degrees (overrides --rig)")
    ap.add_argument("--height", type=float, default=None, help="LOCK camera height (m) instead of estimating it")
    ap.add_argument("--pitch", type=float, default=None, help="LOCK camera pitch (deg, nose-down positive)")
    ap.add_argument("--roll", type=float, default=None, help="LOCK camera roll (deg)")
    ap.add_argument("--nominal-height", type=float, default=None,
                    help="camera height (m) used to scale RELATIVE/AFFINE depth. Default 0.60 for a camera "
                         "rig, 0.17 for --source rover. This one ruler measurement is what gives the whole "
                         "map its metric scale.")
    ap.add_argument("--goal", default=None, help="initial goal 'x,y' (world if sim, robot frame otherwise)")
    ap.add_argument("--auto", action="store_true", help="start the sim in AUTO mode")
    ap.add_argument("--v-max", dest="v_max", type=float, default=None)
    ap.add_argument("--w-max", dest="w_max", type=float, default=None)
    ap.add_argument("--robot-radius", dest="robot_radius", type=float, default=None)
    ap.add_argument("--map-view", dest="map_view", type=float, default=60.0, help="global map crop shown, metres")
    ap.add_argument("--depth-res", dest="depth_res", type=int, default=336, help="depth model input (multiple of 14; 252/280 faster)")
    ap.add_argument("--proc-width", dest="proc_width", type=int, default=640, help="processing width for camera/video sources (16:9)")
    ap.add_argument("--depth-smooth", dest="depth_smooth", type=float, default=0.0, help="EMA on the depth map (0 = off)")
    ap.add_argument("--sem-weights", dest="sem_weights", default=None)
    ap.add_argument("--sem-imgsz", dest="sem_imgsz", type=int, default=640)
    ap.add_argument("--every", type=int, default=1, help="process every Nth captured frame")
    ap.add_argument("--fps-cap", dest="fps_cap", type=float, default=None, help="limit capture rate (video/webcam)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--windows", action="store_true", help="also show OpenCV windows (needs a display)")
    ap.add_argument("--profile", action="store_true", help="print per-stage milliseconds")
    a = ap.parse_args()
    if a.depth == "sim" and a.source != "sim":
        ap.error("--depth sim requires --source sim")
    PerceptionServer(a).run()


if __name__ == "__main__":
    main()
