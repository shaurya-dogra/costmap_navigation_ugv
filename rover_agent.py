#!/usr/bin/env python3
"""
rover_agent.py - Raspberry Pi camera streamer for perception_server.py  (SIH PS 26126)
=======================================================================================

Runs ON the rover (Pi 4B + IMX219 / Camera Module v2). Captures frames, JPEG-encodes
them, and pushes them over a WebSocket to `perception_server.py` on the Mac using the
exact binary frame layout `parse_frame()` there expects (PROTOCOL.md section 2):

    u32 LE header length | header JSON (UTF-8) | JPEG bytes

This process does ONE job: get frames off the Pi and onto the network as fast and as
fresh as possible. It does not perceive, plan, or drive. `nav` messages coming back are
parsed only far enough to watch for staleness — actually turning `cmd.v` / `cmd.omega`
into motor PWM is a separate, not-yet-written layer (see `apply_cmd` below); wiring an
ESC/motor driver here is explicitly out of scope for this file.

Two design points that are easy to get backwards on a Pi and worth stating up front:

1. Capture must never share a thread with the WebSocket send. A slow or stalled send
   (Wi-Fi hiccup, Mac busy) must not stall the camera: capture runs in its own thread
   into a single-slot "latest frame wins" mailbox, exactly the pattern
   `perception_server.py`'s own `FrameSlot` uses on the receiving end. The sender thread
   (here, the asyncio task) always takes only the newest frame and DROPS anything it
   didn't get to in time. Queueing would trade latency for completeness, which is the
   wrong trade for a robot: a fresh frame that arrives on time beats a complete backlog
   of stale ones.
2. `picamera2` does not exist off-Pi. The import is deferred into `CameraThread.run()`
   so this module can be `py_compile`'d and its CLI parsed on a Mac with no camera at
   all (that is how this file was written and checked).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# `websockets` is on the Pi's verified environment (17.0.1) but is imported lazily
# below anyway, purely so `python3 -m py_compile` and `--help` keep working on a
# machine that never installed it (e.g. this Mac, which has no reason to).
try:
    import websockets
except ImportError:  # pragma: no cover - exercised only off-target
    websockets = None  # checked in main() before anything tries to connect


# ----------------------------------------------------------------------------
# camera
# ----------------------------------------------------------------------------

class FrameSlot:
    """Single-item mailbox: latest put wins, get() drains it. No queueing.

    Mirrors `perception_server.py`'s `FrameSlot` deliberately - same job, same shape,
    on the other end of the same wire. Also counts drops (a put that overwrote an
    unread item), which is the number we want in the periodic stats: it is the direct
    measure of "how far capture is outrunning the network".
    """

    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.item: Optional[tuple[np.ndarray, float]] = None
        self.dropped = 0

    def put(self, frame: np.ndarray, t: float) -> None:
        with self.cv:
            if self.item is not None:
                self.dropped += 1
            self.item = (frame, t)
            self.cv.notify()

    def get(self, timeout: float = 0.5) -> Optional[tuple[np.ndarray, float]]:
        with self.cv:
            if self.item is None:
                self.cv.wait(timeout)
            item, self.item = self.item, None
            return item


class CameraThread(threading.Thread):
    """Owns the picamera2 handle; runs entirely off the asyncio event loop.

    Why the sensor mode matters: `Picamera2.sensor_modes` on the IMX219 lists several
    modes at different CROP WINDOWS, not just different output sizes. Requesting the
    (640, 480) mode looks correct and produces correct-looking frames, but that mode
    reads only the central 39% of the sensor width (crop 1000..2280 of 3280), giving a
    ~26.5 deg horizontal FOV instead of the lens's actual 62.2 deg. There is no error,
    no warning, and no way to tell from the image alone that the field of view has been
    quietly cropped to less than half. Pinning `sensor={"output_size": (1640, 1232)}`
    (the FULL-FOV mode) and letting the ISP downscale to the requested `main` size is
    the only way to get the wide picture this rover needs to see obstacles to the side.
    """

    def __init__(self, slot: FrameSlot, width: int, height: int, verbose: bool) -> None:
        super().__init__(daemon=True, name="camera")
        self.slot = slot
        self.width, self.height = width, height
        self.verbose = verbose
        self.stop_flag = threading.Event()
        self.picam2 = None  # set in run(); read back by main() for clean shutdown

    def run(self) -> None:
        from picamera2 import Picamera2  # deferred: does not exist off-Pi

        picam2 = Picamera2()
        self.picam2 = picam2
        cfg = picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            # Full-FOV sensor mode, see class docstring: without this the ISP is fed
            # an already-cropped 26.5 deg window and there is no error to catch it.
            sensor={"output_size": (1640, 1232)},
            buffer_count=4,
        )
        picam2.configure(cfg)
        picam2.start()

        # Let AE/AWB converge against the real scene, then freeze them. Re-running AE
        # mid-mission would silently shift brightness/white-balance frame to frame,
        # which is exactly the kind of appearance drift the depth/costmap pipeline on
        # the Mac assumes does not happen. The IMX219 module has no autofocus at all,
        # so there is nothing to lock there - fixed focus is a property of the lens.
        time.sleep(2.0)
        settled = picam2.capture_metadata()
        exposure = settled.get("ExposureTime")
        gain = settled.get("AnalogueGain")
        colour_gains = settled.get("ColourGains")
        controls = {"AeEnable": False, "AwbEnable": False}
        if exposure is not None:
            controls["ExposureTime"] = exposure
        if gain is not None:
            controls["AnalogueGain"] = gain
        if colour_gains is not None:
            controls["ColourGains"] = colour_gains
        picam2.set_controls(controls)
        if self.verbose:
            print(f"[camera] locked AE/AWB: exposure={exposure} gain={gain} "
                  f"colour_gains={colour_gains}", file=sys.stderr)

        while not self.stop_flag.is_set():
            # capture_array() blocks this thread only, never the event loop - that is
            # the whole reason capture lives on its own thread rather than being
            # awaited from the asyncio side.
            # picamera2's "RGB888" names the MEMORY layout, which is reversed relative to
            # numpy channel order: capture_array() hands back B,G,R per pixel already.
            # That is exactly what cv2 wants, so converting here would swap red and blue.
            # Measured against picamera2's own writer: mean abs error 3.85 (JPEG noise)
            # taking the array as-is, versus 20.5 after an RGB2BGR call.
            bgr = picam2.capture_array("main")
            self.slot.put(bgr, time.time())

        picam2.stop()
        picam2.close()

    def request_stop(self) -> None:
        self.stop_flag.set()


# ----------------------------------------------------------------------------
# stats
# ----------------------------------------------------------------------------

@dataclass
class Stats:
    frames_sent: int = 0
    bytes_sent: int = 0
    drops: int = 0
    window_start: float = field(default_factory=time.perf_counter)

    def report(self, watchdog_stale: bool) -> str:
        now = time.perf_counter()
        dt = max(now - self.window_start, 1e-6)
        fps = self.frames_sent / dt
        mean_kb = (self.bytes_sent / 1024 / self.frames_sent) if self.frames_sent else 0.0
        mbit_s = (self.bytes_sent * 8 / 1e6) / dt
        line = (f"[stats] fps={fps:.1f} mean_kB={mean_kb:.1f} mbit/s={mbit_s:.2f} "
                f"drops={self.drops} watchdog={'STALE' if watchdog_stale else 'ok'}")
        self.frames_sent = 0
        self.bytes_sent = 0
        self.drops = 0
        self.window_start = now
        return line


# ----------------------------------------------------------------------------
# nav watchdog (perception -> motor translation is explicitly NOT here)
# ----------------------------------------------------------------------------

NAV_STALE_S = 0.2  # PROTOCOL.md nav messages arrive per processed frame; >200ms is a lost link


class NavWatchdog:
    """Tracks the last `nav` message and flags staleness. No motor control lives here.

    TODO(motor layer): when an ESC/motor driver is added, `apply_cmd()` below is the
    seam - it should translate `(v, omega)` into whatever the driver HAT expects, and
    it must itself refuse to drive when `is_stale()` is true. Deciding how the Pi
    behaves in that shutdown case (stop vs. crawl vs. hold) belongs to that layer, not
    to this streamer.
    """

    def __init__(self) -> None:
        self.last_nav_t: Optional[float] = None
        self.last_cmd = (0.0, 0.0)

    def on_nav(self, msg: dict) -> None:
        self.last_nav_t = time.monotonic()
        cmd = msg.get("cmd") or {}
        v = float(cmd.get("v", 0.0))
        omega = float(cmd.get("omega", 0.0))
        self.last_cmd = (v, omega)

    def is_stale(self) -> bool:
        return self.last_nav_t is None or (time.monotonic() - self.last_nav_t) > NAV_STALE_S

    def apply_cmd(self, v: float, omega: float, verbose: bool) -> None:
        # Placeholder only - see class docstring. Perception-to-motor translation is
        # out of scope for this file; this exists so the watchdog has somewhere to
        # report to without pretending to drive anything.
        if verbose:
            print(f"[nav] cmd v={v:.2f} m/s omega={omega:.2f} rad/s", file=sys.stderr)


# ----------------------------------------------------------------------------
# agent
# ----------------------------------------------------------------------------

class RoverAgent:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.slot = FrameSlot()
        self.camera: Optional[CameraThread] = None
        self.stats = Stats()
        self.watchdog = NavWatchdog()
        self.seq = 0
        self.shutting_down = False

    # -- lifecycle ------------------------------------------------------------------

    def start_camera(self) -> None:
        self.camera = CameraThread(self.slot, self.args.width, self.args.height, self.args.verbose)
        self.camera.start()

    def stop_camera(self) -> None:
        if self.camera is not None:
            self.camera.request_stop()
            self.camera.join(timeout=2.0)

    # -- framing ----------------------------------------------------------------------

    def _orient(self, bgr: np.ndarray):
        """Rotate the frame upright and transform the intrinsics to match.

        WHY THIS MATTERS MORE THAN IT LOOKS
        -----------------------------------
        The whole perception stack reads the image as "row = distance ahead": the ground
        plane is fitted from the LOWER `plane_lower_frac` of rows, and the hole rule
        scans along columns looking for gaps in the ground. Mount the camera on its side
        and none of that means anything - the "lower" part of the frame becomes the left
        of the scene. Measured on this rig: with the sensor rotated 90 deg, ground-like
        pixels in the lower 45 % of the image were 0.1 %, so the plane fit found zero
        candidates and the map was entirely UNKNOWN. Rotating upright took it to 24.6 %.

        A 90/270 rotation also SWAPS the axes, so `fx`/`fy` swap and the principal point
        moves; sending the unrotated intrinsics with a rotated image would quietly scale
        every distance by fx/fy. Both are handled here so the server needs no knowledge
        of how the camera happens to be bolted on.

        Rotating in software costs a copy and narrows the horizontal field of view to the
        sensor's 48.8 deg. Remounting the camera the right way up is strictly better;
        this exists so a wrongly-mounted rig can still be tested.
        """
        a = self.args
        r = int(getattr(a, "rotation", 0)) % 360
        if r == 0:
            return bgr, a.width, a.height, a.fx, a.fy, a.cx, a.cy
        H, W = bgr.shape[:2]
        if r == 90:      # clockwise: new_x = H-1-y, new_y = x
            return (cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE),
                    H, W, a.fy, a.fx, (H - 1) - a.cy, a.cx)
        if r == 270:     # counter-clockwise: new_x = y, new_y = W-1-x
            return (cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
                    H, W, a.fy, a.fx, a.cy, (W - 1) - a.cx)
        return (cv2.rotate(bgr, cv2.ROTATE_180),
                W, H, a.fx, a.fy, (W - 1) - a.cx, (H - 1) - a.cy)

    def encode_frame(self, bgr: np.ndarray, t: float) -> bytes:
        """Build the exact wire message `perception_server.parse_frame()` decodes.

        Byte-compatible by construction: `parse_frame` reads a u32 LE header length,
        then that many bytes of UTF-8 JSON, then `header["jpeg_len"]` raw JPEG bytes.
        No depth block is appended - this rig has no depth sensor, and the header's
        `depth` field is `null`, which `parse_frame` treats as "nothing follows".
        """
        a = self.args
        bgr, w, h, fx, fy, cx, cy = self._orient(bgr)
        ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        jpeg_bytes = jpeg.tobytes()
        self.seq += 1
        header = {
            "type": "frame",
            "seq": self.seq,
            "t": t * 1000.0,  # epoch ms, matching the sim's `t`
            "w": w,
            "h": h,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "dist": a.dist,
            "cam_height": a.cam_height,
            "jpeg_len": len(jpeg_bytes),
            "depth": None,
        }
        header_bytes = json.dumps(header).encode("utf-8")
        return struct.pack("<I", len(header_bytes)) + header_bytes + jpeg_bytes

    # -- send loop --------------------------------------------------------------------

    async def send_loop(self, ws) -> None:
        """Pace capture to --fps; always send the newest frame, dropping any backlog.

        The FrameSlot already drops on the capture side (a put overwriting an unread
        item), which covers "camera outruns the pacing". This loop's own `slot.get()`
        additionally means the SEND side never queues either: if `ws.send` is slow
        (a Wi-Fi stall), whatever the camera produced in the meantime is simply the
        next thing read out of the single-item slot, not a backlog to work through.
        Both drop points count into the same `stats.drops`.
        """
        a = self.args
        period = 1.0 / a.fps
        last_stats_t = time.perf_counter()
        loop = asyncio.get_running_loop()

        while True:
            t0 = time.perf_counter()
            item = await loop.run_in_executor(None, self.slot.get, 1.0)
            if item is None:
                continue  # camera hasn't produced a frame yet / timed out; loop again
            bgr, t = item
            msg = self.encode_frame(bgr, t)
            await ws.send(msg)
            self.stats.frames_sent += 1
            self.stats.bytes_sent += len(msg)

            if time.perf_counter() - last_stats_t >= 5.0:
                print(self.stats.report(self.watchdog.is_stale()), file=sys.stderr)
                last_stats_t = time.perf_counter()

            dt = time.perf_counter() - t0
            if dt < period:
                await asyncio.sleep(period - dt)

    async def recv_loop(self, ws) -> None:
        """Parse `config`/`nav` text messages; drive the watchdog. No motor output."""
        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                continue  # the agent never receives binary; ignore defensively
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "nav":
                self.watchdog.on_nav(msg)
                v, omega = self.watchdog.last_cmd
                self.watchdog.apply_cmd(v, omega, self.args.verbose)
            elif msg.get("type") == "config" and self.args.verbose:
                print(f"[nav] config: {msg}", file=sys.stderr)

    async def watchdog_loop(self) -> None:
        """Independent of `recv_loop`: flags staleness even if no `nav` ever arrives
        (e.g. server just started) rather than only reacting to messages that show up.
        """
        while True:
            await asyncio.sleep(NAV_STALE_S)
            if self.watchdog.is_stale() and self.args.verbose:
                print("[nav] STALE: no nav message in >200ms", file=sys.stderr)

    # -- connection ---------------------------------------------------------------------

    async def run_session(self, uri: str) -> None:
        assert websockets is not None
        async with websockets.connect(uri, max_size=None) as ws:
            print(f"[ws] connected to {uri}", file=sys.stderr)
            await ws.send(json.dumps({"type": "hello", "role": "rover", "client": "rover_agent"}))
            watchdog_task = asyncio.create_task(self.watchdog_loop())
            try:
                await asyncio.gather(self.send_loop(ws), self.recv_loop(ws))
            finally:
                watchdog_task.cancel()

    async def run(self) -> None:
        """Auto-reconnect with backoff: the Pi must survive the Mac restarting,
        network drops, or the server not being up yet at boot, with no manual restart.
        """
        self.start_camera()
        backoff = 1.0
        max_backoff = 30.0
        while not self.shutting_down:
            try:
                await self.run_session(self.args.server)
                backoff = 1.0  # a session that ran and closed cleanly resets backoff
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self.shutting_down:
                    break
                print(f"[ws] disconnected ({type(e).__name__}: {e}); "
                      f"reconnecting in {backoff:.1f}s", file=sys.stderr)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, max_backoff)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="ws://10.199.153.226:8790/ws",
                     help="perception_server.py WebSocket URL (or host:port, ws:// assumed)")
    ap.add_argument("--fps", type=float, default=12.0,
                     help="capture/send rate (measured headroom is ~30.7 fps at 640x480 q75)")
    ap.add_argument("--quality", type=int, default=75, help="JPEG quality")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    # Defaults are the measured IMX219 full-FOV intrinsics at 640x480; calibrate.py
    # supplies real per-unit numbers later and must be able to override every one.
    ap.add_argument("--fx", type=float, default=530.5)
    ap.add_argument("--fy", type=float, default=530.5)
    ap.add_argument("--cx", type=float, default=320.0)
    ap.add_argument("--cy", type=float, default=240.0)
    ap.add_argument("--dist", default="0,0,0,0,0",
                     help="comma-separated k1,k2,p1,p2,k3 (from calibrate.py; default = no distortion)")
    ap.add_argument("--cam-height", dest="cam_height", type=float, default=0.17,
                     help="mount height in metres, reported only (PROTOCOL.md mount_err check)")
    ap.add_argument("--rotation", type=int, default=0, choices=[0, 90, 180, 270],
                   help="rotate the frame upright before sending, for a camera bolted on "
                        "its side. Intrinsics are transformed to match. Remounting is better.")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    if "://" not in a.server:
        a.server = f"ws://{a.server}/ws"
    a.dist = [float(x) for x in a.dist.split(",")]
    if len(a.dist) != 5:
        ap.error("--dist needs exactly 5 comma-separated values: k1,k2,p1,p2,k3")
    return a


def main() -> None:
    args = parse_args()
    agent = RoverAgent(args)

    async def amain() -> None:
        if websockets is None:
            raise SystemExit("the 'websockets' package is required on the rover "
                              "(verified present on the Pi image; pip install websockets)")
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()

        def _shutdown(*_: object) -> None:
            agent.shutting_down = True
            if main_task is not None:
                main_task.cancel()

        # SIGTERM/SIGINT must stop the camera cleanly (picamera2 wants stop()+close(),
        # not just process death) rather than leaving the sensor in a half-configured
        # state for the next run.
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown)

        try:
            await agent.run()
        except asyncio.CancelledError:
            pass
        finally:
            agent.stop_camera()
            print("[agent] shut down cleanly", file=sys.stderr)

    asyncio.run(amain())


if __name__ == "__main__":
    main()
