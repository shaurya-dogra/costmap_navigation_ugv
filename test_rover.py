"""
Verification of the rover-only pieces: ground-plane visual odometry (ground_vo.py) and
the Pi's ground-metered auto-exposure (rover_agent.GroundAE).
Needs only numpy + opencv. No models, no camera, no GPU, no Pi.

    python test_rover.py

The odometry is driven by RENDERED ground: a textured plane seen through the measured
rover rig (IMX219 at 640x480, fx 530.5, 0.17 m up, 12 deg down), warped exactly through
the same ground->image homography the real camera obeys, plus sensor noise. The true
trajectory is known, so every number below is an honest drift figure for the method -
an upper bound on how good it can be, since real ground adds blur, shadows and relief.
"""
import math, sys, time
import numpy as np
import cv2

import perception_core as pc
import navstack as ns
import ground_vo as gvo
import rover_agent as ra

FAILS = []


def check(name, got, ok):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
    print(f"  {tag}  {name:<60} {got}")


# ---------------------------------------------------------------- synthetic rig --
cfg = pc.rover_cfg()
H0, PITCH = 0.17, math.radians(12.0)
plane = pc.Plane(n=pc.normal_from_angles(PITCH, 0.0), d=H0, ok=True, confidence=1.0, source="fit")
TRES, TOFF, TN = 0.005, -10.0, 4000          # 20 m x 20 m of ground at 5 mm per texel
_rng = np.random.default_rng(0)
_tex = np.zeros((TN, TN), np.float32)
for sigma in (3, 9, 27):                     # gravel-to-patch scales
    _tex += cv2.GaussianBlur(_rng.random((TN, TN), dtype=np.float32), (0, 0), sigma) * sigma
TEX = cv2.normalize(_tex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
K = np.array([[cfg.fx, 0, cfg.cx], [0, cfg.fy, cfg.cy], [0, 0, 1.0]])
H_G2I = K @ plane.R.T @ np.diag([1.0, 1.0, -H0])


def render(x, y, th, tex=TEX, noise=2.0, rng=_rng):
    """The camera image of the ground with the robot at world pose (x, y, th)."""
    c, s = math.cos(th), math.sin(th)
    world_to_robot = np.array([[c, s, -(c * x + s * y)], [-s, c, -(-s * x + c * y)], [0, 0, 1.0]])
    tex_to_world = np.array([[TRES, 0, TOFF], [0, TRES, TOFF], [0, 0, 1.0]])
    img = cv2.warpPerspective(tex, H_G2I @ world_to_robot @ tex_to_world, (cfg.w, cfg.h))
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


def run(traj, **kw):
    vo, P, oks = gvo.GroundVO(), ns.VisualOdomPose(), 0
    kfs = 0
    for x, y, th in traj:
        before = vo.kf
        o = vo.update(render(x, y, th, **kw), cfg, plane)
        kfs += vo.kf is not before
        if o.ok:
            oks += 1
            P.integrate(o.dx, o.dy, o.dtheta, o.confidence)
        else:
            P.miss()
    x, y, th = traj[-1]
    dth = math.degrees(math.atan2(math.sin(P.pose.theta - th), math.cos(P.pose.theta - th)))
    return math.hypot(P.pose.x - x, P.pose.y - y), abs(dth), oks, kfs


def arc(v, w, n, hz=12.0):
    out, x, y, th, dt = [], 0.0, 0.0, 0.0, 1.0 / hz
    for _ in range(n):
        out.append((x, y, th))
        if abs(w) < 1e-9:
            x += v * dt * math.cos(th); y += v * dt * math.sin(th)
        else:                                # exact arc: chord at the mid-heading
            L = 2 * v / w * math.sin(w * dt / 2)
            x += L * math.cos(th + w * dt / 2); y += L * math.sin(th + w * dt / 2)
        th += w * dt
    return out


# -------------------------------------------------------------------- tests --
print("\n1. GROUND VO: DRIFT ON RENDERED GROUND (12 Hz)")
e, dth, oks, kfs = run([(0.0, 0.0, 0.0)] * 240)
check("stationary 20 s: position stays put (keyframe never replaced)", f"{e*1000:.2f} mm, {dth:.3f} deg, {kfs} keyframes", e < 0.002 and dth < 0.1 and kfs == 1)
e, dth, oks, kfs = run(arc(0.3, 0.0, 120))
check("straight 3 m at 0.3 m/s: < 1 % of distance", f"{e*100:.2f} cm ({e/3*100:.2f} %), {dth:.2f} deg, {kfs} keyframes", e < 0.03 and dth < 1.0)
e, dth, oks, kfs = run(arc(0.3, 0.5, 120))
check("3 m arc at 0.5 rad/s (286 deg of turn): < 1 % of distance", f"{e*100:.2f} cm, heading {dth:.2f} deg", e < 0.03 and dth < 1.5)
e, dth, oks, kfs = run([(0.0, 0.0, 0.0)] * 40 + [(0.0, 0.0, math.radians(1.0) * k) for k in range(1, 91)])
check("spin in place 90 deg: heading tracked", f"{dth:.2f} deg err, pos {e*100:.2f} cm", dth < 1.5 and e < 0.02)
e, dth, oks, kfs = run(arc(0.5, 0.0, 60), noise=6.0)
check("noisier sensor (sigma 6), 0.5 m/s: still < 1.5 %", f"{e*100:.2f} cm of 2.5 m", e < 0.0375)

print("\n2. GROUND VO: FAILURE IS REPORTED, NOT INVENTED")
flat = np.full_like(TEX, 128)
vo = gvo.GroundVO()
outs = [vo.update(render(0.02 * k, 0.0, 0.0, tex=flat), cfg, plane) for k in range(3)]
check("textureless floor -> not ok, with a reason", f"ok={[o.ok for o in outs]} why='{outs[-1].why}'", not any(o.ok for o in outs) and outs[-1].why)
vo = gvo.GroundVO()
o = vo.update(render(0, 0, 0), cfg, pc.Plane(n=plane.n, d=H0, ok=False))
check("no usable plane -> not ok", f"ok={o.ok} why='{o.why}'", not o.ok and "plane" in o.why)
vo = gvo.GroundVO(); vo.update(render(0, 0, 0), cfg, plane)
o = vo.update(render(0.9, 0.0, 0.0), cfg, plane)
check("an impossible jump (0.9 m in one frame) is rejected", f"ok={o.ok} why='{o.why}'", not o.ok)
o = vo.update(render(0.92, 0.0, 0.0), cfg, plane)
check("   ...and tracking re-anchors on the next frame", f"ok={o.ok} dx={o.dx*100:.1f} cm", o.ok and abs(o.dx - 0.02) < 0.005)

t = time.perf_counter()
vo = gvo.GroundVO()
frames = [render(0.025 * k, 0.0, 0.0) for k in range(20)]
t = time.perf_counter()
for f in frames:
    vo.update(f, cfg, plane)
ms = (time.perf_counter() - t) / len(frames) * 1000
check("VO step under 15 ms", f"{ms:.1f} ms", ms < 15)

print("\n3. GROUND-METERED AUTO-EXPOSURE (rover_agent.GroundAE)")
ae = ra.GroundAE(exposure_us=4000, gain=1.0)
check("inside the dead band -> no change", f"{ae.update(115.0)}", ae.update(115.0) is None)
c1 = ae.update(40.0)                      # into shade: far too dark
check("too dark -> brighter, but by at most one slew step", f"{c1}",
      c1 is not None and 4000 < c1["ExposureTime"] * c1["AnalogueGain"] <= 4000 * 1.15 + 1)
for _ in range(40):
    ae.update(10.0)
check("exposure time capped for blur; gain makes up the rest", f"exp={ae.exposure:.0f} us gain={ae.gain:.2f}",
      ae.exposure <= 8000 and ae.gain > 1.0 and ae.gain <= 8.0)
check("   ...and pinned at the limits it stops issuing changes", f"{ae.update(10.0)}", ae.update(10.0) is None)
ae = ra.GroundAE(exposure_us=4000, gain=1.0)
c2 = ae.update(250.0)
check("too bright (sun) -> darker, one slew step", f"{c2}", c2 is not None and c2["ExposureTime"] < 4000 and c2["AnalogueGain"] == 1.0)
img = np.zeros((120, 160, 3), np.uint8); img[:40] = 255      # bright sky on top, dark ground below
check("meters the ground band, not the sky", f"mean={ae.meter(img):.1f}", ae.meter(img) < 1.0)

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
