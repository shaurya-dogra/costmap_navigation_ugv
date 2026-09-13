# Rover plan — taking the costmap stack outdoors

**Status: the perception stream is built and running.** This document is the architecture,
what was measured on the real hardware, and what is left to build.

Hard constraint throughout: **the SLAM3D simulator demo must not regress.**
`test_perception_core.py`, `test_nav.py` and `test_geometry.py` pass at every step, and every
new config field defaults to the previous behaviour.

---

## 1. Context

The stack worked in the simulator, which quietly supplied four things hardware does not: true
depth, exact distortion-free intrinsics, ground-truth pose, and a camera 1 m up.

The real rover is an **RPi 4B** streaming a **monocular IMX219** (Camera Module v2) at
**~0.17 m** height, with perception on a Mac and, later, ORB-SLAM3 on a second laptop.
Chassis: **6 wheels, rigid (no suspension), skid/tank steering**, a 60 A brushed ESC per side
driving three motors each, **no wheel encoders**. GPS is ruled out — the problem statement
specifies a GPS-denied environment.

Perception→motor translation is deliberately out of scope for now. The ESC layer appears below
only where it constrains perception.

---

## 2. What is built and working

```
Pi (IMX219, 640x480 from the full-FOV sensor mode)
   --websocket--> Mac: undistort -> depth -> semantics -> ground plane
                       -> costmap -> A* -> pure pursuit
   <--websocket--  (v, omega)
```

Measured end to end: **10–12 fps, ~107 ms** (depth ~50, semantics ~24, core ~20, nav ~2,
render ~15). Observed running: `status: DRIVING`, plane fit with 4927 candidates / 1245 inliers.

```bash
python perception_server.py --source rover --depth metric-indoor --depth-res 280 --profile --port 8790
```

```bash
python3 rover_agent.py --server ws://<mac-ip>:8790/ws --fps 12 --rotation 90
```

**Shipped for this:** `rover_agent.py` (new, runs on the Pi); `--source rover` and four capability
flags (`pushes_frames`, `has_true_depth`, `has_pose`, `is_vehicle`) replacing the scattered
`source_kind == "sim"` tests; `rover_cfg()`; `undistort()`; `pixel_to_ground()`; the `affine` and
`metric-indoor` depth kinds; `PerceptionCore.process(depth_kind=...)`.

---

## 3. Measured hardware

| | measured |
|---|---|
| Board / OS | Pi 4B 4 GB, Debian 13 trixie, kernel 6.18 arm64, Python 3.13.5 |
| Root disk | USB SSD, 45 G free — sustained frame logging is viable |
| Power | `throttled=0x0`, 42.8 °C — clean, but **motors were idle**; recheck under load |
| Camera | IMX219, **rolling shutter**, **fixed focus** (there is no AF to lock) |
| Optics | 62.2° × 48.8° at full FOV |
| Intrinsics @640×480 | fx = fy = 530.5, cx = 320, cy = 240 |
| Pi capture + JPEG q75 | **30.7 fps, 39.8 kB/frame** — 2.5× the rate needed |
| Link (2.4 GHz hotspot) | **24.2 Mbit/s**, RTT **13.5 ms** (σ 4.6) — 6× the rate needed |
| GPIO | `gpiozero`, `RPi.GPIO`, `lgpio`, hardware `pwmchip`. **`pigpio` absent** |

Latency budget: encode ~35 ms + link ~7 ms + Mac perception ~90 ms + nav ~2 ms + command ~7 ms
≈ **140 ms**, so 11 cm of blind travel at 0.8 m/s.

### ⚠ The IMX219 sensor-mode trap

`sensor_modes` reports crop windows, and they are not all full-frame:

```
(640,  480)   crop=(1000, 752, 1280,  960)   <-- 39% of sensor width
(1640,1232)   crop=(   0,   0, 3280, 2464)   <-- FULL FOV  ** use this **
(1920,1080)   crop=( 680, 692, 1920, 1080)   <-- 58% of sensor width
```

Naively requesting 640×480 selects the **cropped** mode: hfov collapses from 62.2° to ~26.5°,
only ±0.59 m of lateral view at 2.5 m. No error is raised anywhere. Always pin
`sensor={"output_size": (1640, 1232)}` — `rover_agent.py` does.

---

## 4. The envelope

Ground sample spacing goes as `stride · r² / (fy · h)` — the expression already inside the hole
rule. At h = 0.17 m, fy = 530, stride 1: **1.1 cm at 1 m, 4.4 cm at 2 m, 18 cm at 4 m, 68 cm at
8 m.** The whole of 3→8 m lands in about 13 pixel rows, where one pixel of error is a
quarter-metre of range.

**Honest horizon: 0.2–2.6 m.** That is sufficient — at 0.8 m/s with ~140 ms latency and 1 m/s²
braking the rover stops in 0.5 m, a 5× margin — and it forces the correct Nav2 local/global
split: a small dense local map, with everything beyond it coming from memory.

`rover_cfg()`: 48 × 52 grid, x 0.20–2.60, y ±1.30, res 0.05, stride 1, `obstacle_h` 0.10,
`ditch_h` −0.08, `robot_radius` 0.20, `plane_gate_max` 0.05, `nominal_height` 0.17.
Hole-rule cutoff computes to 3.68 m, comfortably past the horizon, so negative-obstacle
detection is live across the whole map.

4:3 is deliberate: the binding constraint is how many ground *rows* the sensor gets, and vertical
FOV buys them. Cropping to 16:9 would discard a third of them for nothing.

**No lens purchase needed** — the IMX219's native 62.2° is already about right. **Tilt the camera
~12° down**: at 0° there is a 0.36 m blind zone larger than the rover, half the sensor is sky, and
`plane_lower_frac=0.65` leaks above-horizon pixels into the plane fit.

---

## 5. The affine depth correction

An earlier version of this plan claimed the central fix was to solve both affine unknowns from the
ground being flat, and called it the highest-confidence item here. **That was wrong, and testing
disproved it.**

Relative Depth Anything is affine-invariant in disparity: `1/Z = a·disp + b`. `DepthModel` taking
`1/max(disp,1e-3)` assumes `b == 0`, and when that is false the cloud is **warped, not merely
mis-scaled**. That half of the claim stands, emphatically — measured, `1/disp` loses the plane
completely (height 0.000, pitch 0.00).

But the proposed fix is not solvable. For a ground point,
`a·disp + b = m_x·xn + m_y·yn + m_z` with `m = -n/d`. The column multiplying `b` is all `+1`
and the column multiplying `m_z` is all `−1`, so `(0,1,0,0,1)` is an **exact null direction** —
adding the same delta to both changes nothing. Measured singular values of the design matrix:
`[341, 52.6, 35.8, 1.8e-06, 1.3e-13]`, two vanishing, null vector exactly `(0, .7071, 0, 0, .7071)`.

Physically: on coplanar points, a constant added to inverse depth is indistinguishable from the
plane sitting further away. Only `c = m_z − b` is observable. **A known camera height does not
rescue it** — it supplies one equation for two remaining unknowns.

Seeding orientation from the previous frame **self-confirms**: seeded at 12° against a true 10°,
it sat at 12.79° for eight iterations and never migrated.

| path | recovered pitch (true 10.00°) | height (true 0.1700) |
|---|---|---|
| **`metric`** | **10.00°** | **0.1700** |
| `affine` + seed | 12.79° (= the seed) | 0.1700 |
| `1/disp` | −0.00° | 0.0000 — plane lost |

**Use `metric` outdoors, `metric-indoor` on the bench.** The `affine` path is implemented and
documented but returns `(None, None, info)` without an external orientation constraint — one that
depth cannot bias, meaning **an IMU gravity vector or non-coplanar SLAM map points**. This raises
the value of both considerably: they are not accuracy niceties, they are what makes relative depth
usable at all.

---

## 6. Remaining architecture

```
 1. Undistort (K, dist)                                   [done]
 2. Dense depth                                           [done, metric]
 3. ORB-SLAM3 pose + sparse map points                    [to build]
 4. Affine (a,b) fitted against SLAM points               [blocked on 3]
 5. Backproject -> 3D points, each with sigma(r)          [to build]
 6. ACCUMULATE into a rolling 2.5D ELEVATION grid         [to build]
 7. Geometric traversability: slope, step, roughness      [partly: step + hole rule]
 8. Traversability net (RUGD-tuned -> self-supervised)    [to build]
 9. PROBABILISTIC fusion, range-dependent sigma           [to build]
10. Costmap + inflation + confidence channel              [done, no confidence]
11. Latency-compensated pose prediction                   [to build]
12. Global A* + DWA local planner                         [A* done]
13. Speed governor gated on perception confidence         [to build]
```

**Principle for 6: accumulate geometry, then classify — never the reverse.** Fusing *cost* grids
discards height before averaging; fusing *elevation* lets cost be re-derived at any time. Note
that a per-cell elevation grid is not supportable per-frame at this mount height (at 2 m a cell
gets ~1 sample row), which is exactly why it must accumulate across frames — each cell is
re-observed at decreasing range as the rover approaches.

**Uncertainty is the root gap.** `obstacle_h` is a constant while depth error grows as r², so one
threshold serves regimes differing 40× in noise. `plane.confidence` is computed and never used
beyond hold/reject. And `cost = max(semantic, geometry)` is a veto, not a fusion: depth noise
creates phantom obstacles nothing can overrule.

**Planner.** Skid-steer differential validates the existing A* + pure pursuit — `(v, ω)` are
independent and spin recovery is physically possible. A DWA sampling planner is an upgrade, not a
fix: rollouts are checked against the costmap, latency compensation folds in free, and
`cmd_smooth` can be deleted rather than tuned. Two caveats sharper on six wheels: slip makes
wheel-derived heading unusable, and **pure rotation starves monocular SLAM of parallax** — so
prefer arcs and pause map fusion while rotating.

---

## 7. Naming the destination, GPS-denied

**Click on the live camera view** — primary. Cast a ray through the clicked pixel, intersect the
fitted ground plane, get a metric robot-frame point. `pixel_to_ground()` is written; the dashboard
handler and `set_goal_px` command are not. Needs no map, no localisation, no GPS.

**Teach-and-repeat** — drive the route manually recording breadcrumbs, then replay as a sequence
of world goals while the local costmap avoids obstacles. Degrades gracefully under SLAM drift
because each leg is short.

---

## 8. What to do next

**Immediately, no code:** put the rover on the floor facing open ground and re-run. The plane
should read ≈0.17 m at the mount tilt — that single comparison validates intrinsics, rotation and
scale together. Current bench readings (height 0.444 m, pitch −16.5°, confidence 0.139) are the
rover sitting on a table pointed at a bedroom; the low confidence is the diagnostics working.

Then, roughly in order:

1. **Remount the camera upright.** It is currently 90° rotated — measured, ground-visible pixels in
   the band the plane fit reads went 0.1% → 24.6% after rotation. `--rotation 90` works but costs
   horizontal FOV.
2. Run `calibrate.py` at 640×480 with the full-FOV sensor mode; put real `dist` into the agent.
3. **Accuracy protocol** — a 15 cm box at 1.0/1.5/2.0 m, costmap centroid against a tape measure.
   Without this, threshold tuning is guesswork. This is the most valuable half-day available.
4. ORB-SLAM3 bridge (`role: "slam"` relay + `slam_bridge.py`) → pose → temporal fusion.
5. σ(r) thresholds, confidence channel, probabilistic fusion.
6. Elevation-grid accumulation.
7. Click-to-goal and teach-and-repeat.
8. Traversability net; start collecting driven-terrain logs now — they cost nothing to gather and
   self-supervised labels from your own traverses beat fine-tuning on public off-road datasets.

**Deferred with the motor layer:** `pigpio` is absent and has the tightest servo-pulse timing,
which is what a 1000–2000 µs ESC signal wants; `lgpio` with the hardware `pwmchip` should serve but
deserves a scope check. Six motors on 60 A ESCs can brown out a Pi sharing their rail — watch
`vcgencmd get_throttled` once they run.

---

## 9. Verification

- **Sim unchanged** (hard requirement): three suites green; `./run_sim.sh` identical.
- **Affine degeneracy**: assert the 5-parameter design matrix is rank-deficient and that
  `solve_affine_depth` refuses without a plane constraint. Locks in the finding above.
- **Undistort**: synthetic checkerboard through known distortion and back, residual straightness
  under a pixel.
- **`pixel_to_ground`**: round-trip against `backproject_optical`.
- **On hardware**: the box-at-known-range test in §8.3 — one number validating intrinsics, scale
  and plane fit together.
- **Rotation vs SLAM**: command a 180° spin, confirm fusion pauses rather than smearing the map.
- **Vibration**: drive over a 3 cm obstacle and inspect depth on impact frames, damped vs undamped
  mount. Rolling shutter with no suspension makes this a real failure mode.
