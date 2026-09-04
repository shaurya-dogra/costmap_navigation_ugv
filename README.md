# Vision-Based Autonomous Navigation for an Outdoor UGV — SIH PS 26126

> **Camera in → traversability costmap → Nav2-style planning → wheel commands out.**
>
> Perception AI for a ground robot in a GPS-denied outdoor environment. A single
> camera feed becomes a metric, self-calibrating top-down costmap; a global + local
> planner turns the costmap into `(v, ω)` drive commands. Runs against a Three.js
> rover simulation (the rover drives itself to a destination you click) and against
> the MacBook webcam placed on the ground.

---

## Contents

1. [What you get](#1-what-you-get)
2. [Quick start](#2-quick-start)
3. [How it works](#3-how-it-works)
4. [The 3D simulation demo](#4-the-3d-simulation-demo)
5. [Webcam mode](#5-webcam-mode)
6. [Nav2 compatibility](#6-nav2-compatibility)
7. [Configuration](#7-configuration)
8. [Tests](#8-tests)
9. [Performance](#9-performance)
10. [File overview](#10-file-overview)
11. [Known limitations](#11-known-limitations)
12. [Legacy prototype](#12-legacy-prototype)

---

## 1. What you get

| Piece | File | What it does |
|---|---|---|
| **Self-calibrating perception** | `perception_core.py` | Metric depth + semantic segmentation → per-frame ground plane (camera **height, pitch, roll are measured, not configured**) → 10 cm cost grid |
| **Nav2-style planning** | `navstack.py` | Global costmap with memory, coarse global A\*, carrot hand-off, local A\* + pure pursuit, turn-in-place / blocked recovery, watchdog |
| **Server** | `perception_server.py` | One process for every mode: sim frames over WebSocket, webcam, or video. Broadcasts costmaps, plans and commands as JSON |
| **Dashboard** | `dashboard/index.html` | Camera + semantics, depth, local costmap, global map, plane estimate, goal input |
| **3D rover sim** | `sim3d/` (from [SLAM3D](https://github.com/Klick07/SLAM3D)) | React Three Fiber + Rapier rover on a 100 m outdoor course; streams its camera (RGB + true depth) and drives on the returned commands |
| **ROS shapes** | `ros_msgs.py` | `OccupancyGrid`, `Odometry`, `Path`, `Twist` as JSON, ready for rosbridge |

---

## 2. Quick start

```bash
# one-time
./setup_mac.sh                      # venv + pip install (Apple Silicon: MPS)
# the rover sim is in ./sim3d ; copy rover.glb road.glb tree.glb into sim3d/public/
# (Sketchfab assets, not in git)

# 3D demo: perception server + rover sim, opens the browser
./run_sim.sh                        # ground-truth depth from the renderer
./run_sim.sh --depth metric         # neural depth instead

# webcam: laptop on the ground, dashboard in the browser
./run_webcam.sh

# stop everything (servers, sim, ports)
./stop_all.sh
```

Then, in the sim: press **T** (or the HUD button) for AUTO, and click a destination on
the ground, on the course map (bottom-left) or on the global map. The rover plans and
drives; the HUD shows status, the measured camera pose, collisions, and both maps.

Requires macOS with a display for the webcam and the sim; the tests need neither.

---

## 3. How it works

```
frame (RGB [+ true depth from the sim])
  │
  ├─ depth       Depth Anything V2 Metric-Outdoor (metres)   | relative + nominal height | sim
  ├─ semantics   YOLO26 ADE20K (150 classes) → per-pixel cost via keyword table
  │
  ▼  perception_core.py
  back-project to the OPTICAL frame
  RANSAC ground plane on ground-labelled pixels, near field first, fixed inlier gate
      → camera height, pitch, roll   (measured every frame; --height/--pitch are optional LOCKS)
  rotate every point into a GROUND-ALIGNED robot frame (Z = height above ground)
  cost(cell) = max( semantic vote , positive obstacle , negative obstacle , hole )
  UNKNOWN never free · inflate by the robot radius
  │
  ▼  navstack.py
  GlobalCostmap.fuse   world frame, max-fusion, remembers everything ever seen
  plan_global          A* on a coarse boxed copy ≤ 1 Hz → world path
  carrot               first path point ~10 m ahead → local goal
  local A* + pure pursuit on the frame grid backfilled from global memory → (v, ω)
  Navigator            NO_GOAL / PLANNING / TURNING / DRIVING / BLOCKED / ARRIVED
```

### Why the geometry is self-calibrating

The first prototype trusted a hand-measured camera height and pitch and pinned the
ground to Z = 0 from those constants. On a laptop, a hand-held phone or a rover with
suspension those values are neither known nor constant, roll is never zero, and every
height threshold is then measured against the wrong datum — the webcam costmap did not
match the floor in front of it. Now the plane is fitted per frame; on synthetic scenes
it recovers unknown rigs to within 1 cm and 0.5°, and a 6° roll that used to produce
72 false lethal cells produces none. The sim's mounted camera (1.0 m, 15°) doubles as a
live accuracy check: the HUD shows the estimate against the mount.

### The cost rules (all conservative, all tested)

| Channel | Rule | Catches |
|---|---|---|
| Semantic vote | ≥ 25 % of a cell's points carry a lethal label → LETHAL; otherwise mean of the non-lethal points | water, walls, trees, people, vehicles |
| Tall-label check | a label that implies height (wall, tree, car…) on a cell the geometry measured **flat** is demoted to cost 150, never lethal; water stays lethal | mislabelled ground (a grey floor read as "wall") |
| Positive obstacle | enough points above 0.25 m over the fitted ground | rocks, logs, fences, bushes |
| Negative obstacle | enough points below −0.20 m within 9 m | kerb drops, trench walls |
| **Hole rule** | a run of cells with **no measurement at all**, with measured ground before **and** beyond it along the view, is a depression → LETHAL (ignores shadows of positive obstacles and honest sampling gaps) | trenches whose floor is hidden by their own lip |
| Evidence floors | < 3 points → UNKNOWN; geometry needs ≥ 2 agreeing points and ≥ 20 % of the cell | depth speckle |
| Fusion | `max` of everything; UNKNOWN is expensive for the planner but passable | the safety argument |

---

## 4. The 3D simulation demo

`sim3d/` (React Three Fiber + Rapier; upstream: [Klick07/SLAM3D](https://github.com/Klick07/SLAM3D)). The rover carries a camera at 1.0 m, pitched
15° down, 60° vertical FOV. Each capture (≤ 6 Hz, one frame in flight):

1. renders the scene from the rover's camera into an offscreen target (linear → sRGB
   corrected) → 640×360 JPEG;
2. renders a depth pass (`MeshDepthMaterial`, RGBA packing) → unpacked and linearised
   on the CPU → 320×180 u16 millimetres (the stand-in for a stereo camera);
3. packs header (intrinsics, pose, mode) + JPEG + depth into one binary WebSocket frame.

The server answers with `(v, ω)`; the rover applies each command for one control
period, then holds heading until the next (this is what removed the zig-zag).

**Course** (`src/nav/world.js`, seedable with `?seed=`): road with boulders, rubble
clusters, trees, fallen logs, bushes, signposts, two fences, **two trenches**, two
ponds and a puddle, mud and sand patches, a grassy mound. Every hazard is scored by a
ground-truth contact counter the perception never sees.

**Controls**: WASD drive (takes over from AUTO), Space brake, **T** AUTO/MANUAL, **C**
camera (chase / top / driver), click ground or maps to set the goal, X,Y box in the HUD.

**Frames**: nav world X = −three.z, Y = −three.x, θ = heading (CCW positive), converted
in exactly one place (`src/nav/frames.js`). Protocol in [PROTOCOL.md](PROTOCOL.md).

---

## 5. Webcam mode

```bash
./run_webcam.sh                 # = perception_server.py --source 0 --rig macbook --depth metric
```

Put the laptop on the ground, lid at about 90°. The dashboard shows the measured
camera height (expect ≈ 0.20–0.23 m), pitch and roll, and their confidence; tilt the
lid and watch them track while the map stays put. There is no pose source, so there
is no global map: goals are in the robot frame (click the local costmap) and the
command shown is what would be sent to a base.

Turn off Center Stage / auto-framing (it changes the focal length). The default
intrinsics assume a 78° horizontal FOV; pass `--hfov` or run `calibrate.py` for exact
values.

---

## 6. Nav2 compatibility

ROS 2 is not required. The stack mirrors Nav2's layout (global costmap + planner,
local costmap + controller, recovery behaviours) and emits Nav2-shaped messages:

| HTTP | Message |
|---|---|
| `GET /ros/occupancy_grid` | `nav_msgs/OccupancyGrid`, local grid, frame `base_link` |
| `GET /ros/global_grid` | `nav_msgs/OccupancyGrid`, world grid, frame `map` |
| `GET /ros/odometry` | `nav_msgs/Odometry` |
| `GET /ros/path` | `nav_msgs/Path` |
| `GET /ros/cmd_vel` | `geometry_msgs/Twist` |

These are exactly what a rosbridge publisher would send; swapping in real Nav2 later
means publishing them and reading `cmd_vel` back. The pose enters through
`navstack.PoseSource` — today ground truth from the sim, later the team's visual SLAM.

---

## 7. Configuration

`perception_server.py --help` lists everything. The important ones:

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `0` | `sim`, camera index, or video path |
| `--depth` | `metric` | `metric` (Depth Anything V2 metric-outdoor), `relative` (+ `--nominal-height`), `sim` (renderer depth) |
| `--rig` / `--hfov` | `macbook` / 78° | intrinsics for camera sources; the sim sends exact intrinsics |
| `--height --pitch --roll` | estimate | **lock** the camera pose instead of measuring it |
| `--v-max --w-max --robot-radius` | 2.0 / 0.8 / 1.0 (sim) | controller limits and inflation |
| `--port` | 8790 | dashboard + WebSocket |
| `--profile` | off | per-stage milliseconds |

Live tunables (dashboard or `set_param`): `obstacle_h`, `ditch_h`, `robot_radius`,
`sem_lethal_frac`, `min_cell_pts`, `plane_gate`, `plane_near_range`, `max_depth`.
Grid: sim 0.5–12 m × ±5 m; webcam 0.3–8 m × ±4 m; 0.1 m cells.

---

## 8. Tests

No models, no camera, no GPU; a few seconds each.

```bash
source .venv/bin/activate
python test_perception_core.py   # 63 checks: plane recovery, roll, step-down, trench, tall labels, hold, locks…
python test_nav.py               # 42 checks: fusion, global A*, carrot, memory, state machine, ROS shapes
python test_geometry.py          # legacy prototype, still green
python sim.py --validate         # legacy analytic simulator
```

---

## 9. Performance (Apple M4, MPS)

| Mode | depth | sem | core | nav | render | rate |
|---|---|---|---|---|---|---|
| sim, `--depth sim` (640×360) | 0.5 ms | 35–80 ms | ~20 ms | 1–10 ms (global replan ~80 ms at ≤ 1 Hz) | 35–55 ms | 4–6 Hz |
| video/webcam, `--depth metric` (1280×720) | 85–110 ms | 60–140 ms | 15–90 ms | – | 40–90 ms | 3–4 Hz |

The browser capture adds two offscreen renders per frame; keep the capture at ≤ 6 Hz.

---

## 10. File overview

```
perception_core.py      self-calibrating geometry + costmap + model wrappers
navstack.py             global costmap, global planner, carrot, Navigator
ros_msgs.py             Nav2-shaped message dicts
perception_server.py    aiohttp server: sources, worker, WebSocket, dashboard, /ros
dashboard/index.html    browser dashboard (any source)
PROTOCOL.md             wire protocol between sim, server and viewers
synth_scene.py          analytic scenes for tests and for driving the server without a browser
test_perception_core.py / test_nav.py     test suites
run_sim.sh / run_webcam.sh / stop_all.sh  one-command launchers and shutdown
costmap_prototype.py, sim.py, test_geometry.py, calibrate.py   legacy prototype (see §12)
sim3d/src/nav/*         config, frames, WebSocket link, capture, course + ground truth
sim3d/src/components/*  Environment, Vehicle, RobotCamera, GoalMarker, Hud, MiniMap
```

Models download to the HuggingFace cache on first run: `Depth-Anything-V2-Metric-Outdoor-Small-hf`
(and `-Small-hf` for `--depth relative`). Semantic weights `yolo26{n,s}-sem-ade20k.pt` are
looked for in `../object segmentation/` or next to the server (`--sem-weights`).

---

## 11. Known limitations

- **Single camera, no odometry in webcam mode**: no global map, goals are robot-relative.
- **Monocular metric depth degrades beyond ~8–10 m** and on textureless synthetic
  ground; the sim's `--depth sim` is the stand-in for a stereo rig at the same interface.
- **Hole rule range** depends on sampling density: with the sim's 320×180 depth it reaches
  ~7 m; a 22 cm webcam sees holes only within ~3 m (honest: it cannot see further).
- **Semantics on synthetic imagery** is approximate; the tall-label check protects
  drivable ground from mislabels, water/mud/sand grading depends on the segmenter.
- **Pure Python planners**: the global A\* is boxed and pooled to stay under ~100 ms.

---

## 12. Legacy prototype

`costmap_prototype.py` is the first, fixed-rig version (phone camera, hand-measured
height and pitch, OpenCV windows, A\* + pure pursuit to a carrot 8 m ahead). It is kept
as a reference and its suite `test_geometry.py` still passes; `sim.py` is its analytic
ray-cast test bench. New work goes through `perception_core.py` and `navstack.py`.
