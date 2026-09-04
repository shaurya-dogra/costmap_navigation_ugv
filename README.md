# Monocular Traversability Costmap — SIH PS 26126

> **Phone camera in → top-down traversability costmap out.**
>
> A real-time perception-to-costmap pipeline for ground robot navigation.
> Three parallel heads (depth, semantics, detection) fuse into a single
> 10 cm/cell occupancy grid at ~7 FPS on Apple M4. Designed so the only
> function that changes when stereo/LiDAR arrives is `Perception.depth`.

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [How the Costmap Works](#2-how-the-costmap-works)
3. [Architecture](#3-architecture)
4. [Quick Start](#4-quick-start)
5. [Installation](#5-installation)
6. [Getting a Camera Feed](#6-getting-a-camera-feed)
7. [Camera Calibration](#7-camera-calibration)
8. [Rig Geometry Setup](#8-rig-geometry-setup)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [CLI Reference](#10-cli-reference)
11. [Configuration Reference (`Cfg`)](#11-configuration-reference-cfg)
12. [Understanding the Output Windows](#12-understanding-the-output-windows)
13. [Semantic Cost Table](#13-semantic-cost-table)
14. [Depth Pipeline — v. fast depth](#14-depth-pipeline--v-fast-depth)
15. [Performance Benchmarks](#15-performance-benchmarks)
16. [Sanity Checks](#16-sanity-checks)
17. [Test Suite](#17-test-suite)
18. [Known Limitations](#18-known-limitations)
19. [File Overview](#19-file-overview)

---

## 1. What It Does

```
Phone Camera
     │
     ▼
┌─────────────────────────────────────────────┐
│              Perception Pipeline             │
│                                             │
│  Depth Anything V2 Small (336×336, MPS)    │  → metric depth map
│  YOLO26n ADE20K Semantics(640px,   MPS)    │  → per-pixel scene & object class
└─────────────────────────────────────────────┘
     │
     ▼
Ground-Plane Backprojection (pinhole → X,Y,Z robot frame)
     │
     ▼
Scale Recovery (camera height → ground anchored to Z = 0)
     │
     ▼
┌─────────────────────────────────────────────┐
│            Costmap Fusion                    │
│                                             │
│  A) Semantic cost  (per-pixel terrain &     │
│                     object classification)  │
│  B) Height cost    (positive obstacles &    │
│                     negative ditch geometry)│
│                                             │
│  cell_cost = max(A, B)                      │
│  unseen cells → UNKNOWN (never free)        │
└─────────────────────────────────────────────┘
     │
     ▼
10 cm/cell top-down grid  (0.5–10 m ahead, ±4 m lateral)
```

---

## 2. How the Costmap Works

The costmap is a **robot-centric top-down grid** where every 10 cm × 10 cm cell holds a single cost byte:

| Cost | Display Colour | Meaning |
|------|---------------|---------|
| `0`  | **Green** | Completely free — flat traversable ground |
| `1–99` | **Pale green** | Low cost — slightly rough or uncertain terrain |
| `100–199` | **Yellow** | Medium cost — rough terrain (sand, dirt, grass) |
| `200–253` | **Orange** | High cost — difficult but possibly passable |
| `254` | **Magenta/Pink** | **LETHAL** — wall, person, water, detected object, ditch |
| `255` | **Dark grey** | **UNKNOWN** — no measurement hit this cell |

### How each cost channel is computed

**Channel A — Semantics (a vote, never an average)**
Each pixel is classified into one of 150 ADE20K terrain classes and mapped to a
cost by lookup table (see [§13](#13-semantic-cost-table)). Unmatched classes
default to `100` (uncertain, not free).

Per cell the costs are **voted, not averaged**:

- if at least `sem_lethal_frac` (25%) of the cell's points — and no fewer than
  `geo_min_pts` of them — carry a lethal label, the cell is **LETHAL**
- otherwise the cell's cost is the mean of only its **non-lethal** points

Averaging the raw costs was wrong in both directions. A cell that is 40% wall
and 60% floor averages to 114 and reads as merely awkward terrain, when a wall
inside a 10 cm cell means the cell is impassable. And on a live indoor feed a
few high-cost pixels dragged every otherwise-clean floor cell above 50, so
**nothing was ever reported drivable**.

**Channel B — Height geometry**
Heights are measured against a fitted ground plane (see [§2.1](#21-the-ground-plane)),
and a cell needs real support before geometry may fire — at least `geo_min_pts`
(2) agreeing points and `geo_min_frac` (20%) of the cell's points:

- points above `obstacle_h` (default 0.25 m) → **LETHAL** positive obstacle
- points below `ditch_h` (default −0.20 m) → **LETHAL** negative obstacle / ditch
- points above `max_obstacle_h` (default 2.0 m) are ignored (overhead clearance)

A cell holding fewer than `min_cell_pts` (3) measurements is **not** evidence and
returns to `UNKNOWN`. Previously the consensus floor was a single point, so one
noisy depth sample could declare a cell lethal; on a live feed 544 of 1309
occupied cells held ≤2 points, which sprayed the map with the radial speckle
that made it unusable.

`rough_h` is still in `Cfg` but is **not used**: monocular depth noise
(σ ≈ 4–6 cm) exceeds real terrain roughness, so within-cell Z spread was not a
usable signal.

**Channel C — Object detections & instance masks**
YOLO26n-seg detects objects (people, vehicles, chairs, etc.) and predicts
pixel-perfect instance segmentation masks. Each 3D point belonging to a detected
object is directly projected into the costmap to stamp its true physical footprint
as **LETHAL** (with bounding-box cylinder stamping as fallback).

**Fusion**
`cost(cell) = max(sem_cost, height_cost, obj_cost)`
Any cell without enough measurements stays `UNKNOWN`.

### 2.1 The ground plane

Every height threshold is measured against a plane `Z = a·X + b·Y + c` fitted
per frame. The plane is the datum the whole geometry channel rests on, so it is
constrained hard:

| Constraint | Value | Why |
|---|---|---|
| depth-valid points only | `valid` mask | pixels with no depth backproject to a phantom surface at `(X=0, Z=cam_height)`, and a quarter of a frame is sky |
| near field only | `plane_near_x` 4.0 m | describes the ground the robot is about to drive on, not whatever fills the frame |
| fixed inlier gate | `plane_gate` 0.12 m | below `|ditch_h|`, so a step-down can never be absorbed as "ground" |
| intercept clamp | `plane_max_c` 0.15 m | `cam_height` is measured, so the ground under the robot is `Z=0` by construction |
| tilt rejection | `plane_max_slope` 0.36 (20°) | steeper is not ground; the fit has latched onto a wall or a person, so fall back to flat |

Each of these was added to kill a measured failure:

- **Fitting the whole frame** let the surface beyond a kerb lip outvote the road
  the robot stood on. Approaching a 0.45 m drop, `a` swung to −0.49 and `c` to
  −0.45, the step measured ≈0 m of relative height, lethal cells collapsed from
  31% to 1.5%, and **the drop read as drivable at point-blank range**.
- **A MAD-scaled inlier threshold** treated the two surfaces of a step as the
  noise level and widened to include both, tilting the plane through the step.
- **No tilt bound** let a live indoor feed return `a = +1.30` (52°), which put
  the correctly-labelled floor at `Zr = −1.4 m` and turned flat drivable floor
  into **245 lethal "ditch" cells**.

---

## 3. Architecture

```
costmap_prototype.py
├── Cfg                  — all tunable parameters in one place
├── SEMANTIC_COST        — keyword→cost mapping for ADE20K labels
├── build_cost_lut()     — pre-builds a 150-element LUT from YOLO26 names
├── pick_device()        — selects MPS > CUDA > CPU
├── Perception           — loads and runs the neural models
│   ├── depth()          — Depth Anything V2 Small + temporal filter
│   ├── semantics()      — YOLO26n ADE20K (150 scene classes)
│   ├── render_semantics_overlay() — vibrant color overlay on camera
│   └── render_depth_colormap()    — Turbo colourmap on raw disparity
├── backproject()        — pinhole → robot-frame X,Y,Z
├── recover_scale()      — ground-plane scale anchoring
├── Costmap
│   ├── cell_index()     — 3D points → grid (ix, iy)
│   ├── build()          — per-frame grid construction
│   └── inflate()        — robot-radius obstacle inflation
├── render()             — top-down BGR image with pitch/height HUD
└── main()               — capture loop, live trackbars, profiling
```

---

## 4. Quick Start

```bash
# 1. Clone / navigate into the project
cd sih_prototype_costmap

# 2. One-time setup
./setup_mac.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Verify geometry is correct
export PYTORCH_ENABLE_MPS_FALLBACK=1
python test_geometry.py          # must print: ALL CHECKS PASSED

# 5. Run on iPhone via Continuity Camera
python costmap_prototype.py --source 0 --profile
```

> **Requires macOS with a display.** Metal (MPS), OpenCV windows, and
> Continuity Camera all need the desktop environment. Do not run in a
> headless shell.

---

## 5. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Platform notes

| Hardware | PyTorch backend | Notes |
|---|---|---|
| Apple M-series (M1/M2/M3/M4) | **MPS** (Metal) | Default, no extra steps. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` always. |
| NVIDIA GPU | **CUDA** | Replace `torch` with the CUDA wheel from [pytorch.org](https://pytorch.org). |
| CPU only | CPU | Works, but ~4× slower. Remove `to(device)` is not needed — falls through automatically. |

### Pinned versions (what was tested)

```
torch==2.14.0
torchvision==0.29.0
transformers==5.16.1
ultralytics==8.4.138
opencv-python==5.0.0.93
numpy==2.5.2
pillow==12.3.0
```

---

## 6. Getting a Camera Feed

| Setup | `--source` value | Notes |
|---|---|---|
| **iPhone via Continuity Camera** | `0` | macOS 13+. Plug in or keep on same Wi-Fi. Disable auto-HDR, zoom, and stabilisation. |
| **Recorded clip** | `path/to/clip.mp4` | Best for repeatable development. Use `--record out.mp4` to capture live. |
| **Android IP Webcam** | `http://192.168.x.x:8080/video` | Install "IP Webcam" app, start server, paste the URL. |
| **USB webcam** | `1`, `2`, … | OpenCV device index. Try incrementing if `0` is taken by Continuity Camera. |

**Important:** Lock phone to a fixed resolution and turn **off**:
- Auto-HDR
- Auto-zoom / optical zoom
- Video stabilisation

Any of these silently change the effective focal length mid-run, breaking the metric projection.

---

## 7. Camera Calibration

Without calibration the depth scale and ground projection are wrong — every
distance reading will be off.

```bash
source .venv/bin/activate
python calibrate.py --source 0
```

1. Print a **9×6 chessboard** (or display it on a second screen).
2. Move the board to ~20 different positions and angles.
3. Press `s` to capture each view, `q` when done.
4. The script prints `fx fy cx cy`. Paste these into `Cfg` in `costmap_prototype.py`.

Default values (`fx=940, fy=940, cx=640, cy=360`) are reasonable for an
iPhone 15 at 1280×720 but will not be exact for your device.

---

## 8. Rig Geometry Setup

Measure these two values with a tape and set them in `Cfg`:

```python
cam_height = 0.60   # metres: vertical distance from ground to lens centre
cam_pitch  = np.deg2rad(12.0)   # radians: positive = nose-down tilt
```

Mount the phone **rigidly**. A handheld phone changes both values every frame
and the ground projection will drift.

The `recover_scale()` function corrects the monocular depth scale each frame
by constraining the visible ground to `Z = 0`. This only works reliably when:
- At least 500 ground-labelled pixels are within 8 m
- The rig geometry is set correctly

Scale correction is clamped to the range `[0.4, 2.5]×` — anything outside
that means something is badly wrong with the rig setup.

---

## 9. Running the Pipeline

### Basic run

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python costmap_prototype.py --source 0
```

### With profiling (recommended during development)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python costmap_prototype.py --source 0 --profile
```

Sample output:
```
[PROFILE] depth:  60.0ms | sem:  42.0ms | obj:   8.0ms | costmap:  30.0ms | render:   3.0ms | total: 143.0ms ( 7.0 fps)
```

### Record live camera, then replay

```bash
# Record raw stream to file
python costmap_prototype.py --source 0 --record session.mp4

# Replay the recording for repeatable testing
python costmap_prototype.py --source session.mp4 --profile
```

### Skip frames to reduce CPU load

```bash
# Process every 2nd frame (halves load, keeps display at live rate)
python costmap_prototype.py --source 0 --every 2
```

---

## 10. CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Camera index, video file path, or stream URL |
| `--every N` | `1` | Process every Nth frame (1 = all frames) |
| `--record FILE` | off | Record raw camera stream to `.mp4` before any processing |
| `--profile` | off | Print per-stage millisecond timing to stdout every frame |
| `--depth-res N` | `336` | Depth model input resolution (default: 336) |
| `--pitch DEG` | `12.0` | Camera downward pitch in degrees (e.g. `0.0` for horizontal on desk/floor) |
| `--height M` | `0.60` | Camera height in metres (e.g. `0.10` for testing on floor/books) |

### Runtime Interactive Tuning Trackbars

The `costmap` window includes interactive OpenCV sliders that update the 3D ground geometry **instantly in real time**:
- **`pitch (deg)`**: 0° to 45° downward tilt (slide to `0` if phone is sitting level on a floor or table).
- **`cam_h (cm)`**: 4 cm to 150 cm lens height (slide to `10` or `12` if testing on the floor).
- **`obs_h (cm)`**: positive obstacle lethal threshold (default: 25 cm).
- **`ditch_h (cm)`**: negative obstacle / ditch lethal threshold (default: 20 cm).

Adjusting `pitch` and `cam_h` immediately aligns the ground plane to $Z = 0$, turning flat floor green without restarting the program.

---

## 11. Configuration Reference (`Cfg`)

All parameters live in the `Cfg` class in `costmap_prototype.py`.
Edit them there — no external config file needed.

### Camera intrinsics

```python
fx = 940.0   # horizontal focal length (pixels) — from calibrate.py
fy = 940.0   # vertical focal length (pixels)
cx = 640.0   # principal point x (pixels, usually proc_w / 2)
cy = 360.0   # principal point y (pixels, usually proc_h / 2)
proc_w, proc_h = 1280, 720   # all frames are resized to this
```

### Rig geometry

```python
cam_height = 0.60          # metres above ground to lens centre
cam_pitch  = np.deg2rad(12.0)  # downward tilt, positive = nose down
```

### Costmap grid

```python
x_min, x_max = 0.5, 10.0  # metres ahead (robot is at x=0)
y_min, y_max = -4.0, 4.0  # metres left (+) and right (-)
res          = 0.10        # cell size in metres
```

### Geometry thresholds

```python
obstacle_h   = 0.25   # metres above ground → LETHAL positive obstacle
ditch_h      = -0.20  # metres below ground → LETHAL negative obstacle
rough_h      = 0.15   # PRESENT BUT UNUSED (depth noise > real roughness)
max_obstacle_h = 2.0  # ignore points above this height (ceiling clearance)
```

### Ground-plane fit

```python
plane_near_x    = 4.0   # fit only this far ahead, so a drop cannot re-anchor it
plane_gate      = 0.12  # fixed inlier band; must stay below |ditch_h|
plane_max_c     = 0.15  # max |intercept|: ground under the robot is Z = 0
plane_max_slope = 0.36  # tan(20°); a steeper fit is rejected as not-ground
```

### Evidence thresholds

```python
min_cell_pts    = 3     # fewer measurements than this → cell stays UNKNOWN
geo_min_pts     = 2     # agreeing points before geometry may call LETHAL
geo_min_frac    = 0.20  # ...and this share of the cell's points
sem_lethal_frac = 0.25  # share of lethal-labelled points that makes a cell lethal
```

### Other parameters

```python
robot_radius = 0.35   # robot footprint radius for obstacle inflation
max_depth    = 15.0   # ignore depth readings beyond this (metres)
LETHAL       = 254
UNKNOWN      = 255
```

---

## 12. Understanding the Output Windows

### Window 1: `camera | depth`

A side-by-side view:
- **Left** — raw camera frame with YOLO bounding boxes (green rectangles) and FPS / scale factor overlay
- **Right** — depth map using the **Turbo colourmap** from the `v. fast depth` pipeline:
  - 🔴 **Red/orange** = close (high disparity)
  - 🟡 **Yellow/green** = mid-range
  - 🔵 **Blue/purple** = far (low disparity)

### Window 2: `costmap`

Top-down bird's-eye view of the space ahead. The **white dot** at the bottom centre is the robot position.

```
     ← left       right →
     
 8m  ─────────────────────
     │                   │
 6m  ─────────────────────
     │                   │
 4m  ─────────────────────
     │                   │
 2m  ─────────────────────
     │       ●           │   ← robot here
      ───────────────────
```

- Grid rows → distance ahead (bottom = near, top = far)
- Grid columns → lateral position (left = left, right = right)
- Horizontal lines every 2 m with distance labels

**Why does it look mostly pink indoors?**
Indoor rooms are almost entirely walls, furniture, and people — all LETHAL
by both semantic classification and height geometry. Point the camera outside
at a path and you will see a clear green traversable corridor form in front
of the robot with pink obstacles at the edges.

---

## 13. Semantic Cost Table

Cost is assigned by keyword matching against ADE20K label names:

| Terrain Type | Keywords | Cost | Notes |
|---|---|---|---|
| Road / path | road, path, sidewalk, dirt track, runway | **0** | Fully free |
| Floor / flat ground | floor, land, field, carpet, rug, mat | **20** | Indoor or flat outdoor |
| Grass | grass | **40** | Traversable but bumpy |
| Soft earth | earth, sand, hill | **80** | High effort, low speed |
| Water / hazards | water, river, sea, lake, swimming | **254** | LETHAL |
| Structure / people | tree, plant, rock, building, wall, fence, person, car, truck | **254** | LETHAL |
| Sky / ceiling | sky, ceiling | **−1** | Ignored (no ground projection) |
| Unknown class | *(anything else)* | **100** | Uncertain — not free, not lethal |

---

## 14. Depth Pipeline — v. fast depth

The depth head is a full port of the `v. fast depth` WebGL pipeline into Python/PyTorch.

### Model

`depth-anything/Depth-Anything-V2-Small-hf` — the DINOv2 ViT-S backbone variant.
Outputs **relative disparity** (inverse depth), not metric depth directly.

### Resolution

**336×336** — exactly 24×24 patches of size 14 px (576 patches total vs 1,369 at the default 518×518).
This is the single biggest speed-up: 58% fewer tokens → 2.5× faster inference.

### Temporal stabilisation

Each frame's disparity map is blended with the previous frame using a
**motion-adaptive alpha**:

```
diff       = |raw_disp - prev_disp|
rel_diff   = clamp(diff / dynamic_range, 0, 1)
motion     = smoothstep(0.04, 0.25, rel_diff)   # 0 = static, 1 = fast motion
alpha      = mix(0.22, 1.0, motion)
smoothed   = prev * (1 - alpha) + raw * alpha
```

- **Static regions**: α = 0.22 → heavy temporal smoothing, very stable
- **Moving regions**: α = 1.0 → instant response, no trailing artefacts

### Dynamic range tracking

The display min/max is tracked with a slow EMA (α = 0.15) over each frame's
actual disparity range, so the colourmap auto-adjusts to scene content without
flickering.

### Metric conversion

```
metric_depth = 3.0 / max(disparity, 0.05)
```

This maps disparity to metres in the approximate range 0.3–60 m.
`recover_scale()` then applies a per-frame scalar to pin the ground to Z = 0 m
using the known camera height.

---

## 15. Performance Benchmarks

Measured on **Apple M4** (MacBook Air), `--source test_clip.mp4`, `--depth-res 336`:

| Stage | Model | Latency | Notes |
|---|---|---|---|
| Depth estimation | Depth Anything V2 Small | ~65–85 ms | 336×336 ViT patch-aligned input |
| Semantic scene parsing | YOLO26n-sem-ade20k | ~35–42 ms | 150 ADE20K scene & object classes |
| Costmap synthesis | Backprojection + fusion + inflation | ~35–50 ms | Metric ground projection & obstacle inflation |
| Rendering | Display composition + overlay | ~30–45 ms | Translucent semantics overlay + Turbo depth |
| **Total** | **Full pipeline** | **~185–205 ms** | **~5.0–5.4 FPS** |

> **Thermal note:** On sustained load, Apple Silicon thermally throttles to ~2 FPS.
> Wait ~60 s between long sessions, or use `--every 2` to halve the thermal pressure.

### Costmap stage alone (CPU, no models)

```
python test_geometry.py
# → costmap stage: 31.7 ms/frame at 1280×720 (CPU)  ≈ 32 Hz headroom
```

Reducing `proc_w/proc_h` to 640×360 drops this to ~8 ms/frame.

---

## 16. Sanity Checks

Before trusting the output on a real robot, run through these checks:

1. **Distance accuracy** — stand a 1 m box at a measured 3 m.
   The scale readout on the camera window should show `scale x~1.0` and
   the depth at the box should be ~3 m.

2. **Lateral position** — walk a person to the LEFT of the camera.
   A lethal blob should appear on the LEFT side of the costmap grid
   (positive Y, displayed on the left of the window).

3. **Negative obstacle** — point at a step-down (kerb drop, floor edge).
   The ditch should appear lethal from the height geometry rule,
   not just from semantics.

4. **No-measurement safety** — cover the lens.
   The entire costmap should go UNKNOWN (grey). No cell should ever
   read FREE when there is no depth data.

5. **Scale bounds** — on flat ground the scale overlay should read
   between `x0.4` and `x2.5`. Outside that range the rig geometry is wrong.

---

## 17. Test Suite

```bash
source .venv/bin/activate
python test_geometry.py
```

The test suite covers all 8 geometry and config invariants with no neural models loaded (pure CPU, runs in ~2 s):

| # | Test | Expected |
|---|---|---|
| 1 | Flat ground back-projects to Z = 0 | 0.00000 m RMS |
| 2 | Flat ground produces zero cost | 0 lethal cells |
| 3 | Grid orientation (X forward, Y left) | axis verified |
| 4 | Scale recovery at 1.4× error | recovers 0.714 exactly |
| 5 | Scale recovery at 0.7× error | recovers 1.429 exactly |
| 6 | Ditch triggers negative-obstacle rule | 392 lethal cells |
| 7 | Left obstacle → positive Y in grid | centroid Y = +0.79 m |
| 8 | Right obstacle → negative Y in grid | centroid Y = −0.88 m |
| 9 | Obstacle ahead → centred | centroid Y = −0.05 m |
| 10 | Render: left obstacle draws left on screen | verified |
| 11 | Blank depth → ALL UNKNOWN, never free | verified |
| 12 | All-water semantics → all lethal | verified |
| 13 | Detection stamps lethal footprint | 15 cells |
| 14 | Costmap stage under 150 ms at 1280×720 on CPU | 31.7 ms |
| 15 | Default `fx`/`fy` = 940.0 | verified |
| 16 | Default `cx`/`cy` = 640/360 | verified |

---

## 18. Known Limitations

1. **Monocular scale degrades on slopes.** `recover_scale()` assumes the
   ground is flat and visible. On a slope or when no ground pixels are in view
   (e.g. looking up a hill), the scale estimate is wrong. Clamped to `[0.4, 2.5]×`.

2. **No temporal grid fusion.** Each frame builds a fresh grid from scratch.
   Fast camera panning causes the costmap to flicker. The fix is an EMA per
   cell in the map frame, which requires a pose/odometry source.

3. **No global map.** The grid is robot-centric and resets every frame.
   A global occupancy map requires ORB-SLAM3 or similar for pose.

4. **Thermal throttling on Apple Silicon.** Sustained model inference causes
   the M-chip to downclock (~2 FPS). Use `--every 2` or add cooling pauses
   between sessions.

5. **Indoor scenes look mostly lethal.** Walls, ceilings, furniture, and people
   are all correctly classified as lethal. The pipeline is designed for
   outdoor ground-robot navigation — indoor it just shows you everything is blocked.

6. **Object inflation uses a Python loop.** Fine for a handful of detected
   people; will degrade if 20+ large vehicles fill the frame. Vectorisable
   with `np.meshgrid` if needed.

7. **`PYTORCH_ENABLE_MPS_FALLBACK=1` is required.** Some ops (e.g. `bicubic`
   interpolation in some PyTorch versions) are not implemented in MPS and fall
   back to CPU. Without the env variable, these raise an error.

---

## 19. File Overview

```
sih_prototype_costmap/
├── costmap_prototype.py   Main pipeline: Perception, Costmap, render, main loop
├── test_geometry.py       Offline geometry/config test suite (no GPU needed)
├── calibrate.py           Camera intrinsics calibration using a chessboard
├── requirements.txt       Python dependencies with version pins
├── setup_mac.sh           One-shot venv + pip install script for macOS
├── ANTIGRAVITY_PROMPT.md  Original design spec and hard constraints
├── yolo11n.pt             YOLO11 nano weights (pre-downloaded)
├── test_clip.mp4          Recorded test clip for repeatable benchmarking
└── recorded_test.mp4      Another recorded session for development
```

### Models downloaded automatically on first run

| Model | Source | Size | Purpose |
|---|---|---|---|
| `Depth-Anything-V2-Small-hf` | HuggingFace Hub | ~97 MB | Relative depth / disparity |
| `segformer-b0-finetuned-ade-512-512` | HuggingFace Hub | ~15 MB | 150-class terrain semantics |
| `yolo11n.pt` | bundled | ~5 MB | Object detection |

> **Note:** `*.pt` weights and `*.mp4` test clips are excluded from version
> control via `.gitignore` (large binaries). `yolo11n.pt` is fetched
> automatically by `ultralytics` on first run if not already present; the
> recorded clips can be regenerated with `--record` (see [§9](#9-running-the-pipeline)).
