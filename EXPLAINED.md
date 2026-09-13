# The Complete Solution, Explained

**SIH PS 26126 — Vision-Based Autonomous Navigation for an Outdoor UGV**

> One camera looks at the ground. The software turns what it sees into a
> top-down "danger map". A planner reads that map and outputs two numbers —
> *how fast to drive* and *how hard to turn*. That is the whole system.

This document explains every stage, with the exact equations and the exact
constants used in the code. Read top-to-bottom for the full story, or jump to
the section you need.

---

## Table of contents

1. [The one-paragraph version (for a layman)](#1-the-one-paragraph-version-for-a-layman)
2. [The three problems, and where each is solved](#2-the-three-problems-and-where-each-is-solved)
3. [The data pipeline, end to end](#3-the-data-pipeline-end-to-end)
4. [Coordinate frames — the thing everyone gets wrong](#4-coordinate-frames--the-thing-everyone-gets-wrong)
5. [Stage 1 — Depth: pixels to distances](#5-stage-1--depth-pixels-to-distances)
6. [Lens distortion — rectify before any of the geometry below is true](#6-lens-distortion--rectify-before-any-of-the-geometry-below-is-true)
7. [Stage 2 — Back-projection: distances to 3D points](#7-stage-2--back-projection-distances-to-3d-points)
8. [Stage 3 — The ground plane (self-calibration)](#8-stage-3--the-ground-plane-self-calibration)
9. [Stage 4 — Rotating into the ground frame](#9-stage-4--rotating-into-the-ground-frame)
10. [Stage 5 — THE COSTMAP (the heart of the project)](#10-stage-5--the-costmap-the-heart-of-the-project)
11. [Stage 6 — The hole rule (invisible trenches)](#11-stage-6--the-hole-rule-invisible-trenches)
12. [Stage 7 — Inflation (giving the robot a body)](#12-stage-7--inflation-giving-the-robot-a-body)
13. [Stage 8 — The global costmap (memory)](#13-stage-8--the-global-costmap-memory)
14. [Stage 9 — Global planning (A*)](#14-stage-9--global-planning-a)
15. [Stage 10 — The carrot (global → local hand-off)](#15-stage-10--the-carrot-global--local-hand-off)
16. [Stage 11 — Local planning + pure pursuit → (v, ω)](#16-stage-11--local-planning--pure-pursuit--v-ω)
17. [Stage 12 — The state machine and recoveries](#17-stage-12--the-state-machine-and-recoveries)
18. [How this maps onto Nav2, exactly](#18-how-this-maps-onto-nav2-exactly)
19. [A fully worked numeric example (one pixel → one wheel command)](#19-a-fully-worked-numeric-example-one-pixel--one-wheel-command)
20. [Every constant, in one place](#20-every-constant-in-one-place)
21. [The three safety rules, and why they exist](#21-the-three-safety-rules-and-why-they-exist)
22. [Honest limitations](#22-honest-limitations)

---

## 1. The one-paragraph version (for a layman)

- Imagine you are looking at a photo of the ground in front of a robot.
- The software first guesses **how far away every pixel is** (a depth image).
- It also guesses **what every pixel is** — road, grass, water, tree, rock (semantic segmentation).
- Knowing how far each pixel is, it converts the photo into a **cloud of 3D dots** floating in space.
- It works out **where the flat ground actually is** among those dots — this also tells it how high the camera is and how much it is tilted. Nothing is hand-measured.
- It draws a **grid on the ground**, like graph paper, 10 cm per square, stretching ~12 m ahead and ~5 m either side.
- Each square gets a **score from 0 to 255**: 0 = perfectly safe tarmac, 254 = will kill the robot, 255 = "I have no idea, never saw it".
- That scored grid is the **costmap**.
- A pathfinding algorithm (A*) walks across the grid looking for the **cheapest chain of squares** from the robot to the destination.
- A steering rule (pure pursuit) looks at that chain, picks a point ~1.5–2.5 m along it, and computes the **arc** that would take the robot there.
- The arc becomes two numbers: **forward speed `v`** and **turn rate `ω`**. Those go to the wheels.
- Repeat 4–6 times per second, forever.

---

## 2. The three problems, and where each is solved

The problem statement names three challenges. Here is the map from problem → file:

| # | Problem statement asks for | Solved by | Status |
|---|---|---|---|
| 1 | **Path detection** — safe path vs. hazards (rocks, ditches, trees) | `perception_core.py` → depth + semantics + ground plane → costmap | Complete |
| 2 | **Visual localization** — pose without GPS | `navstack.PoseSource` (the seam). Ground truth from the sim today; the team's VSLAM plugs in here | **Interface done, VSLAM not implemented** |
| 3 | **Collision avoidance** — dynamic routing to a destination | `navstack.py` → global A* + local A* + pure pursuit + recoveries | Complete |
| — | Path planner → wheel/motor commands | `drive_command()` → `(v, ω)` → `geometry_msgs/Twist` | Complete |

- Being blunt about #2 matters: the **shape** of visual localization is in place (`PoseSource.get() -> Pose`), and everything downstream already consumes it, but the actual VSLAM algorithm is not written. In the simulator the renderer hands us the true pose through that same interface, which is exactly what a working VSLAM would do.

---

## 3. The data pipeline, end to end

```
 ┌─ camera (webcam, or the Three.js rover's virtual camera) ─────────────────┐
 │  RGB image  640×360  [+ true depth 320×180 u16 mm, sim only]              │
 └───────────────────────────────┬──────────────────────────────────────────┘
                                 │
   ┌─────────────────────────────┴──────────────────────────────┐
   │                                                             │
   ▼ DEPTH                                                       ▼ SEMANTICS
 Depth Anything V2 Metric-Outdoor          YOLO26 semantic seg (ADE20K, 150 classes)
 → metres per pixel                        → class id per pixel
                                           → keyword table → cost 0..254 per pixel
   │                                                             │
   └─────────────────────────────┬──────────────────────────────┘
                                 ▼
                    BACK-PROJECTION (pinhole)
                    every pixel → a 3D point (Xc, Yc, Zc) in the OPTICAL frame
                                 ▼
                    GROUND PLANE (RANSAC, every frame)
                    → camera height, pitch, roll  (MEASURED, not configured)
                                 ▼
                    ROTATE into the GROUND-ALIGNED ROBOT frame
                    → X forward, Y left, Z = height above the ground
                                 ▼
                    BIN into a 10 cm grid, score each cell
                    cost = max(semantic vote, positive obstacle,
                               negative obstacle, hole rule)
                                 ▼
                    INFLATE by the robot radius
                                 ▼
                    ╔══════════════ LOCAL COSTMAP ══════════════╗
                    ║  115 × 100 uint8, robot-centric, this frame ║
                    ╚═════════════════════════╤═════════════════╝
                                              │ + pose (x, y, θ)
                                              ▼
                    GLOBAL COSTMAP  (world frame, 640×640 @ 0.25 m, max-fusion, permanent memory)
                                              ▼
                    GLOBAL A*  (on a 2× pooled, boxed copy, ≤ 1 Hz)  → world path
                                              ▼
                    CARROT  (first path point ≥ 10.5 m ahead)  → local goal
                                              ▼
                    LOCAL A*  (on this frame's grid, UNKNOWN backfilled from global memory)
                                              ▼
                    PURE PURSUIT  → (v, ω)
                                              ▼
                    ┌──────────────┬─────────────────┬──────────────────┐
                    │ WebSocket    │ GET /ros/cmd_vel│ dashboard render │
                    │ to the rover │ geometry_msgs/  │ PNG costmaps     │
                    │              │ Twist           │                  │
                    └──────────────┴─────────────────┴──────────────────┘
```

**Files:**

| File | Responsibility |
|---|---|
| `perception_core.py` | Everything from depth+semantics to the finished local costmap |
| `navstack.py` | Global costmap, global A*, carrot, Navigator state machine |
| `costmap_prototype.py` | Provides the planner primitives `astar()`, `drive_command()`, `path_metres()` |
| `perception_server.py` | Wires it together, runs the models, serves WebSocket + HTTP |
| `ros_msgs.py` | Converts our data structures into Nav2 / ROS 2 message dicts |
| `sim3d/` | The Three.js rover that produces frames and consumes `(v, ω)` |

---

## 4. Coordinate frames — the thing everyone gets wrong

- Four frames exist. Confusing any two of them produces a costmap that looks plausible and is wrong.

| Frame | Axes | Origin | Used by |
|---|---|---|---|
| **Image** | `u` right (px), `v` down (px) | top-left pixel | the camera, the models |
| **Optical** | `X` right, `Y` **down**, `Z` forward (m) | the lens | back-projection, plane fitting |
| **Robot (ground-aligned)** | `X` forward, `Y` **left**, `Z` **up** (m) | on the ground directly under the lens | the costmap |
| **World (nav)** | `X` east, `Y` north, `θ` CCW from +X | wherever the run started | global map, global path, goals |

- The **optical** frame is the OpenCV convention: Y points *down*, which is why "up" appears as a negative Y in the plane normal.
- The **robot** frame is the ROS convention (REP-103): X forward, Y left, Z up, `θ` counter-clockwise positive. `ω > 0` means **turn left**.
- The Three.js simulator uses yet another convention (Y up, drives toward −Z). It is converted in exactly **one** file, `sim3d/src/nav/frames.js`:

```
nav.x = −three.z        nav.y = −three.x        nav.θ = heading
```

- **Grid indexing** inside the costmap array:
  - `axis 0 = X forward` — row 0 is the nearest row to the robot
  - `axis 1 = Y left` — column 0 is the **rightmost** cell
  - So `grid[0, 0]` is "near-right" and `grid[nx-1, ny-1]` is "far-left".

---

## 5. Stage 1 — Depth: pixels to distances

There are several `--depth` choices. `metric` and `metric-indoor` produce metres directly from the network. `relative` and `affine` share a network that outputs **disparity**, not distance, and differ only in how that disparity is turned into metres — this difference matters more than it looks, and 5c below tells that story in full because it is the most instructive mistake in the project. `sim` bypasses the network entirely and reads ground-truth depth from the renderer.

### 5a. `--depth metric` / `--depth metric-indoor` (the real answer, for the right scene)

- Model: **Depth Anything V2 Metric-Outdoor-Small** (`--depth metric`), or **Metric-Indoor-Small** (`--depth metric-indoor`) — both HuggingFace transformers.
- Input: RGB image resized to 336×336 (or `--depth-res`, any multiple of 14 — 252/280 trade accuracy for speed). Output: metres, directly.
- Runs on Apple MPS / CUDA / CPU (auto-detected by `pick_device()`).
- ~85–110 ms per 1280×720 frame on an M4.
- **Neither model transfers across camera height or scene type.** Each is trained on a particular mounting height and driving-scene statistics, and the learned metric scale is part of that training, not something the network re-derives from geometry. Measured: pointed indoors at a wall known to be ~2 m away, the **outdoor** model read it as 5–9 m, and in doing so threw away most of the frame — only **18%** of pixels landed inside `max_depth` at all. That is the entire reason `metric-indoor` exists as its own `--depth` choice rather than a correction factor bolted onto the outdoor model: use `metric` outdoors, `metric-indoor` on the bench, and expect neither to generalise to a height or scene it never saw in training.

### 5b. `--depth relative` (fallback, assumed scale — the old approach)

- Model: Depth Anything V2 Small — outputs **disparity**, not distance.
- Converted by `depth = 1 / max(disparity, 1e-3)` (`depth_kind="scaled"` in `PerceptionCore.process`). This is proportional to true depth but the constant is unknown.
- The scale is recovered *after* fitting the ground plane:

$$ s = \frac{h_{\text{nominal}}}{d_{\text{fit}}} \qquad (X, Y, Z) \leftarrow s \cdot (X, Y, Z) $$

  where `h_nominal` is a single assumed camera height (default 0.60 m) and `d_fit` is the fitted plane offset in unscaled units.
- **Why this is better than the naive version**: only the *scale* is assumed. The *tilt* is still measured. The old prototype assumed both.
- **Why this is not the same fix as `affine` (5c, below)**: `1/disparity` additionally assumes there is no *shift* in the disparity-to-inverse-depth relationship. When that assumption is false — which, on real data, it is — this path does not simply end up the wrong size. It warps. See 5c for the measurement that found this and the two fixes that turned out not to work.

### 5c. `--depth affine` — the affine story, told honestly

- Relative Depth Anything is not "depth up to an unknown scale". It is **affine-invariant in disparity**: the network's raw output `disp` relates to true inverse depth as

$$ \frac{1}{Z} = a \cdot \text{disp} + b $$

  for a scale `a` **and a shift `b`**, and the network reports neither.

- 5b's `1/disp` silently assumes `b = 0`. When it is not, the result is not merely the wrong size — it is **warped**: near and far points are bent by different, wrong amounts, so no rescaling applied afterwards can undo it. Measured on a synthetic scene built with a true 10° pitch and 0.17 m camera height: `1/disp` lost the ground plane **entirely** — recovered pitch 0.00°, recovered height 0.000 m. Not "off by some margin". Gone.

- **The obvious fix does not work, and it is worth seeing why in full, because the failure is structural, not a shortage of data or a weak solver.** The natural idea is to solve for both `a` and `b` using the one thing already being measured every frame anyway: the ground is flat. For a point on the fitted plane `n·P + d = 0`, with unit ray `r = (x_n, y_n, 1)` and `m = -n/d`:

$$ a \cdot \text{disp} + b = m_x x_n + m_y y_n + m_z $$

  which looks like an ordinary linear least-squares problem in five unknowns `(a, b, m_x, m_y, m_z)`. It is not — the system is **rank-deficient by construction**. The column multiplying `b` is all `+1`; the column multiplying `m_z` is all `-1`. So `(0, 1, 0, 0, 1)` is an *exact* null direction of the design matrix: add any amount to `b` and the same amount to `m_z`, and every equation is still satisfied. Measured singular values of the real design matrix: `[341, 52.6, 35.8, 1.8e-06, 1.3e-13]` — **two** of five vanish, not one, exactly matching that null space.

  Physically this says something simple: on a set of coplanar points, a constant added to inverse depth is **indistinguishable** from the plane sitting at a different distance. Only the combination `c = m_z - b` is observable from flat ground; `b` and the plane's own `m_z` cannot be told apart, ever, from ground points alone.

  A known camera height does not rescue it either. Fixing `|m| = 1/h` (the tape-measure fix) supplies exactly one more equation for the two unknowns that remain — still a one-parameter family, not an answer.

- **The next-most-obvious fix also fails, and more dangerously, because it looks like it worked.** Seed the solve's orientation with the *previous frame's* fitted plane. The loop is then self-confirming rather than self-correcting: seeded at a 12° pitch against a true 10°, the estimate sat at **12.79°** for as many iterations as the test ran (8), and never moved toward the truth. The solve forces the ground onto whatever orientation it is handed; RANSAC then re-discovers that same orientation from the now-consistent cloud; the two agree with each other forever. (`a` itself converges fine — 0.693 against a true 0.700. It is the orientation, and therefore `b`, that gets stuck.)

  | path | recovered pitch | recovered height | notes |
  |---|---|---|---|
  | `metric` | 10.00° | 0.1700 m | correct; a box at 1.5 m produced a LETHAL centroid at exactly 1.50 m |
  | `affine` + previous-frame seed | 12.79° | 0.1700 m | equals the seed — self-confirming, not self-correcting |
  | `1/disp` (`relative`) | −0.00° | 0.0000 m | plane lost entirely |

- **Conclusion.** Half of the original claim stands: `1/disp` really is badly wrong once a shift exists, not just imprecise — that part was right. But the fix cannot be manufactured from inside the depth estimate, however the solve is arranged. It needs an orientation source the depth cannot bias in the first place — an IMU gravity vector, or non-coplanar SLAM map points (ordinary ground pixels are, by definition, coplanar, so they can never supply it). Until one of those exists, `rover_cfg()` ships `affine_depth=False`, and the rig runs on `metric` (`metric-indoor` indoors). The `affine` machinery — `solve_affine_depth()`, `depth_from_affine()`, `CoreCfg.bootstrap_pitch` seeding frame 1 only — is implemented and documented, not deleted: it returns `(None, None, info)` rather than a silently-wrong answer whenever no real orientation constraint is available, and `PerceptionCore.process` falls back to the `relative` path (with a logged warning) when it does.

### 5d. `--depth sim` (ground-truth depth from the renderer)

- The Three.js rover renders a second pass with `MeshDepthMaterial` + `RGBADepthPacking`.
- That gives **non-linear** NDC depth in `[0,1]`, which is un-linearised on the CPU:

$$ z_{\text{view}} = \frac{n \cdot f}{(f - n)\,v_{\text{ndc}} - f}, \qquad \text{depth}_{\text{mm}} = \text{round}(-z_{\text{view}} \times 1000) $$

  with `n = 0.1 m` (near plane), `f = 100 m` (far plane).
- Packed as `u16` millimetres, `0` = "no measurement" (sky, or beyond range).
- This is the **stand-in for a stereo camera**. It enters the pipeline at exactly the same interface as the neural depth, so swapping in a real ZED/RealSense later changes nothing downstream.

### Depth validity

- A pixel is used only if:

$$ \texttt{min\_depth} < z < \texttt{max\_depth} \quad\text{and}\quad z \text{ is finite} $$

- Sim: `0.2 m < z < 20 m`. Webcam: `0.2 m < z < 12 m`.
- Every **2nd** pixel in each direction is used (`stride = 2`), i.e. 25% of pixels. Grid statistics do not need more, and this is a 4× speedup.

---

## 6. Lens distortion — rectify before any of the geometry below is true

- Everything from here on — `backproject_optical`, the plane fit, the ground frame — is the **pinhole camera model**, and the pinhole model has one unstated assumption baked into it: **straight lines in the world stay straight in the image.** A real lens does not promise that. It bends.
- On an uncorrected lens with ordinary barrel distortion, the ground plane — which is flat and should image as a flat wedge — **bows upward toward the edges of the frame**. Pixels that are honestly on the ground get read, by a perfectly correct pinhole equation applied to a not-quite-pinhole image, as points that rise above it. The costmap then invents **LETHAL cells running down both sides of the path**, on ground the robot could have driven straight across.
- This was not a new problem. `calibrate.py` has *always* computed and printed the lens's distortion coefficients (`k1, k2, p1, p2, k3` from `cv2.calibrateCamera`) — the number was sitting in a terminal scroll-back every time calibration ran. Nothing downstream ever read it. It printed, and was ignored.
- `CoreCfg.dist` and `undistort()` close that gap:
  - `CoreCfg.dist: tuple = ()` carries the coefficients through the config, the same way `fx, fy, cx, cy` already do.
  - `undistort(img, cfg)` rectifies the raw RGB frame with a cached `cv2.initUndistortRectifyMap` / `cv2.remap` (the `Undistorter` class), built once per `(size, K, dist)` and reused every frame after that — it is a straight-line correction applied to the *image*, once, before anything else looks at it, not a correction applied piecemeal to depth or points downstream.
  - **`undistort()` is a no-op when `dist` is empty** (`if not len(cfg.dist) or not np.any(...): return img`). That is deliberate, not an oversight: the simulator's virtual camera has no lens to distort in the first place, so it is already "rectified" by construction, and a webcam or rover feed with no measured coefficients is assumed rectified too rather than guessed at. **This is why the simulator is unaffected by any of this** — it never had the problem `undistort()` exists to fix.
  - On the rover this happens once, up front, on the decoded BGR frame, before depth or semantics ever see it — so every equation from here on operates on an image where straight lines really are straight.

---

## 7. Stage 2 — Back-projection: distances to 3D points

- This is the **pinhole camera model**, run backwards.

### The intrinsics

- Four numbers describe the camera: `fx, fy` (focal length in pixels) and `cx, cy` (the optical centre).
- From a **horizontal** field of view (webcam):

$$ f_x = f_y = \frac{W/2}{\tan(\text{hfov}/2)}, \quad c_x = W/2, \quad c_y = H/2 $$

- From a **vertical** field of view (Three.js `PerspectiveCamera.fov` is vertical):

$$ f_y = f_x = \frac{H/2}{\tan(\text{vfov}/2)} $$

- Simulator: `H = 360`, `vfov = 60°` → `fy = 180 / tan(30°) = 311.77 px`.
- MacBook webcam: `W = 640`, `hfov = 78°` → `fx = 320 / tan(39°) = 395.2 px`.
- The sim sends its **exact** intrinsics in every frame header, so there is no calibration error there at all.

### The equations

For a pixel `(u, v)` with depth `z`:

$$
x_n = \frac{u - c_x}{f_x}, \qquad
y_n = \frac{v - c_y}{f_y}
$$

$$
\boxed{\;X_c = x_n \, z, \qquad Y_c = y_n \, z, \qquad Z_c = z\;}
$$

- `(Xc, Yc, Zc)` is a point in the **optical** frame: right, down, forward.
- `xn, yn` depend only on the pixel lattice and the intrinsics, so they are computed once and **cached** (`_RAY_CACHE`) — recomputing them per frame was measurable overhead.
- Note `z` here is the distance **along the optical axis** (a "z-depth"), not the ray length. Both Depth Anything metric and the renderer's `viewZ` are z-depths, so this is consistent.

**Result:** ~57,600 3D points per frame (640×360 at stride 2).

---

## 8. Stage 3 — The ground plane (self-calibration)

This is the single most important idea in the project.

### Why it exists

- The first prototype (`costmap_prototype.py`) **trusted** a hand-measured camera height and pitch, and declared the ground to be `Z = 0` from those constants.
- On a laptop on the floor, a hand-held phone, or a rover with suspension:
  - the height is not known,
  - the pitch changes every time anything moves,
  - **roll is never exactly zero**.
- If the datum is wrong, *every height threshold is measured against the wrong reference*. A 6° roll alone produced **72 false lethal cells** — the robot refuses to drive across a perfectly flat floor because one side of the map "rises".
- So: **the camera pose relative to the ground is a measurement taken every single frame.** `--height` / `--pitch` / `--roll` survive only as optional *locks* (for a rigid rig, or as a cross-check).

### The plane, mathematically

- A plane in the optical frame:

$$ \mathbf{n} \cdot \mathbf{p} + d = 0, \qquad \|\mathbf{n}\| = 1 $$

- `n` is the unit normal, oriented so it points **up** (away from the ground).
- Because the camera is at the origin, substituting `p = 0` gives:

$$ \boxed{\;d = \text{perpendicular distance from lens to ground} = \textbf{camera height}\;} $$

- The normal encodes the tilt. For a camera pitched down by `p` and rolled by `r` (positive = right side lower):

$$ \mathbf{n} = \big(\sin r \cos p,\; -\cos r \cos p,\; -\sin p\big) $$

- And the inverse, which is what we actually read out:

$$ \text{pitch} = \operatorname{atan2}\!\big(-n_z,\; \sqrt{n_x^2 + n_y^2}\big), \qquad
   \text{roll} = \operatorname{atan2}\!\big(n_x,\; -n_y\big) $$

- Sanity check: a perfectly level camera has `n = (0, −1, 0)` — "up" is negative-Y because the optical frame's Y points down. Pitch = 0, roll = 0. ✓

### Step 1: choose candidate points

- Not every point can vote. Candidates must satisfy **all** of:
  1. **Valid depth** (finite, in range).
  2. **In the lower 65% of the image** (`plane_lower_frac`) — the sky and the tops of trees are not ground.
  3. **Semantically ground-like**: the pixel's semantic cost is in `[0, 80]` (`GROUND_COST_MAX`), i.e. road / floor / grass / earth / sand / hill.
  4. **In the near field first**: `Zc < plane_near_range` (5 m webcam, 6 m sim).
- If fewer than `plane_min_pts = 300` candidates survive, the range widens to `plane_max_range = 10 m`. If still too few, the semantic filter is dropped and only the lower 40% of the image is used.

**Why the near field first (this is subtle and important):**

- The plane is the *datum* every height threshold is measured against. It must describe **the ground the robot is about to drive on**, not whichever surface happens to fill the most pixels.
- Approaching a kerb drop, the lower surface *beyond* the lip grows in the image until it outvotes the road under the robot. RANSAC would then lock onto the lower surface, and the kerb would measure as a *rise*, not a drop.
- Restricting to the near field makes that impossible, and costs nothing on flat ground.

### Step 2: RANSAC

- **150 hypotheses** per frame (`plane_iters`), all evaluated **vectorised** in numpy (no Python loop).
- Each hypothesis: pick 3 random candidate points `A, B, C`:

$$ \mathbf{n} = \frac{(B-A)\times(C-A)}{\|(B-A)\times(C-A)\|}, \qquad d = -\mathbf{n}\cdot A $$

- Orient it: if `d < 0`, flip both `n` and `d`, so `d > 0` always means "camera above the surface".
- **Plus one extra hypothesis**: the previous frame's plane, seeded in. Temporal continuity for free.

### Step 3: the inlier gate (the key design decision)

- A point is an inlier if its perpendicular distance to the hypothesis is within a **fixed** band:

$$ \big|\mathbf{n}\cdot\mathbf{p} + d\big| \;\le\; \tau(Z), \qquad
   \tau(Z) = \min\big(\underbrace{0.05}_{\text{base}} + \underbrace{0.01\,Z}_{\text{sensor noise}},\; \underbrace{0.10}_{\text{cap}}\big)\ \text{metres} $$

- **Why fixed, and never a data-driven band (MAD/std):**
  - At a step-down, the ground is **bimodal** — two surfaces at different heights.
  - A data-driven band *widens to swallow both*, tilting the fitted plane straight through the step, so the drop measures **zero height** and the robot drives into it.
  - A fixed band capped at `0.10 m` — comfortably below `|ditch_h| = 0.20 m` — cannot do that. **Anything deep enough to be a ditch is by definition too deep to be ground.**

### Step 4: scoring and plausibility

- Hypotheses are rejected outright unless:
  - `−n_y ≥ cos(60°)` — the normal points roughly toward image-up. This **rejects walls**, which are also perfect planes.
  - `|atan2(n_x, −n_y)| ≤ 45°` — no realistic rig rolls further than that.
  - `0.03 m ≤ d ≤ 5 m` — a sane camera height.
- Surviving hypotheses are scored with **distance-weighted inliers**:

$$ \text{score} = \sum_{i \in \text{inliers}} \frac{1}{\max(Z_i,\; 0.5)} $$

- Near points count more, because they carry more information about the ground directly under the robot.

### Step 5: refinement

- Take the winning hypothesis' inliers and do **two least-squares passes**, re-selecting inliers each time.
- Least squares here = the smallest singular vector of the 3×3 scatter matrix:

$$ Q = P - \bar{P}, \qquad \mathbf{n} = \text{smallest singular vector of } Q^\top Q, \qquad d = -\mathbf{n}\cdot\bar{P} $$

- **Confidence** is reported as the final inlier ratio:

$$ \text{confidence} = \frac{|\text{inliers}|}{|\text{candidates}|} \in [0, 1] $$

- Below `0.30` the estimate is flagged as low-confidence in the UI.

### Step 6: temporal stability (jump gate + EMA)

- Compare with last frame's plane:

$$ \Delta\theta = \arccos\!\big(\mathbf{n}_{\text{prev}}\cdot\mathbf{n}_{\text{new}}\big), \qquad \Delta d = |d_{\text{new}} - d_{\text{prev}}| $$

- It counts as a **jump** if `Δθ > 8°` or `Δd > 0.15 m`.
- A jump is **rejected** (the old plane is held) unless it has strong support: `confidence ≥ 0.60`.
- After 5 consecutive rejections (`plane_hold_frames`) the estimator relocks anyway — the rig really did move.
- Non-jump updates are blended with an **exponential moving average**, `α = 0.5`:

$$ \mathbf{n} \leftarrow \frac{\alpha\,\mathbf{n}_{\text{new}} + (1-\alpha)\,\mathbf{n}_{\text{prev}}}{\|\cdot\|}, \qquad
   d \leftarrow \alpha\,d_{\text{new}} + (1-\alpha)\,d_{\text{prev}} $$

- If the plane cannot be found at all, the previous one is **held** with decaying confidence (×0.7 per frame) for up to 10 frames; after that the map is declared `UNKNOWN` everywhere and the robot stops.

### How well does it work

- On synthetic scenes it recovers an unknown rig to **within 1 cm and 0.5°**.
- The simulator's mounted camera is at a known 1.0 m / 15°, so the HUD shows a **live accuracy check** (`plane.mount_err`) against ground truth on every frame.
- The 6° roll that used to produce 72 false lethal cells now produces **zero**.

---

## 9. Stage 4 — Rotating into the ground frame

- We now have a plane. We build a rotation that makes "height above ground" literally the Z coordinate.
- Build an orthonormal basis from the normal:

$$
\mathbf{f} = \frac{\hat{z} - (\hat{z}\cdot\mathbf{n})\,\mathbf{n}}{\|\hat{z} - (\hat{z}\cdot\mathbf{n})\,\mathbf{n}\|}
\quad\text{(the optical axis, flattened onto the ground = FORWARD)}
$$

$$
\boldsymbol{\ell} = \mathbf{n}\times\mathbf{f} \quad\text{(LEFT)}, \qquad
R = \begin{bmatrix} \mathbf{f}^\top \\ \boldsymbol{\ell}^\top \\ \mathbf{n}^\top \end{bmatrix}
$$

- Degenerate case (camera pointing straight down, `f` undefined): fall back to image-up as the forward reference.
- Then every point transforms as:

$$
\boxed{\;
\begin{aligned}
X &= \mathbf{f}\cdot\mathbf{p} && \text{(metres forward)}\\
Y &= \boldsymbol{\ell}\cdot\mathbf{p} && \text{(metres left)}\\
Z &= \mathbf{n}\cdot\mathbf{p} + d && \text{(\textbf{metres above the ground})}
\end{aligned}\;}
$$

- The `+ d` is the whole trick: because `n·p + d = 0` *defines* the ground, adding `d` makes ground points land at exactly `Z = 0`.
- Everything after this point works in plain metres above a flat floor. Roll, pitch and unknown mounting height have been **eliminated from the problem**, not assumed away.

---

## 10. Stage 5 — THE COSTMAP (the heart of the project)

### What a costmap *is*

- A costmap is a **2D array of bytes**. Nothing more.
- Each byte is one 10 cm × 10 cm square of ground, and its value answers one question:

  > *"How bad would it be to put a wheel here?"*

| Value | Meaning | Planner behaviour |
|---|---|---|
| `0` | Perfect. Tarmac, a paved road. | Free |
| `1–100` | Fine to slightly awkward. Grass, dirt, floor. | Cheap |
| `100–200` | Uncertain or unpleasant. Sand, unknown label, demoted tall label. | Expensive |
| `253` | **Inflation skirt** — a lethal cell is within one robot radius | Very expensive, still passable |
| `254` | **LETHAL** — the robot dies here | **Blocked. Never traversed.** |
| `255` | **UNKNOWN** — never measured | Expensive but passable (see §21) |

- The array is **robot-centric**: it moves with the robot, always oriented forward.

### Grid geometry

| | Simulator | Webcam |
|---|---|---|
| Forward span `x_min .. x_max` | 0.5 → 12.0 m | 0.3 → 8.0 m |
| Lateral span `y_min .. y_max` | −5.0 → +5.0 m | −4.0 → +4.0 m |
| Resolution `res` | 0.10 m | 0.10 m |
| Array shape `nx × ny` | **115 × 100** | **77 × 80** |
| Robot radius (inflation) | 1.0 m | 0.35 m |

$$ n_x = \frac{x_{\max} - x_{\min}}{\text{res}}, \qquad n_y = \frac{y_{\max} - y_{\min}}{\text{res}} $$

- `x_min > 0` because the ground immediately under the robot is not visible to a forward-facing camera. The planner therefore starts at **row 0**, not at the robot's actual centre.

### Binning: which cell does a point fall in?

$$
i_x = \left\lfloor \frac{X - x_{\min}}{\text{res}} \right\rfloor, \qquad
i_y = \left\lfloor \frac{Y - y_{\min}}{\text{res}} \right\rfloor, \qquad
\text{flat index} = i_x \cdot n_y + i_y
$$

- Points outside the grid, or above `max_obstacle_h = 2.0 m` (overhead branches, a bridge — the robot fits under them), are dropped.
- All per-cell statistics are computed with `np.bincount` on the flat index — one pass, fully vectorised, no Python loop over cells.

### The four cost channels

Every cell is scored **four independent ways**, and the final answer is the **worst** of them.

---

#### Channel A — the semantic vote

- The segmenter labels each pixel with one of 150 ADE20K classes. Each class maps to a cost via a **keyword table** matched against the model's own label strings:

| Keywords in the label | Cost |
|---|---|
| `road`, `sidewalk`, `path`, `runway` | **0** |
| `floor`, `carpet`, `rug`, `mat`, `land`, `field` | **20** |
| `grass`, `dirt track` | **40** |
| `earth`, `sand`, `hill` | **80** |
| `water`, `river`, `sea`, `lake`, `swimming`, `waterfall`, `fountain` | **254** — flat hazard, unconditional |
| `tree`, `rock`, `wall`, `fence`, `pole`, `person`, `car`, `building`, `bench`, `stairs`, … (~50 keywords) | **250** — *"tall" lethal*, conditional |
| `sky`, `ceiling` | **−1** = ignore entirely |
| anything unrecognised | **100** (default: uncertain, neither free nor lethal) |

- Matching against label *names* rather than class *ids* means the table survives a model swap.

**The vote is not an average.** A cell is lethal when a **fraction** of its points say so:

$$
\text{vote\_min} = \max\big(\underbrace{0.25}_{\texttt{sem\_lethal\_frac}} \cdot N,\; \underbrace{2}_{\texttt{geo\_min\_pts}}\big)
$$

where `N` is the number of points in the cell.

$$
\text{semantic}(c) =
\begin{cases}
254 & \text{if } N_{\text{flat-lethal}} \ge \text{vote\_min} \quad\text{(water)}\\
254 & \text{if } N_{\text{tall-lethal}} \ge \text{vote\_min} \ \wedge\ \neg\,\text{flat\_cell}\\
150 & \text{if } N_{\text{tall-lethal}} \ge \text{vote\_min} \ \wedge\ \text{flat\_cell} \quad\text{(demoted)}\\
\dfrac{1}{N_{\text{nl}}}\displaystyle\sum_{i \in \text{non-lethal}} \text{cost}_i & \text{otherwise (mean of the benign points)}
\end{cases}
$$

- **Why a vote and not an average?** Averaging in either direction was measured to be wrong:
  - Average *everything*: 30% of a cell being "tree" and 70% "grass" averages to ~103 — the tree disappears.
  - Average only if all agree: segmentation boundaries are ragged, so nothing is ever lethal.
  - A 25% vote with a floor of 2 points is the tested compromise.

---

#### Channel B — the "tall label" check (a genuinely novel safety valve)

- **The observation:** a wall, tree, car, person or rock is lethal **and has height**. A correct label is therefore *always* confirmed by the geometry channel.
- **The corollary:** such a label on a cell whose points all lie within a few centimetres of the fitted ground is a **mislabel**. A flat grey concrete floor read as "wall" is the classic failure, and it blocks perfectly drivable ground.
- So a "tall" label on a measurably-flat cell is **demoted from 254 to 150** — still expensive, so the planner avoids it when it can, but never blocking.

A cell counts as `flat_cell` when it has enough points and:

$$
\text{flat\_cell} \iff N_{\{Z \ge 0.5\,h_{\text{obs}}\}} = 0 \;\wedge\; N_{\{Z \le 0.5\,h_{\text{ditch}}\}} = 0
$$

- i.e. **no** point sits above **+12.5 cm** and **no** point sits below **−10 cm**.
- **Water is deliberately excluded** from this rule. Water has no height, so geometry can never confirm or refute it. It stays at 254 unconditionally. A pond that looks like flat ground *is* flat ground, geometrically — and it will still drown the robot.

---

#### Channel C — positive obstacles (things sticking up)

$$
\text{pos}_i \iff Z_i > h_{\text{obs}} = 0.25\ \text{m}
$$

- Catches rocks, logs, fences, bushes, kerbs, tree trunks.
- `0.25 m` is roughly "taller than the wheels can climb".

#### Channel D — negative obstacles (things dropping away)

$$
\text{neg}_i \iff Z_i < h_{\text{ditch}} = -0.20\ \text{m} \;\wedge\; X_i \le \texttt{ditch\_max\_range}
$$

- Catches kerb drops, trench walls, washouts.
- `ditch_max_range` is 9 m (sim) / 8 m (webcam) — beyond that, depth noise alone produces false drops.

**Both geometry channels require a consensus, not one point:**

$$
\text{min\_sup} = \operatorname{clip}\big(\underbrace{0.20}_{\texttt{geo\_min\_frac}} \cdot N,\; \underbrace{2}_{\min},\; \underbrace{6}_{\max}\big)
$$

$$
\text{geo\_lethal}(c) \iff \text{seen}(c) \;\wedge\; N \ge 2 \;\wedge\; \big(N_{\text{pos}} \ge \text{min\_sup} \;\vee\; N_{\text{neg}} \ge \text{min\_sup}\big)
$$

- Requiring **both** an absolute count (≥ 2) and a fraction (≥ 20%) is what makes this robust to depth speckle. A single noisy pixel cannot create an obstacle; a coherent surface patch can.
- The `clip(..., 2, 6)` cap means a densely-sampled near cell does not need 20 points to agree — 6 is enough evidence.

---

#### Evidence floor: `seen`

$$ \text{seen}(c) \iff N \ge \texttt{min\_cell\_pts} = 3 $$

- Fewer than 3 points → the cell is `UNKNOWN` (255), regardless of what those points said.

---

### Fusion

$$
\boxed{\;\text{cost}(c) = \max\big(\text{semantic}(c),\; \text{geometry}(c),\; \text{hole}(c)\big), \quad\text{and}\quad \neg\,\text{seen}(c) \Rightarrow \text{UNKNOWN}\;}
$$

- **`max`, never average.** This is the safety argument of the whole project in one operator:

  > *If **any** channel says this cell is dangerous, it is dangerous.*

- A false positive costs a detour. A false negative costs the robot.

---

## 11. Stage 6 — The hole rule (invisible trenches)

This is the most unusual rule in the codebase, and it exists because of a specific failure that was observed.

### The failure

- A 1.4 m-wide trench. Its floor is **hidden by its own lip** — the camera, looking forward and slightly down, cannot see into it at all.
- So there are **no depth points** in those cells. Not "points that say it is flat" — literally *nothing*.
- Those cells become `UNKNOWN` (255). And `UNKNOWN` is "expensive but passable".
- The planner did the arithmetic: crossing 14 unknown cells is cheaper than a 4 m detour.
- **The rover drove into the trench.**

### The insight

> A downward-looking camera sees **continuous** ground. A run of cells with *no
> measurement at all*, that has measured ground both **before** it and **beyond**
> it along the viewing direction, is not "no information" — the surface there
> dipped out of sight, **or it would have been measured**.

- That is *exactly* the signature of a trench or a kerb drop.

### The algorithm

Let `any_pt[ix, iy] = (count ≥ 1)`. Note: `≥ 1`, **not** `≥ min_cell_pts` — sparse sampling leaves 1–2 points per cell, but a real occlusion leaves *zero*.

1. **Find the gaps** (a cummax along the forward axis is "have I seen ground yet?"):

$$
\text{behind} = \text{cummax}_{+X}(\text{any\_pt}), \qquad
\text{ahead} = \text{cummax}_{-X}(\text{any\_pt})
$$

$$
\text{gap} = \neg\,\text{any\_pt} \;\wedge\; \text{behind} \;\wedge\; \text{ahead}
$$

2. **Discard occlusion shadows.** A gap behind which sits a lethal cell is the *shadow of a rock*, not a hole. The lethal mask is dilated laterally by one cell (a `1×3` kernel) before the cummax, so a rock shadows its neighbours' columns too:

$$
\text{gap} \;\mathrel{\&}= \neg\,\text{cummax}_{+X}\!\big(\text{dilate}_{1\times3}(\text{lethal})\big)
$$

3. **Discard honest sampling gaps.** A low camera looking far ahead *naturally* leaves gaps, because ground rows spread out with the square of range. The spacing between consecutive sampled ground rows is:

$$
\Delta x \;\approx\; \frac{\text{stride} \cdot r^2}{f_y \cdot h_{\text{cam}}}, \qquad r^2 = x^2 + y^2
$$

   *(Derivation: a ground point at range `x` images at row offset `v ≈ f_y h / x` from the horizon; differentiating, `|dx/dv| = x²/(f_y h)`; multiply by the pixel stride.)*

   Two guards follow from this:

$$
\text{min\_len}(c) = \max\Big(\texttt{hole\_min\_cells}=4,\; \Big\lceil \tfrac{3\,\Delta x}{\text{res}} \Big\rceil\Big)
\qquad\text{and}\qquad
\text{gap} \;\mathrel{\&}= \big(\Delta x \le 3\,\text{res}\big)
$$

   - Where the natural row spacing already exceeds 30 cm, **no hole verdict is issued at all**, whatever the run length. The camera is honestly sparse there and we say so.

4. **Range limit:** nothing beyond `hole_max_range` (9 m sim / webcam-scaled).

5. **Run-length test:** contiguous runs of `gap` along +X are labelled (`cumsum` of run-starts), and a run is converted to **LETHAL** only if its length ≥ `min_len`.

### Honest range

- The rule's reach is set by sampling density, not by wishful thinking:
  - sim's 320×180 depth, camera at 1.0 m → **~7 m**
  - MacBook webcam at 0.22 m → **~3 m**
- Beyond that the guard in step 3 fires and no verdict is given. That is the correct answer: *it cannot see further*.

---

## 12. Stage 7 — Inflation (giving the robot a body)

- A* plans for a **point**. A real robot is a **disc**. Inflation bridges the gap: grow every lethal cell by the robot's radius, so a point-path is automatically a body-safe path.

### The equations

1. Compute the Euclidean distance transform (OpenCV `distanceTransform`, `DIST_L2`) of the non-lethal cells, converted to metres:

$$ D(c) = \text{EDT}\big(\neg\,\text{lethal}\big) \times \text{res} \quad\text{[metres to the nearest lethal cell]} $$

2. Apply a two-band skirt with robot radius `r`:

$$
\text{skirt}(c) =
\begin{cases}
253 & D < r \qquad\qquad\quad \text{(the robot's body would overlap — near-lethal)}\\[4pt]
200\,e^{-2(D-r)/r} & r \le D < 2r \qquad\quad \text{(exponential decay — "prefer to stay clear")}\\[4pt]
0 & D \ge 2r
\end{cases}
$$

3. Merge:

$$
\text{measured cells:}\quad \text{cost} \leftarrow \max(\text{cost},\, \text{skirt})
$$
$$
\text{UNKNOWN cells:}\quad \text{cost} \leftarrow \begin{cases} 253 & D < r \\ 255 & \text{otherwise} \end{cases}
$$

- Worked values at `r = 1.0 m`: `D = 1.0 → 200`, `D = 1.35 → 99`, `D = 1.7 → 49`, `D = 2.0 → 27`, `D > 2.0 → 0`.

### Two design notes

- **253, not 254.** The inner skirt is *near*-lethal, not lethal. The planner treats only 254 as blocked, so a genuinely tight gap gets **squeezed through at high cost** rather than the plan failing outright. Graceful degradation beats a binary answer.
- **UNKNOWN inside the skirt becomes 253.** A never-measured cell right next to a trench edge is not a cheap place to drive. But only the *inner* skirt applies to unknowns — further out they stay `UNKNOWN`, so an unmeasured cell can never read as *cheap*.

---

## 13. Stage 8 — The global costmap (memory)

- The local costmap is **one frame**. It forgets everything the instant the robot turns.
- The global costmap is the **world-frame memory**. It is what makes real navigation (as opposed to reactive obstacle avoidance) possible.

| Property | Value |
|---|---|
| Resolution | 0.25 m |
| Extent | 160 m × 160 m, centred on the world origin |
| Array | **640 × 640** `uint8` |
| Origin of cell (0,0) | `−80.0, −80.0` in world metres |
| Initial state | all `UNKNOWN` |
| Decay | `0.0` = **remember forever** |

### Fusion (`GlobalCostmap.fuse`)

For every **measured** local cell `(ix, iy)`:

1. Cell centre in the robot frame:

$$ r_x = x_{\min} + (i_x + 0.5)\,\text{res}, \qquad r_y = y_{\min} + (i_y + 0.5)\,\text{res} $$

2. Rotate + translate by the robot's pose `(p_x, p_y, θ)` — the standard 2D rigid transform:

$$
\boxed{\;
\begin{aligned}
w_x &= p_x + r_x \cos\theta - r_y \sin\theta \\
w_y &= p_y + r_x \sin\theta + r_y \cos\theta
\end{aligned}\;}
$$

3. World metres → global cell:

$$ g_x = \left\lfloor \frac{w_x - \text{origin}}{\text{res}_g} \right\rfloor, \qquad
   g_y = \left\lfloor \frac{w_y - \text{origin}}{\text{res}_g} \right\rfloor $$

4. Write with **max**:

$$ G[g_x, g_y] \leftarrow \max\big(G[g_x, g_y],\; L[i_x, i_y]\big) $$

- **`UNKNOWN` local cells never write.** Unexplored ground stays unexplored; explored ground is never forgotten.
- Several local cells (10 cm) land in one global cell (25 cm). They are resolved by `max` too, using a sort + `np.maximum.reduceat` so the whole fusion is a handful of vectorised ops.
- The local grid arrives **already inflated**, so nothing inflates it again here.

### Backfilling the local map from memory (`fill_unknown_from_global`)

- Before local planning, every `UNKNOWN` cell in this frame's grid is replaced by whatever the global map remembers at that world position.
- **Why this is essential:**
  - The camera sees a *wedge*. Without memory, the planner happily routes through never-observed cells beside the robot.
  - A trench edge that was lethal one second ago **vanishes** as soon as the rover turns toward it — the exact moment it matters most.

---

## 14. Stage 9 — Global planning (A*)

### Preparing the search grid

- The full 640×640 map is too slow for pure Python (~0.5 s). Two reductions:
  1. **Max-pool by 2** → a 320×320 grid at **0.5 m** per cell. (`UNKNOWN` is treated as 0 while pooling and restored afterwards only if *every* sub-cell was unknown — so a single lethal sub-cell poisons the whole pooled cell. Conservative, deliberately.)
  2. **Box** the search to the bounding rectangle of start and goal, plus a **15 m margin** on each side.
- Result: ≤ ~200×200 cells, **~80 ms worst case**, and the global planner only runs at **≤ 1 Hz** anyway.

### Two pre-planning fixes

- **The robot is standing here, so here is drivable.** A disc of radius `robot_radius` around the start cell has any value ≥ 253 knocked down to 100. Fusion smear from earlier frames must never trap the robot at its own position.
- **A goal on an obstacle is still a direction.** If the goal cell is LETHAL it is softened to 253, so A* heads *toward* it and stops as close as it safely can, instead of refusing to plan.

### The A* itself

- **8-connected** grid search with a binary heap (`heapq`).
- Step lengths: `L = 1.0` orthogonal, `L = √2 ≈ 1.41421` diagonal.
- **Traversal cost split** (`traversal_cost`):

$$
c(\text{cell}) = \begin{cases}
\texttt{plan\_unknown\_cost} & \text{if the cell is UNKNOWN} \\
\text{grid value} & \text{otherwise}
\end{cases}
$$

$$
\text{blocked}(\text{cell}) \iff \text{value} \ge 254 \ \wedge\ \text{value} \ne 255
$$

  - Only **true LETHAL (254)** blocks. `253` (inflation) and `255` (unknown) are passable at cost.

- **Step cost** — the central planning equation:

$$
\boxed{\;g(j) = g(i) + L_{ij}\cdot\Big(1 + w_c \cdot \frac{c(j)}{255}\Big)\;}
$$

  with `w_c = plan_cost_weight = 6.0`.

- **Heuristic**: straight-line cell distance,

$$ h(i) = \sqrt{(g_x - i_x)^2 + (g_y - i_y)^2} $$

- **Admissibility**: because the multiplier `1 + w_c·c/255 ≥ 1` **everywhere**, no step can ever cost less than its geometric length. The Euclidean heuristic therefore never overestimates → **A* is guaranteed to find the optimal path**.

### What the weights actually mean

Multiplier `= 1 + 6·c/255`:

| Cell | cost `c` | Local multiplier | Global multiplier |
|---|---|---|---|
| Road | 0 | **1.00×** | 1.00× |
| Grass | 40 | 1.94× | 1.94× |
| Unknown | `plan_unknown_cost` | **5.71×** (`c`=200) | **1.94×** (`c`=40) |
| Inflation skirt | 253 | 6.95× | 6.95× |
| Lethal | 254 | **blocked** | **blocked** |

- Read that as: **locally, crossing unmapped ground costs 5.7× as much as driving on a known road — the robot will do it, but only if there is no measured alternative.**
- Globally, unknown is only 1.94× — at map scale, unexplored ground is *not* hazardous, it is simply unexplored, and the robot must be willing to head into it or it can never explore.

### Graceful degradation

- When the goal is unreachable, A* does **not** return failure. It returns the path to the expanded cell **closest to the goal**, with `reached = False`.
- The robot still makes progress and re-plans next frame, instead of freezing because something 8 m ahead is walled.

### Replanning triggers

- Every `replan_period = 1.0 s`, **or**
- when the current path has no cached value, **or**
- when `path_blocked()` finds that any point of the stored world path now sits on a LETHAL cell.

---

## 15. Stage 10 — The carrot (global → local hand-off)

- The global path can be 60 m long. The local grid is 12 m. Something must bridge them: that is the **carrot**.

$$
\text{ahead} = \max\big(x_{\max} - 1.5,\; x_{\min} + 1.0\big) \;=\; 10.5\ \text{m (sim)}
$$

**The rule:**

- If the **final goal** is already within `ahead` metres → the carrot **is** the goal (converted to the robot frame). The robot drives at the real thing.
- Otherwise → the **first point on the global path** whose distance from the robot is ≥ `ahead`.
- If there is no global path at all (webcam mode, no pose) → the goal itself, interpreted directly as a robot-frame point.

World → robot conversion (the inverse of the fusion transform):

$$
\begin{aligned}
r_x &= \phantom{-}(w_x - p_x)\cos\theta + (w_y - p_y)\sin\theta \\
r_y &= -(w_x - p_x)\sin\theta + (w_y - p_y)\cos\theta
\end{aligned}
$$

- The carrot is returned **unclamped** — it may be behind or beside the robot. The Navigator decides what that means (see §17), which is exactly the information the turn-in-place logic needs.
- Only for the local A* is it clamped into the grid: `clip(cx, x_min + res, x_max − res)`.

---

## 16. Stage 11 — Local planning + pure pursuit → (v, ω)

### Local A*

- Same `astar()` function, now on the **10 cm** grid, with the carrot as goal.
- Start cell = `(0, iy(y=0))` — row 0, dead centre. The robot itself sits *behind* the grid (`x_min = 0.5 m`), so the plan begins at the nearest measured row.
- **Immediate-stop guard:** if the start cell is blocked (something lethal 0.5 m dead ahead), A* returns an empty path immediately. No path → full stop. That is the only correct default.
- Grid path → metres:

$$ (x, y) = \big(x_{\min} + i_x\,\text{res},\; y_{\min} + i_y\,\text{res}\big) $$

### Pure pursuit

- **The idea in one sentence:** pick a point on the path a fixed distance ahead, then drive the perfect circular arc that passes through it.

**Step 1 — choose the aim point.** The first path point at least `lookahead` metres from the robot at the origin:

$$ \text{aim} = \min\big\{k : \sqrt{x_k^2 + y_k^2} \ge \texttt{lookahead}\big\} $$

- `lookahead = 2.5 m` (sim), `1.5 m` (webcam). Larger = smoother but sloppier cornering.

**Step 2 — curvature.** Let the aim point be `(x, y)` at distance `d = √(x² + y²)`. The unique circle through the origin, tangent to the robot's current heading (the X axis), that also passes through `(x, y)`, has radius `R = d²/(2y)`. Therefore:

$$
\boxed{\;\kappa = \frac{1}{R} = \frac{2y}{d^2}\;}
$$

- `y > 0` (aim point to the left) → `κ > 0` → turn left. Signs are consistent throughout.

**Step 3 — speed.** Two independent de-ratings:

$$
v = \underbrace{\frac{v_{\max}}{1 + \texttt{turn\_slow}\cdot|\kappa|}}_{\text{slow down in tight turns}} \times \underbrace{\min\Big(1, \frac{\text{reach}}{\texttt{lookahead}}\Big)}_{\text{slow down if the plan is short}}
$$

- `reach = ‖last path point‖` — how far the plan **actually gets**.
- **Why the reach term matters:** a path that dead-ends 0.8 m ahead means the way is blocked, however straight its first metre looks. Without this term the robot drives at `v_max` into a wall, because the aim point itself is clear.
- Hard stop: `reach < stop_dist` (1.2 m sim / 0.7 m webcam) → `v = 0, ω = 0`.
- Also `x ≤ 0` (aim point behind) → full stop; that case is handled by turn-in-place instead.

**Step 4 — turn rate.**

$$ \boxed{\;\omega = \operatorname{clip}\big(v\cdot\kappa,\; -\omega_{\max},\; +\omega_{\max}\big)\;} $$

- This is the exact differential-drive relation: for a body moving at speed `v` along a curve of curvature `κ`, the yaw rate *is* `v·κ`.
- **Positive `ω` = turn left**, matching ROS REP-103.

**Step 5 — goal approach, smoothing and acceleration limits.**

$$ \text{if } \text{dist} < \texttt{slow\_dist}: \quad v \mathrel{\times}= \max\Big(0.25,\; \frac{\text{dist}}{\texttt{slow\_dist}}\Big) $$

$$ \omega \leftarrow \texttt{cmd\_smooth}\cdot\omega_{\text{prev}} + (1-\texttt{cmd\_smooth})\cdot\omega, \qquad \texttt{cmd\_smooth} = 0.5 $$

$$ v \leftarrow \operatorname{clip}\big(v,\; v_{\text{prev}} - 2 a\,\Delta t,\; v_{\text{prev}} + a\,\Delta t\big), \qquad a = \texttt{accel\_max} = 1.5\ \text{m/s}^2 $$

- Braking is allowed at **twice** the acceleration limit — asymmetric on purpose. Stopping is always permitted to be more aggressive than starting.
- **Why smooth `ω` at all?** The local path is re-planned from scratch every frame on a 10 cm grid, so raw pure-pursuit `ω` steps by ~0.2 rad/s frame to frame. That reads as a visible zig-zag. The EMA removes it.
- On the simulator side, the rover **applies each command for one control period and then holds heading** until the next arrives. That, plus the EMA, is what eliminated the zig-zag entirely.

### The output

- `(v, ω)` in SI units — m/s and rad/s — which is precisely `geometry_msgs/Twist`:

```json
{"linear": {"x": 1.2, "y": 0, "z": 0}, "angular": {"x": 0, "y": 0, "z": -0.15}}
```

- For a **differential drive** robot the wheel speeds follow immediately (`W` = track width):

$$ v_L = v - \frac{\omega W}{2}, \qquad v_R = v + \frac{\omega W}{2} $$

- For **Ackermann** steering: `δ = atan(L·κ)` for wheelbase `L`.

---

## 17. Stage 12 — The state machine and recoveries

`Navigator.step()` runs once per perceived frame and is always in exactly one state:

| State | Entered when | Output |
|---|---|---|
| `NO_GOAL` | no destination set | `v=0, ω=0` |
| `PLANNING` | goal just set, first global plan pending | — |
| `TURNING` | carrot is behind or far off-axis | `v=0`, `ω = clip(turn_gain·β, ±ω_max)` |
| `DRIVING` | normal operation | pure pursuit `(v, ω)` |
| `BLOCKED` | local A* found no safe first step | `v=0`, then spin recovery |
| `ARRIVED` | within `goal_tol` of the goal | `v=0, ω=0` |
| `STOPPED` | watchdog — no frame for 1 s | `v=0, ω=0` |

### Turn-in-place (hysteresis)

- Bearing to the carrot: `β = atan2(c_y, c_x)`.
- **Enter** `TURNING` if `c_x < turn_min_x` (1.0 m — the carrot is beside or behind us) **or** `|β| > turn_enter_deg` (70° sim / 60° default).
- **Exit** only when `|β| < turn_exit_deg` (15°).
- The gap between 70° and 15° is **hysteresis** — without it the robot chatters between turning and driving at the threshold.
- While turning, `v_prev` and `ω_prev` are reset to 0 so the smoothing filter does not carry stale motion into the next drive.

### Blocked recovery

- `blocked_frames = 3` consecutive frames with no path → escalate.
- Then up to `recovery_frames = 20` frames of **spin recovery**: `v = 0`, `ω = ±0.5·ω_max`, rotating toward the side the goal is on.
- Spinning is the correct recovery for a *vision* robot specifically: the camera sees a wedge, and rotating is the only way to acquire new information about what is beside you.
- After 20 frames the counters reset and it tries again from scratch.

### Watchdog

- If no frame arrives for `watchdog = 1.0 s`, the server broadcasts `status: "STOPPED"` with `v=0, ω=0`.
- This covers the browser tab being backgrounded, the WebSocket stalling, or the perception thread dying. **A robot that has lost its eyes must stop.**

### Reset on reconnect

- A new `hello` with `role: "sim"`, or a **regressing frame sequence number**, resets the global map, the goal and the plane estimator. That is a page reload, and stale memory from a previous run is worse than no memory.

---

## 18. How this maps onto Nav2, exactly

### The architectural correspondence

Nav2 splits navigation into a **global** half (where to go, coarse, whole map) and a **local** half (how to move right now, fine, around the robot). This project keeps *exactly* that split, so a real Nav2 can be swapped in later without redesigning anything.

| Nav2 component | This project | Where |
|---|---|---|
| `global_costmap` (static + obstacle + inflation layers) | `GlobalCostmap` — max-fusion of inflated local grids in the world frame | `navstack.py` |
| `planner_server` (NavFn / Smac / Theta*) | `plan_global()` — 8-connected A* on a 2× pooled, boxed copy | `navstack.py` |
| `local_costmap` (rolling window) | `perception_core` grid + `fill_unknown_from_global()` | `perception_core.py` |
| `controller_server` (DWB / Regulated Pure Pursuit) | local `astar()` + `drive_command()` pure pursuit | `costmap_prototype.py` |
| `behavior_server` (spin / backup / wait) | `BLOCKED` spin recovery, watchdog stop | `navstack.py` |
| `bt_navigator` (behaviour tree) | `Navigator` state machine | `navstack.py` |
| `amcl` / `slam_toolbox` | **`PoseSource`** — the VSLAM seam | `navstack.py` |
| `/cmd_vel` topic | `cmd` in the `nav` message + `GET /ros/cmd_vel` | `perception_server.py` |
| `costmap_2d` inflation layer | `perception_core.inflate()` | `perception_core.py` |

- Note that our **inflation happens in the local costmap, before fusion**, so the global map is already body-safe. Nav2 inflates in each costmap separately; either is valid, ours is one fewer pass.

### The numeric translation: our bytes → `nav_msgs/OccupancyGrid`

Nav2 uses a signed `int8` in `[-1, 100]`. We use a `uint8` in `[0, 255]`. The conversion is exact and lossless where it matters (`ros_msgs.grid_to_occupancy`):

$$
\text{occ} =
\begin{cases}
-1 & \text{if } g = 255 \quad \text{(UNKNOWN)} \\
100 & \text{if } g \ge 254 \quad \text{(LETHAL)} \\
99 & \text{if } g = 253 \quad \text{(inflation skirt)} \\
\left\lfloor g \cdot \dfrac{98}{252} \right\rceil & \text{otherwise} \quad (0..252 \rightarrow 0..98)
\end{cases}
$$

| Our value | Meaning | OccupancyGrid |
|---|---|---|
| `255` | UNKNOWN | `-1` |
| `254` | LETHAL | `100` |
| `253` | inflation skirt | `99` |
| `0..252` | graded cost | `0..98` |

- **Why 253 → 99 and not 100:** so Nav2's *own* inflation layer still sees it as near-lethal but distinguishable, and does not double-inflate it into a hard obstacle.
- **Why cap graded cost at 98:** so nothing in the graded band can ever be mistaken for the inflation skirt or for lethal.

### The layout translation

- **Our grids** are `axis 0 = X`, `axis 1 = Y`.
- **`OccupancyGrid.data`** is row-major with **X varying fastest**: `index = y·width + x`.
- So the array is **transposed** before flattening: `occ = grid_to_occupancy(grid).T` → shape `(ny, nx)` → `.ravel()`.
- `info.origin` is the world position of the grid's `(0,0)` **corner**, not its centre:
  - local grid → `(x_min, y_min)` = `(0.5, −5.0)`, frame `base_link`
  - global grid → `(−80.0, −80.0)`, frame `map`

### The live endpoints

| HTTP | ROS 2 message | Frame | Contents |
|---|---|---|---|
| `GET /ros/occupancy_grid` | `nav_msgs/OccupancyGrid` | `base_link` | the local costmap, 115×100 @ 0.1 m |
| `GET /ros/global_grid` | `nav_msgs/OccupancyGrid` | `map` | the global costmap, 640×640 @ 0.25 m |
| `GET /ros/odometry` | `nav_msgs/Odometry` | `odom` → `base_link` | pose (yaw as a quaternion) + current twist |
| `GET /ros/path` | `nav_msgs/Path` | `map` (or `base_link`) | the global path, each pose oriented along the path tangent |
| `GET /ros/cmd_vel` | `geometry_msgs/Twist` | — | the current `(v, ω)` |

- Yaw → quaternion, since ROS never stores a bare angle:

$$ q = \Big(0,\; 0,\; \sin\tfrac{\theta}{2},\; \cos\tfrac{\theta}{2}\Big) $$

- Path pose orientations use the **tangent** to the next point: `yaw_i = atan2(y_{i+1} − y_i, x_{i+1} − x_i)`.

### The three ways to actually integrate with ROS 2

1. **Publish our costmaps, let Nav2 plan.** Bridge `/ros/occupancy_grid` and `/ros/global_grid` onto `/local_costmap/costmap` and `/global_costmap/costmap` via `rosbridge_suite`, publish `/odom`, send a `NavigateToPose` action goal, and read Nav2's `/cmd_vel` back. Our perception, Nav2's planning.
   - This is the intended production path. Our contribution — *turning a camera into a costmap* — is the part Nav2 does not have.

2. **Publish everything, use Nav2 only for the behaviour tree.** Also publish `/plan` and `/cmd_vel` from our stack; Nav2 supervises recoveries and lifecycle.

3. **Standalone (what runs today).** No ROS installed at all. The Python stack is the whole navigation system, and the `/ros/*` endpoints exist so the shapes are proven correct.

- Practically: `roslibpy.Topic(...).publish(msg)` takes these dicts verbatim. There is no conversion code left to write.

### Where the VSLAM plugs in

```python
class MyVisualSLAM(navstack.PoseSource):
    def get(self) -> Optional[navstack.Pose]:
        return navstack.Pose(x, y, theta)   # world frame, metres and radians
```

- That is the **entire** integration surface for problem #2. Everything downstream — fusion, global planning, the carrot, the world-frame goal — already consumes it.
- In ROS terms this is what `amcl` or `slam_toolbox` publishes as the `map → odom → base_link` transform chain.

---

## 19. A fully worked numeric example (one pixel → one wheel command)

Concrete numbers, simulator configuration, one frame.

### Setup

| Quantity | Value |
|---|---|
| Image | 640 × 360 |
| Vertical FOV | 60° |
| `fx = fy` | `180 / tan(30°) = 311.77 px` |
| `cx, cy` | `320, 180` |
| True mount (unknown to the software) | height 1.0 m, pitch 15° down, roll 0° |

### Step 1 — the plane is measured

RANSAC + refinement returns:

$$ \mathbf{n} = (0,\; -0.9659,\; -0.2588), \qquad d = 1.002 $$

- Read back: `pitch = atan2(0.2588, 0.9659) = 15.0°`, `roll = atan2(0, 0.9659) = 0.0°`, `height = 1.002 m`.
- Reported error against the mount: **+2 mm, −0.0°**. That is the live accuracy check on the HUD.

### Step 2 — one pixel

Take pixel `(u, v) = (320, 300)` with measured depth `z = 1.00 m`.

$$ x_n = \frac{320 - 320}{311.77} = 0, \qquad y_n = \frac{300 - 180}{311.77} = 0.3849 $$

$$ \mathbf{p} = (X_c, Y_c, Z_c) = (0,\; 0.3849,\; 1.000) $$

### Step 3 — height above ground

$$ Z = \mathbf{n}\cdot\mathbf{p} + d = (-0.9659)(0.3849) + (-0.2588)(1.000) + 1.002 $$
$$ Z = -0.3718 - 0.2588 + 1.002 = \mathbf{+0.371\ \text{m}} $$

- **Cross-check:** solving `n·p + d = 0` for that same ray gives `z = 1.589 m` — that is where the *ground* is along this pixel's ray. The measurement came back at 1.00 m, i.e. something is **59 cm nearer than the floor**, which at this viewing angle stands **37 cm proud** of it. Consistent. ✓

### Step 4 — position on the grid

$$ \mathbf{f} = (0,\; -0.2588,\; 0.9659), \qquad \boldsymbol{\ell} = (-1,\; 0,\; 0) $$

$$ X = \mathbf{f}\cdot\mathbf{p} = (-0.2588)(0.3849) + (0.9659)(1.000) = 0.866\ \text{m forward} $$
$$ Y = \boldsymbol{\ell}\cdot\mathbf{p} = 0.000\ \text{m left} $$

$$ i_x = \left\lfloor \frac{0.866 - 0.5}{0.1} \right\rfloor = 3, \qquad i_y = \left\lfloor \frac{0 - (-5.0)}{0.1} \right\rfloor = 50 $$

- So this pixel votes in cell `grid[3, 50]` — 0.87 m ahead, dead centre.

### Step 5 — the cell is scored

Suppose that cell collects `N = 34` points, of which 19 have `Z > 0.25 m`, and the segmenter called 26 of them "rock" (cost 250):

- `min_sup = clip(0.20 × 34, 2, 6) = 6`. `N_pos = 19 ≥ 6` → **geometry says LETHAL**.
- `vote_min = max(0.25 × 34, 2) = 8.5`. `N_tall = 26 ≥ 8.5`, and `flat_cell` is false (points above 12.5 cm exist) → **semantics also says LETHAL (254)**.
- `cost = max(254, 254) = 254`. Both channels agree — this is the normal, healthy case.

### Step 6 — inflation

- With `robot_radius = 1.0 m` and `res = 0.1 m`, this one lethal cell blackens a **disc ~20 cells across** at 253, surrounded by a decaying skirt out to 2.0 m.
- The rock is now 0.87 m ahead **and the robot's body is accounted for**.

### Step 7 — planning

- The goal is 40 m away. Global A* on the 0.5 m pooled map returns a world path that curves left around the rock cluster.
- The carrot is the first path point ≥ 10.5 m out: in the robot frame, `(9.8, 3.1)` m.
- Bearing `β = atan2(3.1, 9.8) = 17.6°` — under the 70° threshold, so we stay in `DRIVING`.
- Local A* runs to the clamped carrot cell and returns ~110 cells that swing left of the inflated rock.

### Step 8 — pure pursuit

- First path point at ≥ 2.5 m: `(2.38, 0.76)` m.

$$ d = \sqrt{2.38^2 + 0.76^2} = 2.498\ \text{m}, \qquad \kappa = \frac{2 \times 0.76}{2.498^2} = 0.2436\ \text{m}^{-1} $$

- Path reach (last point): `‖(9.7, 3.0)‖ = 10.15 m`, well past `stop_dist = 1.2 m`.

$$ v = \frac{2.0}{1 + 0.6 \times 0.2436} \times \min\!\Big(1, \frac{10.15}{2.5}\Big) = \frac{2.0}{1.1462} \times 1 = 1.745\ \text{m/s} $$

$$ \omega = \operatorname{clip}(1.745 \times 0.2436,\; \pm 0.8) = +0.425\ \text{rad/s} $$

- After EMA smoothing against a previous `ω_prev = 0.38`: `ω = 0.5(0.38) + 0.5(0.425) = **0.403 rad/s**`.
- After the acceleration limit from `v_prev = 1.60` with `Δt = 0.2 s`: `v ≤ 1.60 + 1.5(0.2) = 1.90`, so `v = **1.745 m/s**` passes unchanged.

### Step 9 — output

```json
{"type": "nav", "status": "DRIVING",
 "cmd": {"v": 1.745, "omega": 0.403},
 "twist": {"linear": {"x": 1.745, "y": 0, "z": 0},
           "angular": {"x": 0, "y": 0, "z": 0.403}}}
```

- Turning gently left at 1.7 m/s, around a rock it measured 0.87 m ahead. Total latency: ~70 ms.
- For a differential drive with a 0.8 m track: `v_L = 1.745 − 0.161 = 1.584 m/s`, `v_R = 1.745 + 0.161 = 1.906 m/s`.

---

## 20. Every constant, in one place

### Perception (`CoreCfg`)

| Parameter | Sim | Webcam | Meaning |
|---|---|---|---|
| `w, h` | 640, 360 | 640, 360 | processing resolution |
| `fx, fy` | 311.77 (sent per frame) | 395.2 (78° hfov) | focal length, pixels |
| `x_min, x_max` | 0.5, 12.0 | 0.3, 8.0 | forward extent, m |
| `y_min, y_max` | −5.0, 5.0 | −4.0, 4.0 | lateral extent, m |
| `res` | 0.10 | 0.10 | cell size, m |
| `stride` | 2 | 2 | pixel subsampling |
| `min_depth, max_depth` | 0.2, 20.0 | 0.2, 12.0 | depth validity, m |
| `obstacle_h` | 0.25 | 0.25 | positive obstacle threshold, m |
| `ditch_h` | −0.20 | −0.20 | negative obstacle threshold, m |
| `max_obstacle_h` | 2.0 | 2.0 | above this = overhead clearance |
| `ditch_max_range` | 9.0 | 8.0 | trust limit for drops, m |
| `hole_min_cells` | 4 | 4 | minimum hole run length |
| `hole_max_range` | 9.0 | 9.0 | hole-rule range limit, m |
| `min_cell_pts` | 3 | 3 | below this → UNKNOWN |
| `geo_min_pts` | 2 | 2 | absolute geometry consensus |
| `geo_min_frac` | 0.20 | 0.20 | fractional geometry consensus |
| `sem_lethal_frac` | 0.25 | 0.25 | semantic lethal vote threshold |
| `robot_radius` | 1.0 | 0.35 | inflation radius, m |

### Ground plane

| Parameter | Value | Meaning |
|---|---|---|
| `plane_near_range` | 6.0 (sim) / 5.0 | preferred fit range, m |
| `plane_max_range` | 10.0 | widened fit range, m |
| `plane_lower_frac` | 0.65 | candidates from the lower 65% of the image |
| `plane_fallback_frac` | 0.40 | fallback if semantics gives nothing |
| `plane_min_pts` | 300 | minimum candidates |
| `plane_gate` | 0.05 m | inlier band base |
| `plane_gate_rel` | 0.01 | + this × depth |
| `plane_gate_max` | 0.10 m | hard cap (must stay below \|ditch_h\|) |
| `plane_iters` | 150 | RANSAC hypotheses |
| `plane_max_pts` | 4000 | candidate subsample cap |
| `plane_max_pitch_deg` | 60 | steeper = a wall, reject |
| `plane_max_roll_deg` | 45 | reject |
| `plane_min/max_height` | 0.03 / 5.0 m | plausible camera height |
| `plane_ema` | 0.5 | temporal blend |
| `plane_jump_deg` | 8.0 | jump gate, angle |
| `plane_jump_m` | 0.15 | jump gate, height |
| `plane_jump_conf` | 0.60 | confidence needed to accept a jump |
| `plane_hold_frames` | 5 | hold before relocking |
| `plane_low_conf` | 0.30 | flag threshold |

### Navigation (`NavCfg` / `PlannerCfg`)

| Parameter | Sim | Webcam | Meaning |
|---|---|---|---|
| `v_max` | 2.0 | 1.0 | m/s |
| `w_max` | 0.8 | 1.0 | rad/s |
| `goal_tol` | 1.2 | 1.0 | ARRIVED radius, m |
| `slow_dist` | 4.0 | 3.0 | start slowing here, m |
| `lookahead` | 2.5 | 1.5 | pure pursuit, m |
| `stop_dist` | 1.2 | 0.7 | minimum path reach, m |
| `turn_gain` | 1.0 | 1.5 | rad/s per rad of bearing |
| `turn_enter_deg` | 70 | 60 | enter TURNING |
| `turn_exit_deg` | 15 | 15 | exit TURNING |
| `turn_min_x` | 1.0 | 1.0 | carrot nearer than this → turn, m |
| `turn_slow` | 0.6 | 0.6 | curvature speed de-rating |
| `cmd_smooth` | 0.5 | 0.5 | ω EMA |
| `accel_max` | 1.5 | 1.5 | m/s² |
| `blocked_frames` | 3 | 3 | before recovery |
| `recovery_frames` | 20 | 20 | spin recovery length |
| `replan_period` | 1.0 | — | s between global replans |
| `watchdog` | 1.0 | 1.0 | s without a frame → STOP |
| `plan_unknown_cost` (local) | 200 | 200 | UNKNOWN cost, local A* |
| `plan_unknown_cost` (global) | 40 | — | UNKNOWN cost, global A* |
| `plan_cost_weight` | 6.0 | 6.0 | cost → step-cost multiplier |
| global map res / size | 0.25 m / 160 m | — | 640 × 640 |
| global `pool` / `margin_m` | 2 / 15.0 | — | planning downsample and box |

### Live-tunable at runtime

- Via the dashboard or a `set_param` WebSocket message: `obstacle_h`, `ditch_h`, `robot_radius`, `sem_lethal_frac`, `min_cell_pts`, `plane_gate`, `plane_near_range`, `max_depth`.

---

### The rover rig, and why its numbers differ

The simulator's camera sits 1 m up. A real UGV camera sits about **0.17 m** up, and that one
change resizes the whole map. Ground samples thin out as `stride · r² / (fy · h)` — the same
expression the hole rule already uses (§11) — so at `h = 0.17` and `fy = 530`:

| range | gap between ground rows |
|---|---|
| 1 m | 1.1 cm |
| 2 m | 4.4 cm |
| 4 m | 17.7 cm |
| 8 m | **71 cm** |

The whole of 3→8 m lands in roughly **13 pixel rows**, where a single pixel of error is a
quarter-metre of range. An 8 m map at this mount height is not sparse — it is fiction. So
`rover_cfg()` sizes the map to what the optics can actually support:

| constant | sim / webcam | rover | why |
|---|---|---|---|
| `x_max` | 8.0–12.0 m | **2.60 m** | the honest horizon; past 3 m one pixel is 0.1–0.7 m of range |
| `res` | 0.10 m | **0.05 m** | a 0.25 m chassis cannot be planned on a 0.10 m grid |
| `stride` | 2 | **1** | at stride 2 the hole rule's own gate switches ditches off at 2.2 m |
| `obstacle_h` | 0.25 m | **0.10 m** | a rover is stopped by what a car drives over |
| `ditch_h` | −0.20 m | **−0.08 m** | ditto |
| `plane_gate_max` | 0.10 m | **0.05 m** | the invariant of §8: must stay below \|`ditch_h`\| |
| `robot_radius` | 0.35 m | **0.20 m** | the actual chassis |
| `nominal_height` | 0.60 m | **0.17 m** | tape-measured; this one number sets the metric scale |

Grid: **48 × 52 cells**. Intrinsics at 640×480 from the full-FOV sensor mode:
**fx = fy = 530.5, cx = 320, cy = 240** (62.2° × 48.8°).

Two consequences worth stating. The hole-rule cutoff works out at **3.68 m**, comfortably past
the 2.6 m horizon — so unlike the sim, negative-obstacle detection is live across the *entire*
map. And the frame is **4:3, deliberately**: the binding constraint here is how many ground
*rows* the sensor gets, and vertical field of view is what buys them. Cropping to 16:9 would
throw away a third of the ground for nothing.

A 2.6 m horizon sounds alarming until you price it: at 0.8 m/s, with ~140 ms of link-and-compute
latency and 1 m/s² braking, the rover stops in about 0.5 m. That is a five-fold margin, and it
forces the architecture Nav2 intended anyway — a small dense local map, with everything beyond it
coming from an accumulated global one.

---

## 21. The three safety rules, and why they exist

Every non-obvious decision in this codebase traces back to one of three rules. Each was written *after* watching the alternative fail.

### Rule 1 — `cost = max(...)`, never average

- If **any** channel says a cell is dangerous, the cell is dangerous.
- A false positive costs a detour. A false negative costs the robot.
- This applies at three levels: across the four cost channels, across points landing in the same global cell, and across frames in `GlobalCostmap.fuse`.

### Rule 2 — UNKNOWN is expensive but passable

- Treat UNKNOWN as **free** → the robot drives confidently into ditches that simply had no depth measurement.
- Treat UNKNOWN as **blocked** → the robot freezes, because the far half of every monocular frame is *always* partly unmeasured.
- **Expensive but passable** (cost 200 locally, 5.7× a road) means it crosses unmapped ground **only when no measured route exists** — which is exactly the behaviour we want.
- The hole rule (§11) is the narrow, evidence-backed exception where UNKNOWN is promoted to LETHAL.

### Rule 3 — measure, do not configure

- Camera height, pitch and roll are **estimated every frame** from the data itself.
- `--height` / `--pitch` / `--roll` exist only as optional locks, never as requirements.
- Everything downstream — obstacle thresholds, ditch thresholds, the hole rule, the whole grid — is measured against a datum the software **derived**, not one a human typed in.
- This is what makes the same code run unchanged on a MacBook lid at 22 cm and a simulated rover mast at 1 m.

---

## 22. Honest limitations

- **No visual SLAM yet.** `PoseSource` is the interface; the simulator supplies ground truth through it. In webcam mode there is no pose at all, so there is **no global map** and goals are robot-relative. This is problem #2 of the statement, and it is not solved — only wired for.
- **Monocular metric depth degrades beyond ~8–10 m**, and badly on textureless synthetic ground. The sim's `--depth sim` is the stand-in for a stereo rig at the identical interface, so the fix is hardware, not code.
- **The hole rule's range is set by sampling density.** ~7 m with the sim's 320×180 depth at 1 m camera height; ~3 m for a 22 cm webcam. Beyond that it deliberately issues no verdict rather than guessing.
- **Semantics on synthetic imagery is approximate.** The tall-label demotion protects drivable ground from mislabels, but water / mud / sand grading depends entirely on the segmenter's quality.
- **The planners are pure Python.** The global A* is pooled and boxed specifically to stay under ~100 ms; a C++ or `numba` implementation would remove that constraint.
- **Dynamic obstacles are handled reactively, not predictively.** A moving person is re-observed and re-planned around every frame, but there is no velocity estimate and no trajectory prediction.
- **The costmap is 2D.** Overhangs are handled crudely by the `max_obstacle_h = 2.0 m` ceiling; a genuine 2.5D or voxel representation would do better on bridges, low branches and stairs.
- **On real rover hardware there is still no pose**, so `--source rover` runs with no global map, no temporal fusion, and a robot-frame carrot goal. With a 2.6 m sensing horizon that matters more than it did in the sim: everything beyond 2.6 m *is* memory, and there is none yet.
- **The `affine` depth path is not usable on its own.** See §5: the shift is not identifiable from flat ground, so it needs an orientation source depth cannot bias — an IMU or non-coplanar SLAM points. `rover_cfg` ships `affine_depth=False` and the solver refuses rather than guessing.
- **No motor-control layer.** `rover_agent.py` parses `cmd.v` / `cmd.omega` and runs a staleness watchdog, but `apply_cmd()` is a stub. Driving the ESCs is deliberately out of scope for now.
- **The IMX219 sensor-mode trap is a live foot-gun.** Asking picamera2 for a 640×480 *sensor* mode silently crops the field of view from 62.2° to ~26.5° with no error raised. `rover_agent.py` pins the full-FOV 1640×1232 mode; anything else reading the camera must do the same.
- **Rolling shutter on a chassis with no suspension.** Every bump reaches the sensor directly, and depth from a blurred or skewed frame is unrecoverable downstream. A damped camera mount is a real requirement, not a refinement.

---

## Appendix — every equation, on one page

| # | Name | Equation |
|---|---|---|
| 1 | Intrinsics from hfov | `fx = (W/2) / tan(hfov/2)` |
| 2 | Intrinsics from vfov | `fy = (H/2) / tan(vfov/2)` |
| 3 | Sim depth linearisation | `z_view = (n·f) / ((f−n)·v_ndc − f)` |
| 4 | Back-projection | `Xc = ((u−cx)/fx)·z`, `Yc = ((v−cy)/fy)·z`, `Zc = z` |
| 5 | Plane | `n·p + d = 0`, `d = camera height` |
| 6 | Normal from angles | `n = (sin r·cos p, −cos r·cos p, −sin p)` |
| 7 | Pitch from normal | `pitch = atan2(−n_z, √(n_x²+n_y²))` |
| 8 | Roll from normal | `roll = atan2(n_x, −n_y)` |
| 9 | RANSAC hypothesis | `n = (B−A)×(C−A) / ‖·‖`, `d = −n·A` |
| 10 | Inlier gate | `\|n·p + d\| ≤ min(0.05 + 0.01·Z, 0.10)` |
| 11 | RANSAC score | `Σ_inliers 1 / max(Z, 0.5)` |
| 12 | Plane EMA | `n ← normalise(α·n_new + (1−α)·n_prev)`, `α = 0.5` |
| 13 | Ground basis | `f = normalise(ẑ − (ẑ·n)n)`, `ℓ = n × f`, `R = [f; ℓ; n]` |
| 14 | Height above ground | `Z = n·p + d` |
| 15 | Grid index | `ix = ⌊(X − x_min)/res⌋`, `iy = ⌊(Y − y_min)/res⌋` |
| 16 | Semantic vote threshold | `vote_min = max(0.25·N, 2)` |
| 17 | Geometry consensus | `min_sup = clip(0.20·N, 2, 6)` |
| 18 | Flat-cell test | no point `≥ 0.5·obstacle_h`, none `≤ 0.5·ditch_h` |
| 19 | Fusion | `cost = max(semantic, geometry, hole)` |
| 20 | Hole gap | `gap = ¬any_pt ∧ cummax₊ₓ(any_pt) ∧ cummax₋ₓ(any_pt)` |
| 21 | Ground row spacing | `Δx = stride·r² / (fy·h_cam)` |
| 22 | Hole run length | `min_len = max(4, ⌈3·Δx/res⌉)`, require `Δx ≤ 3·res` |
| 23 | Inflation skirt | `253` if `D<r`; `200·e^{−2(D−r)/r}` if `r≤D<2r`; else `0` |
| 24 | Robot → world | `wx = px + rx·cosθ − ry·sinθ`, `wy = py + rx·sinθ + ry·cosθ` |
| 25 | World → robot | `rx = Δx·cosθ + Δy·sinθ`, `ry = −Δx·sinθ + Δy·cosθ` |
| 26 | Global fusion | `G[c] ← max(G[c], L[c])`, UNKNOWN never writes |
| 27 | A* step cost | `g(j) = g(i) + L·(1 + 6·c(j)/255)` |
| 28 | A* heuristic | `h = √((gx−ix)² + (gy−iy)²)` (admissible: multiplier ≥ 1) |
| 29 | Carrot distance | `ahead = max(x_max − 1.5, x_min + 1.0)` |
| 30 | Pure-pursuit curvature | `κ = 2y / d²`, `d = √(x²+y²)` |
| 31 | Speed | `v = v_max/(1 + turn_slow·\|κ\|) · min(1, reach/lookahead)` |
| 32 | Turn rate | `ω = clip(v·κ, ±ω_max)` |
| 33 | Goal slowdown | `v ×= max(0.25, dist/slow_dist)` |
| 34 | ω smoothing | `ω ← 0.5·ω_prev + 0.5·ω` |
| 35 | Acceleration limit | `v ← clip(v, v_prev − 2aΔt, v_prev + aΔt)`, `a = 1.5` |
| 36 | Turn-in-place | `β = atan2(cy, cx)`; enter if `cx<1.0` or `\|β\|>70°`; exit at `\|β\|<15°` |
| 37 | Turn-in-place rate | `ω = clip(turn_gain·β, ±ω_max)` |
| 38 | Differential drive | `v_L = v − ωW/2`, `v_R = v + ωW/2` |
| 39 | Ackermann | `δ = atan(L·κ)` |
| 40 | Cost → OccupancyGrid | `255→−1`, `≥254→100`, `253→99`, else `round(g·98/252)` |
| 41 | Yaw → quaternion | `q = (0, 0, sin(θ/2), cos(θ/2))` |
| 42 | Lens rectification | `x_d = x(1 + k₁r² + k₂r⁴ + k₃r⁶) + 2p₁xy + p₂(r²+2x²)`, applied before eq. 3 |
| 43 | Affine depth | `1/Z = a·disp + b` — **`(a, b)` are NOT determined by flat ground alone** |
| 44 | …why not | ground gives `a·disp + b = m·r`, `m = −n/d`; `b` and `m_z` share the exact null direction `(0,1,0,0,1)`, so only `c = m_z − b` is observable (§5) |
| 45 | Ground sample spacing | `Δr = stride · r² / (f_y · h)` — sets the sensing horizon (§20) |
| 46 | Pixel → ground point | `t = −d / (n·r)`, `P = t·r`, then `(x,y) = (R₀·P, R₁·P)` — click-to-goal |

---

*Companion documents: [`README.md`](README.md) (how to run it), [`PROTOCOL.md`](PROTOCOL.md) (the
wire protocol), [`ROVER_PLAN.md`](ROVER_PLAN.md) (real-hardware architecture and what is left to build).*
