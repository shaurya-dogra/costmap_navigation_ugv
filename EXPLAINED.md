# How the whole thing works — a plain-language walkthrough

This document explains the complete solution: how one camera turns into a map of
"where can the robot drive", how that map becomes a path, and how the path becomes
wheel commands. Every stage lists the maths it uses. If you only read the bullets in
bold you will still get the whole picture.

---

## 0. The one-minute version

- **The camera takes a picture.**
- **Two neural networks look at the picture.** One guesses *how far away* every pixel
  is (depth). The other guesses *what* every pixel is (road, grass, water, tree, …).
- **Geometry turns pixels into 3D points** in metres, then works out *where the ground
  is* from those points. From the ground we learn the camera's height, tilt and roll,
  fresh every frame, so nothing has to be measured with a tape.
- **The 3D points are dropped into a top-down grid** of 10 cm squares in front of the
  robot. Each square gets one number, its **cost**: 0 means "drive freely", 254 means
  "never", 255 means "no idea".
- **A planner finds the cheapest route** across the grid to the destination you clicked,
  in two layers exactly like ROS Nav2: a rough global route on a map that remembers
  everything seen so far, and a precise local route on the fresh grid.
- **A controller turns the first metres of the route into a speed and a turn rate**,
  `(v, ω)`, which is a ROS `Twist` message. The rover (or a real base) drives on it.
- **Repeat 5–10 times a second.**

---

## 1. Inputs

- **Camera image** `I`, width `W`, height `H` (sim: 640×360, webcam: 640×360 after resize).
- **Camera intrinsics** (how the lens maps 3D to pixels): focal lengths `fx, fy` and image
  centre `cx, cy`.
  - Sim: exact, from the Three.js camera:
    `fy = (H/2) / tan(vfov/2)`, `fx = fy`, `cx = W/2`, `cy = H/2`.
  - Webcam: from the horizontal field of view:
    `fx = (W/2) / tan(hfov/2)` (78° default; `calibrate.py` gives exact numbers).
- **In the simulation only**, the renderer also sends its *true* depth image. This stands
  in for the stereo camera a real robot would carry. The neural depth can be switched on
  instead to show the fully camera-only pipeline.

---

## 2. Depth — how far is every pixel?

- **Model**: Depth Anything V2, *metric outdoor small*. Input the RGB picture, output a
  depth `Z(u,v)` in **metres** for every pixel `(u,v)`.
- "Metric" matters: because the answer is in real metres, the camera's height above the
  ground becomes something we can *measure* rather than assume.
- Fallback mode (`--depth relative`): the model only gives *relative* depth (bigger =
  farther, unknown scale). Then the ground plane is fitted in those units and the whole
  cloud is scaled so the camera height equals a nominal value:
  `scale = h_nominal / h_measured`.

---

## 3. Semantics — what is every pixel?

- **Model**: YOLO26 trained on ADE20K, 150 classes. Output: a class label per pixel.
- **A lookup table turns labels into costs** (keyword matching on the class name):

  | class examples | cost | meaning |
  |---|---|---|
  | road, sidewalk, path, runway | 0 | free |
  | floor, field, land, carpet | 20 | nearly free |
  | grass, dirt track | 40 | bumpy |
  | earth, sand, hill | 80 | slow, high effort |
  | anything unrecognised | 100 | uncertain: not free, not forbidden |
  | wall, tree, rock, fence, pole, person, car, … | 250 | "tall lethal" (see §6) |
  | water, river, lake, pool | 254 | lethal, no exceptions |
  | sky, ceiling | −1 | ignored, nothing to drive on |

---

## 4. From pixels to 3D points (back-projection)

- The pinhole camera model, run backwards. For pixel `(u,v)` with depth `Z`:

  ```
  Xc = (u − cx) · Z / fx        (right)
  Yc = (v − cy) · Z / fy        (down)
  Zc = Z                        (forward)
  ```
- This gives one 3D point per pixel in the **optical frame** (camera-centred: X right,
  Y down, Z forward). Every 2nd pixel is used; the grid does not need more.

---

## 5. Finding the ground — the self-calibrating step

Why this exists: the first prototype asked the user to type in the camera height and
tilt. On a laptop on the floor, a phone in a hand or a rover on suspension, those numbers
are neither known nor constant, and roll is never zero. Every height threshold was then
measured against the wrong reference and the map was wrong. Now the ground is **measured
every frame**.

- **Pick candidate ground points**: pixels labelled ground-like (cost ≤ 80), in the lower
  65 % of the image, nearer than 5 m first (widening to 10 m only if too few).
- **RANSAC plane fit** (robust fitting that ignores outliers):
  - Repeat ~150 times: pick 3 random candidate points, compute the plane through them.
  - A plane is `n · p + d = 0`, with `n` the unit normal (points "up") and `d` the offset.
  - Count how many candidates lie within a small band of the plane:
    `|n · p + d| ≤ gate`, where `gate = min(0.05 m + 0.01·Z, 0.10 m)`.
    The band is deliberately **fixed and small** (below the ditch threshold): a surface
    deep enough to be a ditch must never be counted as ground.
  - Keep the plane with the most (near-weighted) supporters, then refine it with a
    least-squares fit on those supporters (twice).
- **Reject impossible planes**: tilt > 60° or roll > 45° is a wall, not ground; a camera
  height outside 3 cm … 5 m is nonsense. Hold the last good plane for a few frames if
  the fit fails; after that the map goes fully UNKNOWN (never free).
- **Read the camera pose off the plane**:

  ```
  height h = d                              (distance from lens to ground)
  pitch   = atan2(−n_z, √(n_x² + n_y²))     (nose-down positive)
  roll    = atan2(n_x, −n_y)                (right side lower positive)
  ```
- **Smooth over time**: blend with the previous frame (50 %), but a big jump is accepted
  only when the new fit has strong support (≥ 60 % inliers), otherwise hold.
- Accuracy on synthetic scenes: height within 1 cm, angles within 0.5°. In the sim the
  known mount (1.0 m, 15°) is shown next to the estimate as a live check.

---

## 6. From optical frame to a ground-aligned robot frame

- Build a rotation `R` whose rows are: forward `f` (optical axis projected onto the
  ground), left `l = n × f`, up `n`.
- Every point becomes
  ```
  [X, Y, Z]ᵀ = R · [Xc, Yc, Zc]ᵀ + [0, 0, h]ᵀ
  ```
  so **`Z` is literally the height above the ground** (0 on the ground, negative in a
  hole, positive on a rock), `X` is metres ahead, `Y` metres to the left.
- Roll is corrected here as a by-product; the old pipeline could not do this at all.

---

## 7. The costmap — one number per 10 cm square

- **Grid**: robot-centred, `X` from 0.5 m to 12 m ahead (webcam: 0.3–8 m), `Y` from
  −5 m to +5 m (webcam ±4 m), cell size 0.1 m. Cell index:
  `ix = ⌊(X − x_min)/res⌋`, `iy = ⌊(Y − y_min)/res⌋`.
- Every 3D point lands in one cell. Per cell we count points and look at their heights
  and labels. Then the rules, all conservative, all covered by tests:

### 7.1 Evidence first
- **Fewer than 3 points → UNKNOWN (255).** One or two stray depth samples are not a
  measurement; treating them as evidence sprayed the old map with speckle.

### 7.2 Semantic vote (what it looks like)
- Let `n` be the cell's points and `n_lethal` those with a lethal label.
- **Lethal if** `n_lethal ≥ max(0.25 · n, 2)` — a quarter of the cell says "danger".
- **Otherwise** the cost is the *mean of the non-lethal points only*. Averaging all
  points was wrong both ways: it diluted a wall to "awkward terrain", and it dragged
  clean floor above 50 so nothing was ever free.
- **Tall-label check**: labels that imply height (wall, tree, car, person …) *must* be
  confirmed by the geometry. If every point in the cell lies within a few centimetres of
  the ground, the label is a mistake (a grey floor read as "wall"). Such a cell gets cost
  **150**: expensive, never free, never lethal. **Water keeps 254**: it is flat, so
  geometry cannot check it, and it is genuinely deadly.

### 7.3 Geometry (what shape it is)
- **Positive obstacle**: enough points with `Z > 0.25 m` → LETHAL (rocks, logs, fences).
- **Negative obstacle**: enough points with `Z < −0.20 m` (within 9 m) → LETHAL
  (kerb drops, trench walls).
- "Enough" = at least 20 % of the cell's points and never fewer than 2 (capped at 6).
- Points above 2 m are ignored: overhead branches are clearance, not obstacles.

### 7.4 The hole rule (the trench that looks like nothing)
- A trench's floor is hidden behind its own near edge, so the camera simply sees
  *nothing* there: no points, UNKNOWN. UNKNOWN was "expensive but passable", and a
  1.4 m stripe of it was cheaper to cross than a 4 m detour — the rover drove in.
- Physics: a downward-looking camera sees **continuous** ground. A run of cells with
  **zero** points, with measured ground **before it and beyond it** along the view, can
  only be a place where the surface dipped out of sight.
- **Rule**: such a run → LETHAL, provided
  - the nearest measured cell behind it is not itself lethal (then it is just the shadow
    of a rock), and
  - the run is longer than the natural sampling gap at that range, which grows with
    distance: `gap ≈ stride · r² / (fy · h)`. Beyond where that gap exceeds ~2 cells,
    no verdict is given (the camera honestly cannot tell).

### 7.5 Fusion — the safety argument
- ```
  cost(cell) = max( semantic , positive_obstacle , negative_obstacle , hole )
  ```
  **Never averaged, never weighted, never learned.** If any source says dangerous, the
  cell is dangerous. Cells with too little evidence stay UNKNOWN.

### 7.6 Inflation — the robot is not a point
- Distance transform from every lethal cell. Cells within one robot radius `r` become
  **253** (a skirt you must not enter); from `r` to `2r` the cost decays as
  `200 · e^(−2(dist − r)/r)`, so paths keep a comfortable margin.
- UNKNOWN cells inside the inner skirt also become 253 (an unmeasured cell right next to
  a trench edge is not a cheap place to drive); further out they stay UNKNOWN.

### 7.7 Reading the colours
- green 0 → yellow ~100 → orange ~200 → magenta 254 lethal; dark grey 255 unknown.

---

## 8. From costmap to Nav2-style planning

ROS Nav2 splits navigation into two layers. This project keeps exactly that split (in
Python) and speaks Nav2's message shapes, so a real Nav2 can be attached later over
rosbridge without changing the perception.

### 8.1 Local costmap = the grid above
- Fresh every frame, robot-centred, 0.1 m cells. In Nav2 terms this is the
  `local_costmap` with the robot's sensor layer + inflation layer already applied.
- Published as `nav_msgs/OccupancyGrid` (`GET /ros/occupancy_grid`, frame `base_link`):
  each byte maps to Nav2's scale, `−1` unknown, `0…100`, lethal → 100, skirt → 99.

### 8.2 Global costmap = memory
- World-fixed grid, 0.25 m cells, 160 m across. Needs the robot's **pose** `(x, y, θ)`
  — in the sim it is ground truth (the placeholder for the team's visual SLAM); the
  webcam has none, so webcam mode is local-only.
- **Fusion**: every *measured* local cell is moved into the world frame,
  ```
  wx = x + X·cos θ − Y·sin θ
  wy = y + X·sin θ + Y·cos θ
  ```
  and written with `max` (danger is never forgotten). UNKNOWN local cells never
  overwrite, so unexplored ground stays UNKNOWN and explored ground stays known.
- Published as `GET /ros/global_grid`, frame `map`.

### 8.3 Global planner (Nav2's NavFn / Smac role)
- **A\*** over a coarse copy of the global map: 0.5 m cells (2×2 max-pooled), boxed
  around start and goal with a 15 m margin. Full-map A\* took 0.5 s in pure Python;
  the boxed copy takes ~10–80 ms. Replans at most once a second, or immediately if a
  new lethal cell lands on the current path.
- **Step cost**:
  ```
  cost(step) = length · (1 + 6 · cell_cost / 255)
  ```
  so straight-line distance stays a valid heuristic and higher-cost ground bends the
  route without forbidding it. Lethal (254) is impassable. UNKNOWN costs 40 on the
  global map (unexplored is not hazardous at map scale).
- The robot's own cell is always made walkable (fusion smear can paint it lethal).
- Output: a world path, `nav_msgs/Path` (`GET /ros/path`).

### 8.4 Carrot — handing the global path to the local layer
- Take the first global-path point about 10 m ahead (or the goal itself if it is inside
  the local window), expressed in the robot frame. That is the local planner's goal.
- If the carrot is behind or far to the side (bearing > 70° or closer than 1 m ahead),
  the robot **turns in place** first (`v = 0`, `ω = clip(1.0 · bearing, ±ω_max)`) until
  the bearing is under 15°. A forward-only grid cannot plan backwards.

### 8.5 Local planner + controller (Nav2's controller role)
- The local planning grid is the fresh frame grid **backfilled from the global memory**
  wherever it is UNKNOWN: the camera only sees a wedge, and without memory the planner
  routed through never-seen cells beside the robot and lost a trench edge the moment the
  rover turned.
- **A\*** again, 8-connected on the 0.1 m grid, same step-cost formula, UNKNOWN costs
  200 here (expensive, but the far half of a camera frame is always partly unmeasured,
  so it must stay passable). An empty path (lethal dead ahead) means STOP.
- **Pure pursuit** turns the path into a command. Take the path point `L = 2.5 m`
  ahead (lookahead), at `(x, y)` in the robot frame, `d = √(x² + y²)`:
  ```
  κ  = 2y / d²                       curvature of the arc through that point
  v  = v_max / (1 + 0.6 · |κ|)        slow down in tight turns
  v ·= min(1, reach / L)             slow down when the plan ends soon
  ω  = clip(v · κ, −ω_max, ω_max)     positive = turn left
  ```
  plus: stop if the plan reaches less than 1.2 m, slow within 4 m of the goal, ramp `v`
  at ≤ 1.5 m/s², low-pass `ω` between frames.
- Output: `geometry_msgs/Twist` with `linear.x = v`, `angular.z = ω`
  (`GET /ros/cmd_vel`).

### 8.6 The state machine (Nav2's behaviour tree role)
- `NO_GOAL` → `PLANNING` → `TURNING` / `DRIVING` → `ARRIVED` (within 1.2 m).
- `BLOCKED`: no safe first step → stop 3 frames, then spin slowly toward the carrot for
  up to 20 frames (Nav2's *Spin* recovery), then try again.
- `STOPPED`: watchdog — no camera frame for 1 s → `v = ω = 0`.

---

## 9. Closing the loop in the simulation

- The rover carries a camera 1.0 m up, 15° down. Every ≤ 6 Hz it renders what the camera
  sees into an offscreen buffer (RGB, corrected from linear to sRGB) and a depth pass
  (`MeshDepthMaterial`, decoded to millimetres), and sends both with its pose over a
  WebSocket.
- The server returns `(v, ω)`. The rover applies **each command for one control period
  only**, then holds its heading until the next one — integrating a stale turn rate
  across the whole gap between frames was the zig-zag.
- Ground truth (rocks, trenches, ponds …) is used only to *score* the run; the
  perception never sees it.

---

## 10. Webcam mode

- Same pipeline, camera source, 640×360, nano segmenter, no pose → local layer only.
- A carrot goal 7 m ahead is always set, so the dashboard always shows the Nav2 plan and
  the `Twist` that would be sent to a base. Clicking the costmap moves the goal.
- The plane estimate (height ≈ 0.2 m with the laptop on the floor, pitch, roll) is shown
  live: tilt the lid and watch it track while the map stays put.

---

## 11. Where each piece lives

| stage | file | function |
|---|---|---|
| intrinsics, back-projection | `perception_core.py` | `intrinsics_from_hfov`, `backproject_optical` |
| ground plane | `perception_core.py` | `GroundPlaneEstimator`, `normal_from_angles`, `rotation_from_normal` |
| costmap rules | `perception_core.py` | `build_costmap`, `inflate` |
| models | `perception_core.py` | `DepthModel`, `SemanticModel`, `build_cost_lut` |
| global map, planners, controller | `navstack.py` | `GlobalCostmap.fuse`, `plan_global`, `carrot`, `Navigator.step` |
| A\*, pure pursuit | `costmap_prototype.py` | `astar`, `drive_command` |
| Nav2 messages | `ros_msgs.py` | `occupancy_grid`, `odometry`, `path_msg`, `twist` |
| server, dashboard | `perception_server.py`, `dashboard/` | — |
| rover sim | `sim3d/src/nav`, `sim3d/src/components` | capture, link, Vehicle, Environment |

---

## 12. Numbers to remember

| constant | value | why |
|---|---|---|
| cell size | 0.10 m | 10 cm resolution, 12 000 cells, fast |
| positive obstacle | Z > 0.25 m | above wheel-climbable height |
| negative obstacle | Z < −0.20 m | deeper than a kerb the base can take |
| plane inlier band | ≤ 0.10 m | below the ditch threshold, so a ditch is never "ground" |
| min points per cell | 3 | speckle is not evidence |
| lethal vote | 25 % of points | a wall in a 10 cm cell is a wall |
| robot radius | 1.0 m sim / 0.35 m webcam | inflation |
| UNKNOWN cost | 200 local / 40 global | passable but expensive; unexplored ≠ hazardous |
| lookahead | 2.5 m | smooth steering at 2 m/s |
| v_max, ω_max | 2.0 m/s, 0.8 rad/s | rover limits |
