# Perception server protocol

`perception_server.py` speaks one WebSocket endpoint, `ws://<host>:8790/ws`, to three
kinds of client:

| role     | who                         | sends                         | receives          |
|----------|-----------------------------|-------------------------------|-------------------|
| `sim`    | the Three.js rover (SLAM3D) | binary **frames**, commands   | `config`, `nav`   |
| `rover`  | the Pi (`rover_agent.py`)   | binary **frames**, `hello`    | `config`, `nav`   |
| `viewer` | `dashboard/index.html`      | commands                      | `config`, `nav`   |

With `--source <camera|video>` the server captures frames itself and every client is a
viewer. With `--source rover` the server instead waits for the Pi to push frames, exactly
as it does for `sim`. All text messages are JSON objects with a `type` field.

## Capability flags

`--source` no longer drives behaviour directly. `PerceptionServer.__init__` sets four
booleans once from `source_kind`, and every downstream branch (handshake reset, depth
selection, mount-error reporting, watchdog) reads only these:

| flag             | true for       | meaning                                               |
|------------------|----------------|--------------------------------------------------------|
| `pushes_frames`  | `sim`, `rover` | frames arrive over the socket, not captured locally     |
| `has_true_depth` | `sim`          | renderer depth rides in the frame (`depth` block)       |
| `has_pose`       | `sim`          | a pose source exists (ground truth today, SLAM later)   |
| `is_vehicle`     | `sim`, `rover` | `cmd_vel` reaches something real                        |

`webcam` and `video` sources leave all four `false`. Adding a new source means declaring
what it can do, not re-editing every call site that used to switch on `source_kind`.

## 1. Handshake

Client → server, first message:

```json
{"type": "hello", "role": "sim" | "rover" | "viewer", "client": "free text"}
```

Server → that client:

```json
{"type": "config", "source": "sim" | "rover" | "webcam" | "video", "has_pose": true,
 "depth_mode": "metric" | "metric-indoor" | "relative" | "affine" | "sim",
 "depth_modes": ["metric", "metric-indoor", "relative", "affine", "sim"],
 "v_max": 2.0, "w_max": 1.0, "robot_radius": 0.8,
 "grid": {"x_min": 0.5, "x_max": 12.0, "y_min": -5.0, "y_max": 5.0, "res": 0.1},
 "goal_frame": "world" | "robot"}
```

`"sim"` is appended to `depth_modes` only when `has_true_depth` (the sim source); every
other source's `config` lists just the first four.

* `metric` — Depth Anything V2 metric-outdoor (default). `metric-indoor` — same model,
  indoor-tuned: the outdoor model reads a ~2 m indoor wall as 5-9 m and leaves most pixels
  outside `max_depth`. `relative` — `1/disp` scaled by `nominal_height`; wrong whenever the
  true affine shift isn't ~0. `affine` — solves the disparity-to-depth shift from ground
  planarity instead of assuming it's zero. `sim` — renderer ground truth.
* `affine` is **not** the recommended default: the ground-planarity system has an exact
  null direction between the affine offset `b` and the plane's own `m_z` term (adding the
  same delta to both changes nothing), so it needs an external
  orientation constraint (an IMU gravity vector, or non-coplanar SLAM points) that the
  server does not have yet — without one it returns `(None, None, info)` and the plane is
  lost. Use `metric` outdoors, `metric-indoor` indoors.

A new `hello` with role `sim` or `rover` resets the global map, the goal and the
ground-plane state (the page was reloaded, or the rover reconnected).

## 2. Frames (sim/rover → server, binary)

```
u32 little-endian header length | header JSON (UTF-8) | JPEG bytes | [u16 LE depth]
```

Header:

```json
{"type": "frame", "seq": 1234, "t": 1725500000123.4,
 "w": 640, "h": 360, "fx": 311.8, "fy": 311.8, "cx": 320, "cy": 180,
 "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
 "cam_height": 1.0, "cam_pitch": 0.2618,
 "pose": {"x": 12.3, "y": -4.5, "theta": 1.57},
 "mode": "auto" | "manual",
 "jpeg_len": 43210,
 "depth": {"w": 320, "h": 180, "unit": "mm"} | null}
```

* `fx fy cx cy` are exact pinhole intrinsics of the POV camera **at the JPEG's
  resolution**. Three.js `PerspectiveCamera.fov` is the vertical FOV, so
  `fy = (h/2) / tan(fov/2)`, `fx = fy`.
* `dist`, when present, is `[k1, k2, p1, p2, k3]` OpenCV distortion coefficients (as
  printed by `calibrate.py`). The server rectifies the frame with these **before** any
  geometry is computed — `backproject_optical` is a pure pinhole model, and uncorrected
  barrel distortion bows the ground plane upward at the image edges and invents LETHAL
  cells along both sides of the path. An absent or all-zero `dist` is a no-op, which is
  why the simulator (no `dist` field at all) is unaffected.
* `cam_height` / `cam_pitch` are the mount values, used only to report the estimator's
  error against them (`plane.mount_err`). The server never trusts them for geometry. The
  rover sends `cam_height` (its tape-measured mount height, default 0.17 m) but no
  `cam_pitch` — it has no orientation source. That same tape measurement, passed to the
  server as `--nominal-height`, is what gives the whole map its metric scale whenever the
  depth model itself carries none (`relative` / `affine`).
* `pose` and `mode` are sim-only: the sim is the only source with a pose feed and a
  manual/auto toggle. The rover sends neither key.
* `depth`, when present, is `w*h` unsigned 16-bit millimetres, row-major from the top
  row, `0` = no measurement (sky / beyond range). It is used only in `--depth sim` mode.
  The rover sends no depth block at all — `"depth": null`, always; only the sim sends one.
* The rover may rotate a frame upright before sending it (`--rotation 90/180/270`, to
  correct a sideways camera mount without physically remounting it). When it does, `w`,
  `h`, `fx`, `fy`, `cx`, `cy` describe the **rotated** image — a 90°/270° turn also swaps
  `fx`/`fy` and moves the principal point — so the server needs no knowledge of how the
  camera happens to be bolted on.
* `seq` regressing (page reload, or the rover reconnecting) resets the server's map and
  goal.

The server processes the **latest** frame only; a frame arriving while another is being
processed replaces it. Send at most one frame per received `nav` (or at ≤ 8 Hz).

## 3. Commands (any client → server)

```json
{"type": "set_goal", "x": 20.0, "y": -3.0}     // world frame if has_pose, else robot frame
{"type": "clear_goal"}
{"type": "set_mode", "auto": true}             // relayed to the sim inside `nav`
{"type": "reset"}                              // global map + goal + plane state
{"type": "event", "kind": "contact", "id": "ditch", "hazard": "ditch"}  // sim ground truth: dump the flight recorder
{"type": "set_depth", "mode": "metric" | "metric-indoor" | "relative" | "affine" | "sim"}
{"type": "set_param", "name": "obstacle_h", "value": 0.3}   // tunables, see server --help
```

## 4. Result (server → all clients, text, one per processed frame)

```json
{"type": "nav", "seq": 1234, "t": 1725500000456.7, "source": "sim", "has_pose": true,
 "status": "NO_GOAL" | "PLANNING" | "TURNING" | "DRIVING" | "BLOCKED" | "LOST" | "ARRIVED" | "STOPPED",
 "mode": "auto" | "manual",
 "cmd": {"v": 1.2, "omega": -0.15},
 "twist": {"linear": {"x": 1.2, "y": 0, "z": 0}, "angular": {"x": 0, "y": 0, "z": -0.15}},
 "goal": {"x": 20.0, "y": -3.0} | null,
 "pose": {"x": 12.3, "y": -4.5, "theta": 1.57} | null,
 "dist_to_goal": 8.4,
 "plane": {"height": 1.002, "pitch_deg": 14.9, "roll_deg": 0.1, "confidence": 0.93,
           "ok": true, "source": "fit", "mount_err": {"height": 0.002, "pitch_deg": -0.1}},
 "local": {"path_m": [[0.5, 0.0], [0.6, 0.0]], "reached": true,
           "grid": {"x_min": 0.5, "x_max": 12.0, "y_min": -5.0, "y_max": 5.0, "res": 0.1}},
 "global": {"path_world": [[12.3, -4.5], ...],
            "meta": {"origin_x": -30.0, "origin_y": -30.0, "res": 0.25, "w": 240, "h": 240, "scale": 2}} | null,
 "images": {"costmap": "data:image/png;base64,...", "global": "data:image/png;base64,...",
            "camera": "data:image/jpeg;base64,...", "depth": "data:image/jpeg;base64,..."},
 "depth_mode": "sim", "fps": 6.1,
 "profile": {"decode": 3.1, "depth": 0.4, "sem": 21.0, "core": 24.0, "nav": 9.0, "render": 11.0, "total": 70.0},
 "warnings": [], "note": ""}
```

* `cmd.v` is m/s forward, `cmd.omega` rad/s with **positive = turn left**. `STOPPED` is
  emitted by the watchdog when no frame has arrived for 1 s.
* `images.global` is a north-up crop centred on the robot; `global.meta` lets a click be
  inverted: `world_x = origin_x + (px / scale) * res`,
  `world_y = origin_y + ((img_h − py) / scale) * res`.
* `images.costmap` is the robot-centric local grid (forward = up, left = left).

## 5. ROS-shaped snapshots (HTTP)

`GET /ros/occupancy_grid` (local grid, frame `base_link`), `GET /ros/global_grid`
(frame `map`), `GET /ros/odometry`, `GET /ros/path`, `GET /ros/cmd_vel` return the
matching `nav_msgs` / `geometry_msgs` dictionaries from `ros_msgs.py`. These are the
messages a rosbridge publisher would send to a real Nav2.
