# Perception server protocol

`perception_server.py` speaks one WebSocket endpoint, `ws://<host>:8790/ws`, to two
kinds of client:

| role     | who                         | sends                         | receives          |
|----------|-----------------------------|-------------------------------|-------------------|
| `sim`    | the Three.js rover (SLAM3D) | binary **frames**, commands   | `config`, `nav`   |
| `viewer` | `dashboard/index.html`      | commands                      | `config`, `nav`   |

With `--source <camera|video>` the server captures frames itself and every client is a
viewer. All text messages are JSON objects with a `type` field.

## 1. Handshake

Client → server, first message:

```json
{"type": "hello", "role": "sim" | "viewer", "client": "free text"}
```

Server → that client:

```json
{"type": "config", "source": "sim" | "webcam" | "video", "has_pose": true,
 "depth_mode": "metric" | "relative" | "sim", "depth_modes": ["metric", "relative", "sim"],
 "v_max": 2.0, "w_max": 1.0, "robot_radius": 0.8,
 "grid": {"x_min": 0.5, "x_max": 12.0, "y_min": -5.0, "y_max": 5.0, "res": 0.1},
 "goal_frame": "world" | "robot"}
```

A new `hello` with role `sim` resets the global map, the goal and the ground-plane state
(the page was reloaded).

## 2. Frames (sim → server, binary)

```
u32 little-endian header length | header JSON (UTF-8) | JPEG bytes | [u16 LE depth]
```

Header:

```json
{"type": "frame", "seq": 1234, "t": 1725500000123.4,
 "w": 640, "h": 360, "fx": 311.8, "fy": 311.8, "cx": 320, "cy": 180,
 "cam_height": 1.0, "cam_pitch": 0.2618,
 "pose": {"x": 12.3, "y": -4.5, "theta": 1.57},
 "mode": "auto" | "manual",
 "jpeg_len": 43210,
 "depth": {"w": 320, "h": 180, "unit": "mm"} | null}
```

* `fx fy cx cy` are exact pinhole intrinsics of the POV camera **at the JPEG's
  resolution**. Three.js `PerspectiveCamera.fov` is the vertical FOV, so
  `fy = (h/2) / tan(fov/2)`, `fx = fy`.
* `cam_height` / `cam_pitch` are the mount values, used only to report the estimator's
  error against them (`plane.mount_err`). The server never trusts them.
* `pose` is the robot in the **nav world frame** (X = −three.z, Y = −three.x,
  θ = heading, CCW positive). This is the VSLAM placeholder.
* `depth`, when present, is `w*h` unsigned 16-bit millimetres, row-major from the top
  row, `0` = no measurement (sky / beyond range). It is used only in `--depth sim` mode.
* `seq` regressing (page reload) resets the server's map and goal.

The server processes the **latest** frame only; a frame arriving while another is being
processed replaces it. Send at most one frame per received `nav` (or at ≤ 8 Hz).

## 3. Commands (any client → server)

```json
{"type": "set_goal", "x": 20.0, "y": -3.0}     // world frame if has_pose, else robot frame
{"type": "clear_goal"}
{"type": "set_mode", "auto": true}             // relayed to the sim inside `nav`
{"type": "reset"}                              // global map + goal + plane state
{"type": "set_depth", "mode": "metric" | "relative" | "sim"}
{"type": "set_param", "name": "obstacle_h", "value": 0.3}   // tunables, see server --help
```

## 4. Result (server → all clients, text, one per processed frame)

```json
{"type": "nav", "seq": 1234, "t": 1725500000456.7, "source": "sim", "has_pose": true,
 "status": "NO_GOAL" | "PLANNING" | "TURNING" | "DRIVING" | "BLOCKED" | "ARRIVED" | "STOPPED",
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
