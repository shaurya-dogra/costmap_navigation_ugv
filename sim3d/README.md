# SLAM3D — Autonomous Rover Simulation (React Three Fiber / Rapier)

A 3D outdoor course with a physics-driven rover that **drives itself** to a destination
you click. The rover's camera is streamed (RGB + true depth) to the perception server in
[`costmap_navigation_ugv`](https://github.com/shaurya-dogra/costmap_navigation_ugv)
(`perception_server.py`), which returns a traversability costmap, a plan and `(v, ω)`
drive commands. Manual WASD driving, three camera modes and the telemetry HUD from the
original simulator are kept.

## Run

```bash
npm install
# assets are not in git: copy rover.glb road.glb tree.glb into public/
npm run dev                      # http://localhost:5173

# in the perception repo, the server the rover talks to:
python perception_server.py --source sim --depth sim --auto --port 8790
# or, from that repo, both at once:  ./run_sim.sh
```

Open `http://localhost:5173/?auto=1`. URL params: `ws=` (server, default
`ws://<host>:8790/ws`), `seed=` (course randomisation), `auto=1` (start autonomous).

## Controls

| Key / click | Action |
|---|---|
| **W A S D** / arrows | drive (takes over from AUTO) |
| **Space** | brake |
| **T** | toggle AUTO / MANUAL |
| **C** | camera: chase → top-down → driver |
| click ground | set destination |
| click course map (bottom-left) or global map (right) | set destination |
| X, Y box in the HUD | set destination in nav-world metres |

## What the HUD shows

Status (`NO_GOAL / PLANNING / TURNING / DRIVING / BLOCKED / ARRIVED`), the command being
applied, distance to goal, perception frame rate, the **ground plane measured by the
server every frame** (height / pitch / roll vs the known 1.0 m / 15° mount), a
ground-truth **collision counter**, the robot's camera feed, the local costmap and the
accumulated global costmap with the planned path.

## The course (`src/nav/world.js`)

Road with boulders, rubble clusters, trees, fallen logs, bushes, signposts, two fences,
two **trenches** (real holes: the ground has gaps and a sunken floor), two ponds and a
puddle (flat — only semantics can see them), mud and sand patches, a grassy mound.
`groundTruthHits()` scores contacts with all of them; the perception never sees this list.

## Structure

```
src/nav/config.js        capture, mount, rover, URL params
src/nav/frames.js        three.js ↔ nav world axes (the one place)
src/nav/link.js          WebSocket link, watchdog, useNav()
src/nav/capture.js       offscreen RGB (sRGB-corrected) + depth pass → binary frame
src/nav/world.js         course layout + ground truth
src/components/Environment.jsx   ground with trench holes, textures, obstacles
src/components/Vehicle.jsx       physics, AUTO/MANUAL, pose, collision scoring
src/components/RobotCamera.jsx   the rover's camera + capture loop
src/components/GoalMarker.jsx    goal flag, global + local path lines
src/components/Hud.jsx           overlay panels and controls
src/components/MiniMap.jsx       clickable, expandable course map
```

Frames: nav `x = −three.z`, `y = −three.x`, `θ = heading` (left = positive). The
camera's pose sent to the server is the ground point under the lens.
Wire protocol: `PROTOCOL.md` in the perception repo.
