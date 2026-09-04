# Prompt for Antigravity

Copy everything below the line into Antigravity.

---

You are working in `~/Developer/coding/anitgravity/sih_prototype_costmap` on macOS
(Apple M4, 16 GB). Read every file in the folder before writing any code.

## Context

This is a monocular traversability-costmap prototype for Smart India Hackathon
problem statement 26126, "Vision Based Autonomous Navigation for Unmanned Ground
Vehicle for Outdoor environment". A phone camera feeds a perception stack that
produces a top-down drivability costmap in metres. It is a stand-in for a stereo
rig (Luxonis OAK-FFC-3P) that arrives later.

`costmap_prototype.py` already contains the complete, working, verified core.
`test_geometry.py` is its regression suite and currently prints ALL CHECKS PASSED
with the costmap stage at 19.7 ms per 1280x720 frame. Do not rewrite either file
from scratch. Extend them.

## The pipeline

    phone frame
      -> Depth Anything V2 (Metric Outdoor Small, HuggingFace)  -> metric depth
      -> SegFormer-B0 ADE20K (nvidia/segformer-b0-finetuned-ade-512-512) -> class per pixel
      -> Ultralytics YOLO nano -> object boxes
      -> back-project every pixel to 3D using intrinsics + camera height + pitch
      -> bin into a 10 cm top-down grid, 0.5-10 m ahead, +/-4 m lateral
      -> cost = max(semantic_cost, height_cost, object_cost)
      -> unknown cells = expensive, then inflate by robot radius
      -> display

## Your tasks, in this order

1. Create the venv and install from `requirements.txt`. Torch must be the plain
   pip wheel so it uses Metal (MPS). Export `PYTORCH_ENABLE_MPS_FALLBACK=1`.
   Stay in float32 on Apple silicon; FP16 on MPS produces silent NaNs.
2. Run `python test_geometry.py`. It must print ALL CHECKS PASSED before you
   change anything. If it does not, stop and report.
3. Get the live pipeline running end to end with `--source 0` (iPhone over
   Continuity Camera). Fix whatever breaks in model loading, dtype or device
   placement. Report the achieved fps.
4. Add `--record out.mp4` to save the raw camera stream, and make `--source`
   accept that file, so development stops depending on a live camera.
5. Add a runtime tuning panel: OpenCV trackbars for `obstacle_h`, `ditch_h`,
   `rough_h` and `cam_pitch`, so thresholds can be set against real terrain
   without editing code.
6. Add a simple A* over the costmap to a goal cell 8 m ahead, drawn on the
   costmap window. Treat UNKNOWN as high cost, not as blocked and not as free.
7. Add `--profile` printing per-stage milliseconds (depth, semantics, detection,
   costmap, render) so the bottleneck is measurable rather than guessed.
8. Extend `test_geometry.py` with a case for every behaviour you add. The suite
   must stay green and must keep needing no models, no camera and no GPU.

## Hard constraints, do not violate

- `cost = max(semantic, height, object)`. Never average, never weight, never
  learn the combination. If either evidence source says dangerous, the cell is
  dangerous. This is the safety argument of the whole project.
- Cells with no measurement are UNKNOWN and expensive. Never free.
- The negative-obstacle rule (`z_min < ditch_h` is lethal) is what detects
  ditches. A ditch and a shadow are identical in pixels, so geometry is the only
  thing that catches it. Never remove it or fold it into the semantic channel.
- `Perception.depth` is the single boundary where stereo will replace monocular.
  Do not spread depth assumptions into other functions.
- The grid is `(forward, lateral)`, axis 0 = X ahead, axis 1 = Y left. This was
  a bug once. The orientation tests exist to catch it. Do not change the axis
  order without updating both the renderer and the tests.
- Do not add temporal smoothing across frames. That needs a pose source and
  monocular SLAM has no scale. Single-frame grids only, for now.
- Keep it a plain Python project. No ROS, no Docker, no packaging.

## Configuration the user must set

In `Cfg`: `fx = fy = 940` and `cx, cy = 640, 360` for a 1280x720 phone feed is a
usable starting guess. `cam_height` and `cam_pitch` are physically measured, so
prompt the user for them rather than inventing values.

## Definition of done

`python costmap_prototype.py --source 0` opens three windows, runs at a reported
frame rate, shows a costmap where a kerb drop or a step down turns lethal from
geometry rather than from appearance, and `python test_geometry.py` still prints
ALL CHECKS PASSED.
