"""
Synthetic-scene verification of perception_core: the self-calibrating geometry.
Needs only numpy + opencv. No models, no camera, no GPU.

    python test_perception_core.py

The camera in every scene has an UNKNOWN height, pitch and roll as far as the
module is concerned; the tests check that it measures them, and that the costmap
built on the measured plane is right.
"""
import math, sys, time
import numpy as np

import perception_core as pc

FAILS = []


def check(name, got, ok):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
    print(f"  {tag}  {name:<52} {got}")


from synth_scene import synth


def mk_cfg(w=640, h=360, hfov=78.0, **kw):
    fx, fy, cx, cy = pc.intrinsics_from_hfov(w, h, hfov)
    return pc.CoreCfg(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, **kw)


def run(cfg, depth, sem, metric=True, core=None):
    core = core or pc.PerceptionCore(cfg, seed=1)
    return core.process(depth, sem, depth_is_metric=metric), core


def lethal_centroid(cfg, grid):
    i = np.where(grid == cfg.LETHAL)
    if len(i[0]) == 0:
        return None, None, 0
    return (cfg.x_min + (i[0].mean() + 0.5) * cfg.res, cfg.y_min + (i[1].mean() + 0.5) * cfg.res, len(i[0]))


# --------------------------------------------------------------------- tests --
print("\n1. INTRINSICS AND ANGLE CONVENTIONS")
fx, fy, cx, cy = pc.intrinsics_from_hfov(1280, 720, 90.0)
check("90 deg hfov at 1280 px -> fx = 640", f"fx={fx:.1f} cx={cx} cy={cy}", abs(fx - 640) < 1e-6 and cx == 640 and cy == 360)
fy2 = pc.intrinsics_from_vfov(640, 360, 60.0)[1]
check("60 deg vfov at 360 px -> fy = 311.8", f"fy={fy2:.1f}", abs(fy2 - 311.77) < 0.1)
for p, r in ((0.0, 0.0), (0.2, 0.0), (-0.1, 0.07), (0.3, -0.2)):
    p2, r2 = pc.angles_from_normal(pc.normal_from_angles(p, r))
    check(f"pitch/roll round trip ({math.degrees(p):+.0f},{math.degrees(r):+.0f}) deg",
          f"{math.degrees(p2):+.2f},{math.degrees(r2):+.2f}", abs(p - p2) < 1e-9 and abs(r - r2) < 1e-9)
R = pc.rotation_from_normal(pc.normal_from_angles(0.0, 0.0))
check("level camera: optical Z -> robot +X, optical -Y -> +Z", f"R@[0,0,1]={np.round(R @ [0,0,1],3)} R@[0,-1,0]={np.round(R @ [0,-1,0],3)}",
      np.allclose(R @ [0, 0, 1], [1, 0, 0]) and np.allclose(R @ [0, -1, 0], [0, 0, 1]) and np.allclose(R @ [-1, 0, 0], [0, 1, 0]))

print("\n2. GROUND PLANE IS MEASURED, NOT ASSUMED")
cfg = mk_cfg()
worst_h = worst_a = 0.0
for h_true, p_true, r_true in ((0.22, 0.0, 0.0), (0.60, 12.0, 0.0), (0.25, -5.0, 4.0), (1.00, 15.0, -3.0), (0.45, 30.0, 8.0)):
    depth, sem = synth(cfg, h_true, math.radians(p_true), math.radians(r_true))
    res, _ = run(cfg, depth, sem)
    pl = res.plane
    eh = abs(pl.height - h_true); ep = abs(math.degrees(pl.pitch) - p_true); er = abs(math.degrees(pl.roll) - r_true)
    worst_h = max(worst_h, eh); worst_a = max(worst_a, ep, er)
    check(f"recovers h={h_true:.2f} pitch={p_true:+.0f} roll={r_true:+.0f}",
          f"h={pl.height:.3f} pitch={math.degrees(pl.pitch):+.2f} roll={math.degrees(pl.roll):+.2f} conf={pl.confidence:.2f}",
          pl.ok and eh < 0.01 and ep < 0.5 and er < 0.5)
    seen = res.grid != cfg.UNKNOWN
    check(f"   ...and the ground reads free (max cost {int(res.grid[seen].max()) if seen.any() else '-'})",
          f"{int((res.grid == cfg.LETHAL).sum())} lethal / {int(seen.sum())} seen",
          seen.any() and (res.grid == cfg.LETHAL).sum() == 0 and res.grid[seen].max() <= 20)
check("worst-case plane error over all rigs", f"{worst_h*100:.2f} cm / {worst_a:.3f} deg", worst_h < 0.01 and worst_a < 0.5)

print("\n3. NEGATIVE OBSTACLE: A STEP DOWN THAT LOOKS EXACTLY LIKE ROAD")
h_true, p_true = 0.60, math.radians(12.0)
depth, sem = synth(cfg, h_true, p_true, 0.0, drop=(3.5, 0.35))
res, _ = run(cfg, depth, sem)
li = np.where(res.grid == cfg.LETHAL)
check("step-down labelled road is lethal by geometry", f"{len(li[0])} lethal cells", len(li[0]) > 0)
check("plane stays anchored on the near ground (not the lower surface)",
      f"h={res.plane.height:.3f} (true 0.600) pitch={math.degrees(res.plane.pitch):.2f}",
      abs(res.plane.height - h_true) < 0.02 and abs(res.plane.pitch - p_true) < math.radians(0.5))
shadow = 3.5 * (h_true + 0.35) / h_true
first = cfg.x_min + li[0].min() * cfg.res if len(li[0]) else 99
check("lethal band starts AT THE LIP (hole rule fills the occlusion shadow)", f"X={first:.2f} m (lip 3.50, lower ground visible from {shadow:.2f} m)", abs(first - 3.5) < 0.4)
band = res.grid[int((3.5 - cfg.x_min) / cfg.res):, :]
check("nothing beyond the lip reads free", f"{int((band < 100).sum())} free cells beyond 3.5 m", (band < 100).sum() == 0)

print("\n3b. NARROW TRENCH: FLOOR HIDDEN BY ITS OWN LIP -> HOLE RULE")
cfg_s = mk_cfg(hfov=60.0, x_max=12.0, y_min=-5.0, y_max=5.0, stride=1, robot_radius=0.0)
depth, sem = synth(cfg_s, 1.0, math.radians(15.0), 0.0, trench=(4.0, 5.4, 0.5))
res, _ = run(cfg_s, depth, sem)
band = res.grid[int((4.1 - cfg_s.x_min) / cfg_s.res):int((5.3 - cfg_s.x_min) / cfg_s.res), int((-1.5 - cfg_s.y_min) / cfg_s.res):int((1.5 - cfg_s.y_min) / cfg_s.res)]
check("trench band is lethal (was UNKNOWN and therefore crossable)", f"{int((band == cfg_s.LETHAL).sum())} lethal, {int((band == cfg_s.UNKNOWN).sum())} unknown, {int((band < 100).sum())} free of {band.size}",
      (band == cfg_s.LETHAL).sum() > 0.7 * band.size and (band < 100).sum() == 0)
before = res.grid[:int((3.6 - cfg_s.x_min) / cfg_s.res), :]; bs = before != cfg_s.UNKNOWN
check("   ...ground before the trench still free", f"{int((before[bs] == cfg_s.LETHAL).sum())} lethal of {int(bs.sum())}", bs.any() and (before[bs] == cfg_s.LETHAL).sum() == 0)
after = res.grid[int((6.0 - cfg_s.x_min) / cfg_s.res):int((9.0 - cfg_s.x_min) / cfg_s.res), int((-1.0 - cfg_s.y_min) / cfg_s.res):int((1.0 - cfg_s.y_min) / cfg_s.res)]; as_ = after != cfg_s.UNKNOWN
check("   ...ground beyond the trench still free", f"{int((after[as_] == cfg_s.LETHAL).sum())} lethal of {int(as_.sum())}", as_.any() and (after[as_] == cfg_s.LETHAL).sum() == 0)
depth, sem = synth(cfg_s, 1.0, math.radians(15.0), 0.0)
res, _ = run(cfg_s, depth, sem)
check("   ...flat ground has no false holes within hole_max_range", f"{int((res.grid[:int((cfg_s.hole_max_range - cfg_s.x_min) / cfg_s.res)] == cfg_s.LETHAL).sum())} lethal", (res.grid[:int((cfg_s.hole_max_range - cfg_s.x_min) / cfg_s.res)] == cfg_s.LETHAL).sum() == 0)

print("\n4. POSITIVE OBSTACLES LAND WHERE THEY ARE (lethal by height, labelled 'rock')")
for name, box, want in (("left", (3.0, 3.6, 0.8, 1.6, 0.0, 0.9), lambda y: y > 0.5),
                        ("right", (3.0, 3.6, -1.6, -0.8, 0.0, 0.9), lambda y: y < -0.5),
                        ("ahead", (3.0, 3.6, -0.4, 0.4, 0.0, 0.9), lambda y: abs(y) < 0.3)):
    depth, sem = synth(cfg, 0.6, math.radians(12), 0.0, boxes=[box])
    res, _ = run(cfg, depth, sem)
    fx_, fy_, n = lethal_centroid(cfg, res.grid)
    check(f"obstacle {name:<5} -> Y sign correct, X at near face 3.0 m",
          f"X={fx_ if fx_ is None else round(fx_,2)} Y={fy_ if fy_ is None else round(fy_,2)} n={n}",
          n > 0 and want(fy_) and abs(fx_ - 3.15) < 0.4)
    check(f"   ...obstacle {name:<5} does not disturb the plane", f"h={res.plane.height:.3f}", abs(res.plane.height - 0.6) < 0.01)

print("\n5. ROLL IS CORRECTED (a laptop never sits level)")
depth, sem = synth(cfg, 0.22, 0.0, math.radians(6.0))
res, _ = run(cfg, depth, sem)
seen = res.grid != cfg.UNKNOWN
check("6 deg roll on flat ground -> no lethal, no ditch", f"{int((res.grid == cfg.LETHAL).sum())} lethal, roll est {math.degrees(res.plane.roll):+.2f}",
      (res.grid == cfg.LETHAL).sum() == 0 and abs(math.degrees(res.plane.roll) - 6.0) < 0.5)
# the same scene through a FIXED-rig assumption (roll = 0) is what the old pipeline did
cfg_lock = mk_cfg(lock_height=0.22, lock_pitch=0.0, lock_roll=0.0)
res_l, _ = run(cfg_lock, depth, sem)
check("(for contrast) ignoring roll DOES break the map", f"{int((res_l.grid == cfg.LETHAL).sum())} false lethal cells", (res_l.grid == cfg.LETHAL).sum() > 0)

print("\n6. FAIL-SAFE BEHAVIOUR")
res, _ = run(cfg, np.zeros((cfg.h, cfg.w), np.float32), np.zeros((cfg.h, cfg.w), np.float32))
check("no depth -> plane lost, everything UNKNOWN, never free", f"ok={res.plane.ok} unknown={bool((res.grid == cfg.UNKNOWN).all())}",
      not res.plane.ok and (res.grid == cfg.UNKNOWN).all() and res.warnings)
depth, sem = synth(cfg, 0.6, math.radians(12), 0.0, water=(2.0, 6.0))
res, _ = run(cfg, depth, sem)
band = res.grid[int((2.3 - cfg.x_min) / cfg.res):int((5.7 - cfg.x_min) / cfg.res), :]
seen_b = band != cfg.UNKNOWN
check("flat water band -> lethal by semantics alone", f"{int((band[seen_b] >= 253).sum())}/{int(seen_b.sum())} cells",
      seen_b.any() and (band[seen_b] >= 253).all())
check("   ...water does not corrupt the ground plane (it is excluded as non-ground)", f"h={res.plane.height:.3f}", abs(res.plane.height - 0.6) < 0.01)

print("\n6b. TALL LABELS NEED HEIGHT, FLAT HAZARDS DO NOT")
depth, sem = synth(cfg, 0.6, math.radians(12), 0.0, boxes=[(3.0, 3.6, -0.4, 0.4, 0.0, 0.9)])
sem_wall = np.where(sem >= 0, 250.0, sem)          # segmenter calls EVERYTHING "wall"
res, _ = run(cfg, depth, sem_wall)
seen = res.grid != cfg.UNKNOWN
flat_rows = res.grid[:int((2.5 - cfg.x_min) / cfg.res), :]
fr_seen = flat_rows != cfg.UNKNOWN
check("'wall' on measured-flat ground -> demoted to high cost, not lethal",
      f"flat rows: {int((flat_rows[fr_seen] == cfg.LETHAL).sum())} lethal, {int((flat_rows[fr_seen] == pc.TALL_DEMOTED).sum())} demoted of {int(fr_seen.sum())}",
      fr_seen.any() and (flat_rows[fr_seen] == cfg.LETHAL).sum() == 0 and (flat_rows[fr_seen] == pc.TALL_DEMOTED).sum() > 0.8 * fr_seen.sum())
fx_, fy_, n = lethal_centroid(cfg, res.grid)
check("   ...while the real box (has height) is still lethal where it is", f"X={fx_ if fx_ is None else round(fx_,2)} n={n}", n > 0 and abs(fx_ - 3.15) < 0.4)
check("   ...and the plane was not fooled (wall is excluded from ground candidates -> fallback region)", f"h={res.plane.height:.3f}", abs(res.plane.height - 0.6) < 0.02)
sem_water = np.where(sem >= 0, 254.0, sem)
res, _ = run(cfg, depth, sem_water)
seen = res.grid != cfg.UNKNOWN
check("'water' on flat ground stays lethal (cannot be checked by height)", f"{int((res.grid[seen] >= 253).sum())}/{int(seen.sum())}", seen.any() and (res.grid[seen] >= 253).all())
lut = pc.build_cost_lut({0: "wall", 1: "water", 2: "person"})
check("LUT: wall/person are TALL lethal (250), water is flat lethal (254)", f"{lut.tolist()}", lut[0] == 250 and lut[2] == 250 and lut[1] == 254)

print("\n7. TEMPORAL HOLD")
core = pc.PerceptionCore(cfg, seed=3)
depth, sem = synth(cfg, 0.6, math.radians(12), 0.0)
r1 = core.process(depth, sem)
r2 = core.process(np.zeros_like(depth), sem)
check("ground occluded for a frame -> last plane held", f"source={r2.plane.source} h={r2.plane.height:.3f}", r2.plane.ok and r2.plane.source == "held" and abs(r2.plane.height - 0.6) < 0.01)
check("   ...but the held frame's grid is UNKNOWN (no depth = no evidence)", f"{int((r2.grid != cfg.UNKNOWN).sum())} seen cells", (r2.grid == cfg.UNKNOWN).all())
for _ in range(cfg.plane_hold_frames * 2 + 1):
    r3 = core.process(np.zeros_like(depth), sem)
check("held too long -> declared lost", f"ok={r3.plane.ok} source={r3.plane.source}", not r3.plane.ok)
r4 = core.process(depth, sem)
check("relocks as soon as ground is visible again", f"ok={r4.plane.ok} h={r4.plane.height:.3f}", r4.plane.ok and abs(r4.plane.height - 0.6) < 0.01)

print("\n8. LOCKS AND RELATIVE-DEPTH FALLBACK")
res_lock, _ = run(cfg_lock, *synth(cfg, 0.22, 0.0, 0.0))
check("fully locked rig skips estimation", f"source={res_lock.plane.source} h={res_lock.plane.height}", res_lock.plane.source == "locked" and res_lock.plane.height == 0.22)
depth, sem = synth(cfg, 0.6, math.radians(12), 0.0, boxes=[(3.0, 3.6, -0.4, 0.4, 0.0, 0.9)])
cfg_rel = mk_cfg(nominal_height=0.60)
res_rel, _ = run(cfg_rel, depth * 1.7, sem, metric=False)
check("unscaled depth x1.7 -> scale recovered from nominal height", f"scale={res_rel.scale:.3f} (want {1/1.7:.3f})", abs(res_rel.scale - 1 / 1.7) < 0.02)
fx_, fy_, n = lethal_centroid(cfg_rel, res_rel.grid)
check("   ...and the obstacle lands at the right range", f"X={fx_ if fx_ is None else round(fx_,2)} n={n}", n > 0 and abs(fx_ - 3.15) < 0.4)

print("\n9. SEMANTIC COST TABLE")
names = {0: "wall", 1: "sky", 2: "road", 3: "grass", 4: "tree", 5: "water", 6: "signboard", 7: "sidewalk", 8: "earth", 9: "zebra"}
lut = pc.build_cost_lut(names)
check("road/sidewalk 0, grass 40, earth 80", f"{lut[2]},{lut[7]},{lut[3]},{lut[8]}", lut[2] == 0 and lut[7] == 0 and lut[3] == 40 and lut[8] == 80)
check("wall/tree/signboard TALL lethal (250), water lethal (254), sky ignored, unknown 100", f"{lut[0]},{lut[4]},{lut[5]},{lut[6]},{lut[1]},{lut[9]}",
      lut[0] == 250 and lut[4] == 250 and lut[5] == 254 and lut[6] == 250 and lut[1] == -1 and lut[9] == 100)

print("\n10. RENDERING")
depth, sem = synth(cfg, 0.6, math.radians(12), 0.0, boxes=[(3.0, 3.6, 0.8, 1.6, 0.0, 0.9)])
res, _ = run(cfg, depth, sem)
img = pc.render_costmap(res.grid, cfg, 4, plane=res.plane, status="test")
mag = img[:, :, 2].astype(int) - img[:, :, 1].astype(int)
L, R_ = mag[:, :img.shape[1] // 2].sum(), mag[:, img.shape[1] // 2:].sum()
check("left obstacle draws on screen LEFT", f"L={L} R={R_}", L > R_ and img.dtype == np.uint8)
ix, iy = pc.px_to_cell(*pc.cell_to_px(10, 20, cfg.nx, cfg.ny, 4), cfg.nx, cfg.ny, 4)
check("cell_to_px / px_to_cell round trip", f"{(ix, iy)}", (ix, iy) == (10, 20))

print("\n11. THROUGHPUT (CPU, 1280x720)")
cfg_big = mk_cfg(1280, 720)
depth, sem = synth(cfg_big, 0.6, math.radians(12), 0.0, boxes=[(4.0, 4.6, -0.4, 0.4, 0.0, 0.9)])
core = pc.PerceptionCore(cfg_big)
core.process(depth, sem)
t = time.perf_counter(); N = 10
for _ in range(N):
    r = core.process(depth, sem)
ms = (time.perf_counter() - t) / N * 1000
check("full front-end under 90 ms/frame (CPU, 1280x720)", f"{ms:.1f} ms  ({', '.join(f'{k} {v:.1f}' for k, v in r.timing_ms.items())})", ms < 90)

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
