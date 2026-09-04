"""
Verification of the Nav2-style stack (navstack.py) and message shapes (ros_msgs.py).
Needs only numpy + opencv. No models, no camera, no GPU.

    python test_nav.py
"""
import math, sys, time, types
import numpy as np

# costmap_prototype imports torch at module level; stub it like test_geometry does
_t = types.ModuleType("torch")
_t.cuda = types.SimpleNamespace(is_available=lambda: False)
_t.backends = types.SimpleNamespace(mps=None)
_t.inference_mode = lambda: (lambda f: f)
sys.modules.setdefault("torch", _t)

import perception_core as pc
import navstack as ns
import ros_msgs as rm

FAILS = []


def check(name, got, ok):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
    print(f"  {tag}  {name:<56} {got}")


cfg = pc.CoreCfg(w=640, h=360, x_min=0.5, x_max=10.0, y_min=-4.0, y_max=4.0, res=0.1)
NX, NY = cfg.nx, cfg.ny
CTR = int(round((0.0 - cfg.y_min) / cfg.res))


def local_free():
    return np.zeros((NX, NY), np.uint8)


def local_with_block(x_m, y_m, half=0.3):
    g = local_free()
    i0, i1 = int((x_m - half - cfg.x_min) / cfg.res), int((x_m + half - cfg.x_min) / cfg.res)
    j0, j1 = int((y_m - half - cfg.y_min) / cfg.res), int((y_m + half - cfg.y_min) / cfg.res)
    g[max(i0, 0):i1, max(j0, 0):j1] = pc.LETHAL
    return g


# -------------------------------------------------------------------- tests --
print("\n1. POSE TRANSFORMS")
p = ns.Pose(2.0, 3.0, math.pi / 2)
rx, ry = p.to_robot(2.0, 5.0)
check("point 2 m north of a north-facing robot is 2 m ahead", f"({rx:.2f},{ry:.2f})", abs(rx - 2) < 1e-9 and abs(ry) < 1e-9)
rx, ry = p.to_robot(1.0, 3.0)
check("point 1 m west of a north-facing robot is 1 m LEFT", f"({rx:.2f},{ry:.2f})", abs(rx) < 1e-9 and abs(ry - 1) < 1e-9)
wx, wy = p.to_world(*p.to_robot(-4.0, 7.5))
check("to_robot / to_world round trip", f"({wx:.3f},{wy:.3f})", abs(wx + 4) < 1e-9 and abs(wy - 7.5) < 1e-9)

print("\n2. GLOBAL COSTMAP FUSION")
gm = ns.GlobalCostmap(res=0.25, size_m=60)
ix, iy = gm.world_to_cell(0.0, 0.0)
wx, wy = gm.cell_to_world(ix, iy)
check("world<->cell round trip lands inside the same cell", f"({wx:.3f},{wy:.3f})", abs(wx) <= 0.25 and abs(wy) <= 0.25)
for th_deg, want in ((0, (4.0, 1.0)), (90, (-1.0, 4.0)), (180, (-4.0, -1.0))):
    gm = ns.GlobalCostmap(res=0.25, size_m=60)
    pose = ns.Pose(0.0, 0.0, math.radians(th_deg))
    gm.fuse(local_with_block(4.0, 1.0), cfg, pose)      # obstacle 4 m ahead, 1 m left
    li = np.where(gm.grid == pc.LETHAL)
    cxw, cyw = gm.cell_to_world(li[0].mean(), li[1].mean())
    check(f"lethal 4 m ahead / 1 m left fuses at true world spot (heading {th_deg})",
          f"({cxw:.2f},{cyw:.2f}) want {want}", len(li[0]) > 0 and abs(cxw - want[0]) < 0.4 and abs(cyw - want[1]) < 0.4)
gm = ns.GlobalCostmap(res=0.25, size_m=60)
gm.fuse(local_with_block(4.0, 1.0), cfg, ns.Pose())
before = int((gm.grid == pc.LETHAL).sum())
gm.fuse(np.full((NX, NY), pc.UNKNOWN, np.uint8), cfg, ns.Pose())
check("UNKNOWN local cells never overwrite known cells", f"{before} -> {int((gm.grid == pc.LETHAL).sum())} lethal", (gm.grid == pc.LETHAL).sum() == before)
gm.fuse(local_free(), cfg, ns.Pose())
check("a later free observation does not erase lethal (max-fusion)", f"{int((gm.grid == pc.LETHAL).sum())} lethal", (gm.grid == pc.LETHAL).sum() == before)
known = gm.grid != pc.UNKNOWN
check("observed ground becomes known, rest stays UNKNOWN", f"{int(known.sum())} known of {gm.grid.size}", 0 < known.sum() < gm.grid.size * 0.2)
pooled = gm.pooled(2)
check("pooled copy keeps lethal and halves size", f"{pooled.shape} lethal={int((pooled == pc.LETHAL).sum())}", pooled.shape == (120, 120) and (pooled == pc.LETHAL).sum() > 0)

print("\n3. GLOBAL PLANNER")
gm = ns.GlobalCostmap(res=0.25, size_m=120)
pose = ns.Pose()
# a wall across the way from y=-6..6 at x=10, gap at y=7..8
wall = local_free()
gm.fuse(local_free(), cfg, pose)
gx0, gy0 = gm.world_to_cell(10.0, -6.0); gx1, gy1 = gm.world_to_cell(10.5, 6.0)
gm.grid[gx0:gx1 + 1, gy0:gy1 + 1] = pc.LETHAL
t = time.perf_counter()
path, reached = ns.plan_global(gm, pose, (20.0, 0.0))
ms = (time.perf_counter() - t) * 1000
check("global plan reaches a goal 20 m away around a wall", f"reached={reached} len={len(path)} {ms:.0f} ms", reached and len(path) > 0)
xs = [p_[0] for p_ in path]; ys = [p_[1] for p_ in path]
near_wall = [y for x, y in path if 9.5 <= x <= 11.0]
check("path detours around the wall ends (wall spans |y| <= 6)", f"|y| at wall = {min(abs(y) for y in near_wall) if near_wall else None}", near_wall and min(abs(y) for y in near_wall) >= 5.5)
check("global plan runs under 150 ms", f"{ms:.0f} ms", ms < 150)
check("path_blocked() is false for a clear path, true after a new obstacle",
      f"{ns.path_blocked(gm, path)} -> ", not ns.path_blocked(gm, path))
mx, my = path[len(path) // 2]
bx, by = gm.world_to_cell(mx, my); gm.grid[bx, by] = pc.LETHAL
check("   ...true once a path cell turns lethal", f"{ns.path_blocked(gm, path)}", ns.path_blocked(gm, path))
gm2 = ns.GlobalCostmap(res=0.25, size_m=120)
gm2.fuse(local_free(), cfg, pose)
sx, sy = gm2.world_to_cell(0.0, 0.0); gm2.grid[sx - 2:sx + 3, sy - 2:sy + 3] = pc.LETHAL
path2, reached2 = ns.plan_global(gm2, pose, (8.0, 0.0))
check("robot standing on smeared lethal cells can still plan", f"reached={reached2} len={len(path2)}", reached2 and len(path2) > 0)

print("\n3b. LOCAL PLANNER MEMORY")
gm = ns.GlobalCostmap(res=0.25, size_m=60)
gm.fuse(local_with_block(4.0, 1.0), cfg, ns.Pose())            # lethal seen at world (4, 1) while facing +x
turned = ns.Pose(0.0, 0.0, math.pi / 2)                        # now facing +y: that spot is 4 m to the RIGHT, 1 m ahead
filled = ns.fill_unknown_from_global(np.full((NX, NY), pc.UNKNOWN, np.uint8), gm, cfg, turned)
li = np.where(filled == pc.LETHAL)
fx_ = cfg.x_min + (li[0].mean() + 0.5) * cfg.res if len(li[0]) else None
fy_ = cfg.y_min + (li[1].mean() + 0.5) * cfg.res if len(li[0]) else None
check("lethal cell seen before the turn reappears in the local plan grid", f"robot-frame ({fx_ and round(fx_,1)}, {fy_ and round(fy_,1)}) want (1.0, -4.0)",
      len(li[0]) > 0 and abs(fx_ - 1.0) < 0.5 and abs(fy_ + 4.0) < 0.5)
check("   ...cells never observed stay UNKNOWN", f"{int((filled == pc.UNKNOWN).sum())} unknown of {filled.size}", (filled == pc.UNKNOWN).sum() > 0.5 * filled.size)

print("\n4. CARROT")
straight = [(float(x), 0.0) for x in range(0, 40)]
cx, cy = ns.carrot(straight, ns.Pose(), cfg, (39.0, 0.0))
check("far goal -> carrot ~x_max-1.5 ahead on the path", f"({cx:.1f},{cy:.1f})", abs(cx - (cfg.x_max - 1.5)) < 1.01 and abs(cy) < 1e-9)
cx, cy = ns.carrot(straight, ns.Pose(), cfg, (5.0, 1.0))
check("goal inside the local window -> carrot is the goal", f"({cx:.1f},{cy:.1f})", (cx, cy) == (5.0, 1.0))
cx, cy = ns.carrot(straight, ns.Pose(3.0, 0.0, math.pi), cfg, (39.0, 0.0))
check("goal behind the robot -> carrot has negative x (NOT clamped)", f"({cx:.1f},{cy:.1f})", cx < 0)

print("\n5. NAVIGATOR STATE MACHINE (scripted unicycle, free world)")
ncfg = ns.NavCfg(v_max=1.5, w_max=1.0, replan_period=0.5)
nav = ns.Navigator(cfg, ncfg, ns.GlobalCostmap(res=0.25, size_m=120))
out = nav.step(local_free(), ns.Pose(), now=0.0)
check("no goal -> NO_GOAL, stop", f"{out.status} v={out.v}", out.status == "NO_GOAL" and out.v == 0.0)
nav.set_goal(15.0, 8.0)
x = y = th = 0.0; dt = 0.2; states = []; vmax = wmax = 0.0; arrived = False
for k in range(400):
    out = nav.step(local_free(), ns.Pose(x, y, th), now=k * dt)
    states.append(out.status)
    vmax = max(vmax, out.v); wmax = max(wmax, abs(out.omega))
    if out.status == "ARRIVED":
        arrived = True
        break
    x += out.v * math.cos(th) * dt; y += out.v * math.sin(th) * dt; th += out.omega * dt
check("arrives at a goal 17 m away, off-axis", f"arrived={arrived} at ({x:.2f},{y:.2f}) in {k} steps", arrived and math.hypot(x - 15, y - 8) <= ncfg.goal_tol + 0.05)
check("visited DRIVING (and TURNING or not), never BLOCKED", f"{sorted(set(states))}", "DRIVING" in states and "BLOCKED" not in states)
check("commands within limits", f"v<={vmax:.2f} |w|<={wmax:.2f}", vmax <= ncfg.v_max + 1e-6 and wmax <= ncfg.w_max + 1e-6)

nav.set_goal(-10.0, 0.0)          # directly behind
out = nav.step(local_free(), ns.Pose(x, y, th), now=100.0)
# heading th ~ toward (15,8); goal (-10,0) is roughly behind
gx, gy = ns.Pose(x, y, th).to_robot(-10.0, 0.0)
check("goal behind -> TURNING with v = 0 and omega toward it", f"{out.status} v={out.v:.2f} w={out.omega:+.2f} (bearing {math.degrees(math.atan2(gy,gx)):+.0f})",
      out.status == "TURNING" and out.v == 0.0 and (out.omega > 0) == (math.atan2(gy, gx) > 0))

print("\n6. BLOCKED AND WATCHDOG")
nav = ns.Navigator(cfg, ncfg, None)          # local-only mode (no pose)
nav.set_goal(8.0, 0.0)
blk = local_free(); blk[0, CTR] = pc.LETHAL   # lethal dead ahead -> astar returns []
outs = [nav.step(blk, None, now=k * 0.2) for k in range(ncfg.blocked_frames + 3)]
check("lethal dead ahead -> BLOCKED, first stop then spin", f"{[o.status for o in outs][:3]}... v={outs[-1].v} w={outs[-1].omega:+.2f}",
      all(o.status == "BLOCKED" for o in outs) and outs[0].omega == 0.0 and outs[-1].omega != 0.0 and outs[-1].v == 0.0)
out = nav.step(local_free(), None, now=10.0)
check("clear again -> DRIVING straight at the carrot", f"{out.status} v={out.v:.2f} w={out.omega:+.2f}", out.status == "DRIVING" and out.v > 0 and abs(out.omega) < 0.05)
check("watchdog trips after a silent second", f"{nav.watchdog(now=11.5)} / {nav.watchdog(now=10.5)}", nav.watchdog(now=11.5) and not nav.watchdog(now=10.5))
nav2 = ns.Navigator(cfg, ncfg, None); nav2.set_goal(6.0, 2.0)
out = nav2.step(local_with_block(3.0, 0.0, 0.6), None, now=0.0)
check("local-only mode steers around an obstacle ahead", f"{out.status} v={out.v:.2f} w={out.omega:+.2f} path={len(out.local_path)}", out.status == "DRIVING" and out.v > 0 and len(out.local_path) > 0)

print("\n7. ROS MESSAGE SHAPES")
g = local_with_block(3.0, 0.0)
g[0, :] = pc.UNKNOWN
msg = rm.occupancy_grid(g, cfg.res, (cfg.x_min, cfg.y_min), frame_id="base_link")
data = np.array(msg["data"])
check("OccupancyGrid width/height/data length", f"{msg['info']['width']}x{msg['info']['height']} len={len(data)}",
      msg["info"]["width"] == NX and msg["info"]["height"] == NY and len(data) == NX * NY)
check("values in {-1, 0..100}, lethal -> 100, unknown -> -1", f"min={data.min()} max={data.max()} n100={int((data == 100).sum())} n-1={int((data == -1).sum())}",
      data.min() == -1 and data.max() == 100 and (data == 100).sum() == (g == pc.LETHAL).sum() and (data == -1).sum() == NY)
# data index = y*width + x  -> the unknown first row (x=0) must appear at every y
check("row-major with X fastest (index = y*width + x)", f"data[0::{NX}] all -1: {bool((data[0::NX] == -1).all())}", (data[0::NX] == -1).all())
od = rm.odometry(ns.Pose(1, 2, math.pi / 2), 0.5, 0.1)
check("Odometry yaw quaternion", f"z={od['pose']['pose']['orientation']['z']:.3f} w={od['pose']['pose']['orientation']['w']:.3f}",
      abs(od["pose"]["pose"]["orientation"]["z"] - math.sqrt(0.5)) < 1e-6 and od["twist"]["twist"]["linear"]["x"] == 0.5)
pm = rm.path_msg([(0, 0), (1, 0), (1, 1)])
check("Path has one pose per point with heading along the path", f"{len(pm['poses'])} poses, first yaw z={pm['poses'][0]['pose']['orientation']['z']:.2f}",
      len(pm["poses"]) == 3 and abs(pm["poses"][0]["pose"]["orientation"]["z"]) < 1e-9 and abs(pm["poses"][1]["pose"]["orientation"]["z"] - math.sin(math.pi / 4)) < 1e-6)
import json
check("all messages are JSON serialisable", "ok", bool(json.dumps([msg, od, pm, rm.twist(1, 2)])))

print("\n8. THROUGHPUT")
gm = ns.GlobalCostmap(res=0.25, size_m=120)
t = time.perf_counter()
for k in range(20):
    gm.fuse(local_with_block(4.0, 1.0), cfg, ns.Pose(k * 0.5, 0.0, 0.1 * k))
ms = (time.perf_counter() - t) / 20 * 1000
check("fuse() under 15 ms", f"{ms:.1f} ms", ms < 15)
t = time.perf_counter()
png = gm.to_png(pose=ns.Pose(5, 0, 0), crop_m=60)
ms = (time.perf_counter() - t) * 1000
check("global map PNG (60 m crop) under 30 ms and under 200 kB", f"{ms:.1f} ms, {len(png)/1024:.0f} kB", ms < 30 and len(png) < 200 * 1024)

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
