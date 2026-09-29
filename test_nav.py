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

print("\n6a. NO SILENT STANDSTILL (seen in the sim: sat still 'DRIVING' for minutes)")
walled = np.full((NX, NY), pc.UNKNOWN, np.uint8); walled[1, :] = pc.LETHAL; walled[0, :] = pc.LETHAL; walled[0, CTR] = 0
nav = ns.Navigator(cfg, ncfg, None); nav.set_goal(8.0, 0.0)
outs = [nav.step(walled, None, now=k * 0.2) for k in range(ncfg.blocked_frames + 3)]
check("a one-cell local path is BLOCKED (-> spin recovery), not DRIVING at v=0", f"{[o.status for o in outs][-2:]} w={outs[-1].omega:+.2f}",
      all(o.status == "BLOCKED" for o in outs) and outs[-1].omega != 0.0)

print("\n6a2. A GOAL NEXT TO A TRENCH (the sim rover drove into trench1 like this)")
cfg_t = pc.CoreCfg(w=640, h=360, x_min=0.5, x_max=10.0, y_min=-4.0, y_max=4.0, res=0.1, robot_radius=1.0)
def trench_world(x0=4.0, x1=5.4):
    g = np.zeros((cfg_t.nx, cfg_t.ny), np.uint8)
    g[int((x0 - cfg_t.x_min) / cfg_t.res):int((x1 - cfg_t.x_min) / cfg_t.res), :] = pc.LETHAL
    return g
nav = ns.Navigator(cfg_t, ns.NavCfg(v_max=1.0, goal_tol=0.3, slow_dist=1.0), None)
nav.set_goal(3.1, 0.0)                               # 0.9 m short of the lip: inside the 1 m clearance
x = 0.0; out = None
for k in range(200):
    loc = pc.inflate(trench_world(4.0 - x, 5.4 - x), cfg_t)
    nav.goal = (3.1 - x, 0.0)                        # robot-frame goal as the rover advances
    out = nav.step(loc, None, now=k * 0.2)
    if out.status == "ARRIVED":
        break
    x += out.v * 0.2
check("stops at the clearance, not at the lip (centre >= 1 robot radius from the edge)", f"{out.status} centre {x:.2f} m, lip 4.00 m, note='{out.note}'",
      out.status == "ARRIVED" and 4.0 - x >= 0.95 and "hazard" in out.note)
nav = ns.Navigator(cfg_t, ns.NavCfg(v_max=1.0), None); nav.set_goal(7.0, 0.0)
outs = [nav.step(pc.inflate(trench_world(), cfg_t), None, now=k * 0.2) for k in range(3)]
check("goal beyond a full-width trench: no path is ever planned into it", f"{[o.status for o in outs]} local path {len(outs[-1].local_path)}",
      all(o.status in ("BLOCKED", "DRIVING") for o in outs) and all(
          all(ix * cfg_t.res + cfg_t.x_min < 3.0 for ix, _ in o.local_path) for o in outs))

print("\n6a3. MEMORY HOLDS RAW OBSTACLES, INFLATION IS APPLIED AT PLAN TIME")
cfg_m = pc.CoreCfg(w=640, h=360, x_min=0.5, x_max=10.0, y_min=-4.0, y_max=4.0, res=0.1, robot_radius=1.0)
rock = np.zeros((cfg_m.nx, cfg_m.ny), np.uint8); rock[40:44, 38:42] = pc.LETHAL      # 0.4 m rock ~4.5 m ahead
gm_raw = ns.GlobalCostmap(res=0.1, size_m=40); gm_inf = ns.GlobalCostmap(res=0.1, size_m=40)
nav_raw = ns.Navigator(cfg_m, ns.NavCfg(), gm_raw); nav_inf = ns.Navigator(cfg_m, ns.NavCfg(), gm_inf)
nav_raw.set_goal(15.0, 0.0); nav_inf.set_goal(15.0, 0.0)
rng_m = np.random.default_rng(0)
for k in range(30):                              # the same rock, seen through a slightly noisy pose
    p_ = ns.Pose(rng_m.normal(0, 0.05), rng_m.normal(0, 0.05), rng_m.normal(0, 0.01))
    nav_raw.step(pc.inflate(rock, cfg_m), p_, now=k * 0.2, raw=rock)
    nav_inf.step(pc.inflate(rock, cfg_m), p_, now=k * 0.2)
n_raw, n_inf = int((gm_raw.grid >= 253).sum() - (gm_raw.grid == pc.UNKNOWN).sum()), int((gm_inf.grid >= 253).sum() - (gm_inf.grid == pc.UNKNOWN).sum())
check("remembered hazard footprint stays the rock, not a growing skirt", f"raw memory {n_raw} cells vs inflated-fusion {n_inf} cells",
      n_raw < 40 and n_inf > 10 * n_raw)
pg = nav_raw.last_plan_grid
ctr_m = int(round((0.0 - cfg_m.y_min) / cfg_m.res))
check("   ...and the planning grid is still inflated by a robot radius", f"cell 0.8 m from the rock = {pg[40, 38 - 8]} (skirt 253)",
      pg[40, 38 - 8] == 253 and pg[40, 38 - 12] < 253)

print("\n6b. LOST POSE (a global map, but no pose)")
nav = ns.Navigator(cfg, ncfg, ns.GlobalCostmap(res=0.25, size_m=60))
nav.set_goal(0.0, 6.0)                       # world goal 6 m north
out = nav.step(local_free(), ns.Pose(0.0, 0.0, 0.0), now=0.0)
check("with a pose: the goal is read in the world frame (turn left toward it)", f"{out.status} w={out.omega:+.2f}",
      out.status == "TURNING" and out.omega > 0)
out = nav.step(local_free(), None, now=0.2)
check("pose lost -> LOST, full stop (not the world goal read as robot-frame)", f"{out.status} v={out.v} w={out.omega}",
      out.status == "LOST" and out.v == 0.0 and out.omega == 0.0)
out = nav.step(local_free(), ns.Pose(0.0, 0.0, math.pi / 2), now=0.4)
check("pose back -> navigation resumes toward the world goal", f"{out.status} v={out.v:.2f}", out.status == "DRIVING" and out.v >= 0.0)
nav_lo = ns.Navigator(cfg, ncfg, None); nav_lo.set_goal(6.0, 0.0)
out = nav_lo.step(local_free(), None, now=0.0)
check("local-only mode (no map) still drives with no pose", f"{out.status} v={out.v:.2f}", out.status == "DRIVING" and out.v > 0)

print("\n6c. POSE INTEGRATION (exact SE(2))")
v_, w_, dt_ = 0.3, 0.5, 1 / 12
L_, d_ = 2 * v_ / w_ * math.sin(w_ * dt_ / 2), w_ * dt_
vp = ns.VisualOdomPose(); tx = ty = tth = 0.0
for _ in range(120):                          # 3 m of arc, as GroundVO reports it: chord in the previous frame
    vp.integrate(L_ * math.cos(d_ / 2), L_ * math.sin(d_ / 2), d_)
    tx += L_ * math.cos(tth + d_ / 2); ty += L_ * math.sin(tth + d_ / 2); tth += d_
err = math.hypot(vp.pose.x - tx, vp.pose.y - ty)
check("perfect arc increments integrate to the true pose (no double-counted turn)", f"err {err*1000:.4f} mm", err < 1e-6)

print("\n6d. GLOBAL MAP CLEARING (rover option; the sim keeps clear_after=0)")
gm = ns.GlobalCostmap(res=0.25, size_m=60, clear_after=3)
gm.fuse(local_with_block(4.0, 1.0), cfg, ns.Pose())
n0 = int((gm.grid == pc.LETHAL).sum())
gm.fuse(local_free(), cfg, ns.Pose()); gm.fuse(local_free(), cfg, ns.Pose())
n2 = int((gm.grid == pc.LETHAL).sum())
gm.fuse(local_free(), cfg, ns.Pose())
n3 = int((gm.grid == pc.LETHAL).sum())
check("an obstacle that moved away is cleared only after N free observations", f"{n0} -> {n2} (2 free) -> {n3} (3 free)", n0 > 0 and n2 == n0 and n3 == 0)
gm.fuse(local_with_block(4.0, 1.0), cfg, ns.Pose())
check("   ...and one lethal observation marks it again at once", f"{int((gm.grid == pc.LETHAL).sum())} lethal", (gm.grid == pc.LETHAL).sum() == n0)
gm.fuse(local_free(), cfg, ns.Pose()); gm.fuse(local_with_block(4.0, 1.0), cfg, ns.Pose())
gm.fuse(local_free(), cfg, ns.Pose()); gm.fuse(local_free(), cfg, ns.Pose())
check("   ...an interrupted free streak does not clear", f"{int((gm.grid == pc.LETHAL).sum())} lethal", (gm.grid == pc.LETHAL).sum() == n0)
gm_default = ns.GlobalCostmap(res=0.25, size_m=60)
check("default GlobalCostmap keeps pure max-fusion (sim unchanged)", f"clear_after={gm_default.clear_after}", gm_default.clear_after == 0 and gm_default.free_streak is None)

print("\n6e. D* LITE GLOBAL PLANNER")
from costmap_prototype import astar as _astar, traversal_cost as _tc
from dstar_lite import DStarLite
_pc = ns.PlannerCfg()


def _pcost(path, grid):
    c_, _ = _tc(grid, _pc); m_ = 1 + _pc.plan_cost_weight / 255 * c_.astype(np.float64)
    return sum(math.hypot(b_[0] - a_[0], b_[1] - a_[1]) * m_[b_] for a_, b_ in zip(path, path[1:]))


rng = np.random.default_rng(1); worst = 0.0; mism = 0; trials = 0
for trial in range(30):
    N_ = 50; grid = (rng.random((N_, N_)) * 120).astype(np.uint8)
    grid[rng.random((N_, N_)) < 0.25] = pc.LETHAL; grid[rng.random((N_, N_)) < 0.1] = pc.UNKNOWN
    st, gl = (2, 2), (N_ - 3, N_ - 3); grid[st] = 0; grid[gl] = 0
    c_, b_ = _tc(grid, _pc); ds = DStarLite(grid.shape, gl, c_, b_, _pc.plan_cost_weight)
    for step in range(6):
        pa, ra = _astar(grid, _pc, start=st, goal=gl); pd = ds.plan(st); trials += 1
        if ra != (pd is not None):
            mism += 1
        elif ra:
            worst = max(worst, abs(_pcost(pd, grid) - _pcost(pa, grid)) / max(_pcost(pa, grid), 1e-9))
        if pa and len(pa) > 3:
            st = pa[3]
        for i_, j_ in rng.integers(0, N_, (15, 2)):
            if (i_, j_) not in (st, gl):
                grid[i_, j_] = rng.choice([0, 60, pc.LETHAL, pc.UNKNOWN])
        c_, b_ = _tc(grid, _pc); ds.update_costs(c_, b_, st)
check("D* Lite == A* path cost on random grids, across incremental changes", f"{trials} plans, worst rel diff {worst:.1e}, reachability mismatches {mism}",
      worst < 1e-9 and mism == 0)
gm = ns.GlobalCostmap(res=0.25, size_m=120)
gm.fuse(local_free(), cfg, ns.Pose())
gx0, gy0 = gm.world_to_cell(10.0, -6.0); gx1, gy1 = gm.world_to_cell(10.5, 6.0)
gm.grid[gx0:gx1 + 1, gy0:gy1 + 1] = pc.LETHAL
dp = ns.DStarGlobalPlanner(ns.PlannerCfg(algo="dstar"))
pw_d, r_d = dp.plan(gm, ns.Pose(), (20.0, 0.0)); pw_a, r_a = ns.plan_global(gm, ns.Pose(), (20.0, 0.0))
lw = lambda p_: sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(p_, p_[1:]))
check("DStarGlobalPlanner detours the wall like plan_global()", f"reached={r_d} len {lw(pw_d):.1f} m vs A* {lw(pw_a):.1f} m",
      r_d and abs(lw(pw_d) - lw(pw_a)) < 1.5)
gm.grid[gm.world_to_cell(*pw_d[len(pw_d) // 2])] = pc.LETHAL
pw_d2, r_d2 = dp.plan(gm, ns.Pose(0.5, 0.0, 0.0), (20.0, 0.0))
check("   ...repairs incrementally when a path cell turns lethal", f"reached={r_d2} changed={dp.last_changed} cells, blocked={ns.path_blocked(gm, pw_d2)}",
      r_d2 and 0 < dp.last_changed < 50 and not ns.path_blocked(gm, pw_d2))
gm.grid[gx0 - 4:gx1 + 5, :] = pc.LETHAL          # wall the goal off entirely
pw_d3, r_d3 = dp.plan(gm, ns.Pose(0.5, 0.0, 0.0), (20.0, 0.0))
check("   ...unreachable goal falls back to A*'s closest-point path", f"reached={r_d3} len={len(pw_d3)} fallbacks={dp.fallbacks}",
      not r_d3 and len(pw_d3) > 0 and dp.fallbacks == 1)
nav = ns.Navigator(cfg, ncfg, ns.GlobalCostmap(res=0.25, size_m=120), ns.PlannerCfg(algo="dstar"))
nav.set_goal(15.0, 8.0)
x = y = th = 0.0; arrived = False
for k in range(400):
    out = nav.step(local_free(), ns.Pose(x, y, th), now=k * 0.2)
    if out.status == "ARRIVED":
        arrived = True
        break
    x += out.v * math.cos(th) * 0.2; y += out.v * math.sin(th) * 0.2; th += out.omega * 0.2
check("Navigator with D* Lite arrives at a goal 17 m away", f"arrived={arrived} at ({x:.2f},{y:.2f}) in {k} steps", arrived)

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

print("\n7b. STARTING INSIDE A SKIRT: THE ESCAPE LEADS OUT, NOT IN (FPV footage: drove 0.1 m from a rock)")
import cv2
cfg_s = pc.CoreCfg(w=640, h=360, x_min=0.3, x_max=8.0, y_min=-4.0, y_max=4.0, res=0.1, robot_radius=0.35)
ctr_s = int(round((0.0 - cfg_s.y_min) / cfg_s.res))
raw_s = np.zeros((cfg_s.nx, cfg_s.ny), np.uint8)
raw_s[3:8, ctr_s - 6:ctr_s + 7] = pc.LETHAL          # a rock 0.6-1.0 m ahead, 1.3 m wide
grid_s = pc.inflate(raw_s, cfg_s)
clear_s = cv2.distanceTransform((grid_s != pc.LETHAL).astype(np.uint8), cv2.DIST_L2, 5)
from costmap_prototype import astar
path_s, _ = astar(grid_s, ns._LocalCfg(cfg_s, ns.NavCfg(), 7.0, 0.0), goal=ns.local_goal_cell(cfg_s, 7.0, 0.0))
c0 = clear_s[0, ctr_s]
worst = min((clear_s[c] for c in path_s), default=c0)
check("start cell is inside the skirt (the precondition)", f"start={int(grid_s[0, ctr_s])} clearance {c0 * cfg_s.res:.2f} m", grid_s[0, ctr_s] == pc.LETHAL - 1)
check("no step of the local A* plan gets closer to the rock than the start", f"{len(path_s)} cells, closest {worst * cfg_s.res:.2f} m vs start {c0 * cfg_s.res:.2f} m",
      worst >= c0 - 1e-3)

nav_c = ns.Navigator(cfg_s, ns.NavCfg(accel_max=100.0, escape_v=0.15), None); nav_c.set_goal(7.0, 0.0)
side = np.zeros((cfg_s.nx, cfg_s.ny), np.uint8); side[0:40, ctr_s + 3:ctr_s + 7] = pc.LETHAL   # a wall 0.3 m to the side
out = nav_c.step(pc.inflate(side, cfg_s), None, now=0.0)
check("inside a skirt with a way out: crawls at escape_v, says why", f"{out.status} v={out.v:.2f} note='{out.note}'",
      out.status == "DRIVING" and 0.0 < out.v <= 0.15 + 1e-9 and "clearance" in out.note)

print("\n7c. UNKNOWN-PATH GATE (real cameras only; the sim keeps it off)")
unk_all = np.full((NX, NY), pc.UNKNOWN, np.uint8)
nav = ns.Navigator(cfg, ns.NavCfg(), None); nav.set_goal(8.0, 0.0)
out = nav.step(unk_all, None, now=0.0)
check("gate off (the sim default): an all-UNKNOWN map still drives", f"{out.status} v={out.v:.2f}", out.status == "DRIVING")
nav = ns.Navigator(cfg, ns.NavCfg(unknown_gate=True), None); nav.set_goal(8.0, 0.0)
out = nav.step(unk_all, None, now=0.0)
check("gate on: an all-UNKNOWN path is BLOCKED at v=0", f"{out.status} v={out.v:.2f} note='{out.note}'", out.status == "BLOCKED" and out.v == 0.0)
part = local_free(); part[5:10, :] = pc.UNKNOWN        # 0.5 of the first 1.5 m unknown, no way round
nav_a = ns.Navigator(cfg, ns.NavCfg(accel_max=100.0), None); nav_a.set_goal(8.0, 0.0)
nav_b = ns.Navigator(cfg, ns.NavCfg(accel_max=100.0, unknown_gate=True), None); nav_b.set_goal(8.0, 0.0)
oa, ob = nav_a.step(part, None, now=0.0), nav_b.step(part, None, now=0.0)
check("gate on: a partly-unknown path drives at half speed", f"{ob.status} v={ob.v:.2f} vs {oa.v:.2f} ungated, note='{ob.note}'",
      ob.status == "DRIVING" and abs(ob.v - 0.5 * oa.v) < 1e-6 and "unknown" in ob.note)
nav = ns.Navigator(cfg, ns.NavCfg(unknown_gate=True), None); nav.set_goal(8.0, 0.0)
out = nav.step(local_free(), None, now=0.0)
check("gate on: a fully measured free path is untouched", f"{out.status} note='{out.note}'", out.status == "DRIVING" and out.note == "")

print("\n7d. GLOBAL MEMORY: ONE LETHAL SPECK IS NOT A WALL (FPV trail: 17% specks blocked the plan)")
speck = local_free(); speck[20, CTR] = pc.LETHAL                         # one cell, 2.5 m ahead
def fused_value(gm_, times):
    for _ in range(times):
        gm_.fuse(speck, cfg, ns.Pose(0.0, 0.0, 0.0))
    return int(gm_.grid[gm_.world_to_cell(cfg.x_min + 20.5 * cfg.res, 0.05)])
v_def = fused_value(ns.GlobalCostmap(res=0.1, size_m=30.0, clear_after=4), 1)
v_one = fused_value(ns.GlobalCostmap(res=0.1, size_m=30.0, clear_after=4, lethal_confirm=3), 1)
v_three = fused_value(ns.GlobalCostmap(res=0.1, size_m=30.0, clear_after=4, lethal_confirm=3), 3)
check("default (the sim): one lethal sighting is remembered lethal", f"{v_def}", v_def == pc.LETHAL)
check("lethal_confirm=3: one sighting is stored expensive, not blocking", f"{v_one} (= UNCONFIRMED {ns.UNCONFIRMED})", v_one == ns.UNCONFIRMED)
check("lethal_confirm=3: three sightings confirm it lethal", f"{v_three}", v_three == pc.LETHAL)

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
