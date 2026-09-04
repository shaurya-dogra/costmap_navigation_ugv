#!/usr/bin/env python3
"""
Geometric ground-truth simulator - SIH PS 26126
----------------------------------------------
Closed-loop test bench for the navigation half of the pipeline:

    world  ->  exact depth + semantics  ->  costmap  ->  A*  ->  (v, omega)
      ^                                                              |
      +--------------------- robot drives ---------------------------+

Why this exists: the neural perception stack can only be judged against real
footage, but the COSTMAP and PLANNER can be judged against known geometry. This
file ray-casts an analytic outdoor scene through the same `Cfg` camera model, so
every obstacle has a known metric position and the costmap can be scored rather
than eyeballed.

It deliberately does NOT run Depth Anything or the segmenter. It substitutes for
`Perception` at exactly the boundary the project reserves for that swap, and it
makes no claim about how well monocular depth works outdoors.

    python sim.py --validate       # metric accuracy of costmap + planner
    python sim.py --demo out.mp4   # closed-loop drive, written to video
    python sim.py                  # live windows

Semantic classes are ground truth, so the interesting cases are the ones where
semantics is USELESS and only geometry saves the robot:
  - `rock`  looks like unremarkable terrain (cost 100) and is lethal by HEIGHT
  - `ditch` looks exactly like road      (cost 0)   and is lethal by DEPTH
"""

import argparse
import numpy as np
import cv2

from costmap_prototype import (Cfg, Costmap, backproject, render, astar,
                               drive_command, goal_cell, fit_ground_plane)

# --- ground-truth semantic costs (what the segmenter would ideally output) ---
C_ROAD, C_GRASS, C_ROCK, C_SKY = 0.0, 40.0, 100.0, -1.0


class World:
    """
    An outdoor scene in world coordinates (X east, Y north, Z up, metres).

    cylinders : (x, y, radius, height) positive obstacles
    trenches  : (x_lo, x_hi, depth)    narrow ditches running across the path
    drops     : (x_edge, depth)        step-down / kerb: ground falls beyond x_edge
    road_half : half-width of the road corridor about Y = 0

    A trench and a drop are both negative obstacles but they behave completely
    differently to a low camera. Past a drop you eventually see the lower
    surface, so geometry catches it. A narrow trench's floor is hidden behind
    its own far wall, so it can only ever read UNKNOWN. See §S3 of validate().
    """

    def __init__(self, cylinders=(), trenches=(), drops=(), road_half=1.6):
        self.cylinders = list(cylinders)
        self.trenches = list(trenches)
        self.drops = list(drops)
        self.road_half = road_half

    @staticmethod
    def default():
        """A course that exercises every channel: swerve, swerve, then stop."""
        return World(
            cylinders=[(4.0,  0.9, 0.40, 0.8),    # rock, left of centre
                       (7.5, -1.0, 0.35, 1.1)],   # post, right of centre
            drops=[(11.0, 0.45)],                 # kerb drop at 11 m
            road_half=1.8,
        )

    def shadow_end(self, cfg):
        """Where the lower ground past the first drop becomes visible."""
        if not self.drops:
            return None
        xe, d = self.drops[0]
        return xe * (cfg.cam_height + d) / cfg.cam_height


def look(world, cfg, pose, w, h):
    """
    Ray-cast the world from `pose` = (x, y, theta) and return
    (depth, sem_cost, rgb) exactly as Perception would.

    `depth` is optical-axis distance, matching what backproject() expects:
    a pixel ray is parametrised as Xc,Yc,Zc = t*(xn, yn, 1), so t IS the depth.
    """
    px, py, th = pose
    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    xn = (u - cfg.cx * w / cfg.proc_w) / (cfg.fx * w / cfg.proc_w)
    yn = (v - cfg.cy * h / cfg.proc_h) / (cfg.fy * h / cfg.proc_h)

    c, s = np.cos(cfg.cam_pitch), np.sin(cfg.cam_pitch)
    # robot-frame ray: X = t*A, Y = t*B, Z = t*C + cam_height
    A = c - yn * s
    B = -xn
    C = -(s + yn * c)

    INF = 1e6
    t_best = np.full((h, w), INF, np.float32)
    sem = np.full((h, w), C_SKY, np.float32)

    def commit(t, mask, cost):
        nonlocal t_best, sem
        take = mask & (t > 0.05) & (t < t_best)
        t_best = np.where(take, t, t_best)
        sem = np.where(take, cost, sem)
        return take

    # ---- ground plane Z = 0, and any trench floor below it -----------------
    denom = np.where(np.abs(C) < 1e-6, np.nan, C)
    t_gnd = -cfg.cam_height / denom                       # Z(t) = 0
    with np.errstate(invalid="ignore"):
        hit_g = np.isfinite(t_gnd) & (t_gnd > 0.05)

    def to_world(t):
        X, Y = t * A, t * B
        return (px + X * np.cos(th) - Y * np.sin(th),
                py + X * np.sin(th) + Y * np.cos(th))

    wx_g, wy_g = to_world(np.where(hit_g, t_gnd, 0.0))
    on_road_g = np.abs(wy_g) <= world.road_half

    # trench floors are horizontal planes at Z = -d, seen through the opening
    trench_hit = np.zeros((h, w), bool)
    t_floor = np.full((h, w), INF, np.float32)
    for x_lo, x_hi, d in world.trenches:
        t_d = -(cfg.cam_height + d) / denom
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(t_d) & (t_d > 0.05)
        wx_d, wy_d = to_world(np.where(ok, t_d, 0.0))
        inside = ok & (wx_d >= x_lo) & (wx_d <= x_hi) & (np.abs(wy_d) <= 6.0)
        t_floor = np.where(inside & (t_d < t_floor), t_d, t_floor)
        trench_hit |= inside
        # ground above an open trench is not there at all
        hit_g &= ~((wx_g >= x_lo) & (wx_g <= x_hi) & (np.abs(wy_g) <= 6.0))

    # ---- step-downs: ground sits at Z = -d beyond x_edge -------------------
    # Both planes are cast and each is kept only where its own hit point is on
    # the correct side of the edge. The wedge between them is the occlusion
    # shadow of the lip, and it correctly receives no ray at all -> UNKNOWN.
    for xe, d in world.drops:
        hit_g &= ~(wx_g >= xe)                       # upper plane ends at the lip
        t_lo = -(cfg.cam_height + d) / denom
        with np.errstate(invalid="ignore"):
            ok_lo = np.isfinite(t_lo) & (t_lo > 0.05)
        wx_lo, wy_lo = to_world(np.where(ok_lo, t_lo, 0.0))
        lower = ok_lo & (wx_lo >= xe)
        commit(t_lo, lower & (np.abs(wy_lo) <= world.road_half), C_ROAD)
        commit(t_lo, lower & (np.abs(wy_lo) > world.road_half), C_GRASS)

    commit(t_gnd, hit_g & on_road_g, C_ROAD)
    commit(t_gnd, hit_g & ~on_road_g, C_GRASS)
    # a trench floor still LOOKS like road - only geometry can catch it
    commit(t_floor, trench_hit & (t_floor < INF), C_ROAD)

    # ---- vertical cylinders -------------------------------------------------
    for cxw, cyw, r, hz in world.cylinders:
        dx, dy = cxw - px, cyw - py
        mx = dx * np.cos(th) + dy * np.sin(th)      # centre in robot frame
        my = -dx * np.sin(th) + dy * np.cos(th)
        qa = A * A + B * B
        qb = -2.0 * (A * mx + B * my)
        qc = mx * mx + my * my - r * r
        disc = qb * qb - 4.0 * qa * qc
        ok = disc > 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t_c = np.where(ok, (-qb - sq) / (2.0 * qa), INF)
        z_at = t_c * C + cfg.cam_height
        commit(t_c, ok & (z_at >= 0.0) & (z_at <= hz), C_ROCK)

    depth = np.where(t_best < INF, t_best, 0.0).astype(np.float32)

    # ---- a viewable image (cosmetic only, never fed to the maths) ----------
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (196, 158, 116)                                  # sky
    shade = np.clip(1.0 - depth / 25.0, 0.35, 1.0)[..., None]
    tex = ((np.sin(u * 0.7) * np.sin(v * 0.9)) * 9).astype(np.int16)[..., None]
    for cost, base in ((C_ROAD, (96, 96, 100)), (C_GRASS, (70, 124, 74)),
                       (C_ROCK, (105, 112, 126))):
        m = sem == cost
        if m.any():
            rgb[m] = np.clip(np.array(base) * shade + tex, 0, 255).astype(np.uint8)[m]
    # lane markings, purely so the video reads as a road
    lane = (sem == C_ROAD) & (np.abs(np.abs(wy_g) - world.road_half + 0.15) < 0.06)
    rgb[lane & (depth > 0)] = (210, 210, 210)
    return depth, sem, rgb


def fov_floor_range(cfg, h_img, drop_depth):
    """
    Forward range at which a surface `drop_depth` below the robot first enters
    the camera's downward field of view.

    A step-down has TWO visibility limits and the costmap can only mark what it
    can see past the later of them: the lip's occlusion shadow, and the bottom
    edge of the sensor. Close to a drop the sensor edge is the binding one.
    """
    fy = cfg.fy * h_img / cfg.proc_h
    cy = cfg.cy * h_img / cfg.proc_h
    yn = (h_img - 1 - cy) / fy                  # bottom row of the image
    c, s = np.cos(cfg.cam_pitch), np.sin(cfg.cam_pitch)
    A = c - yn * s
    C = -(s + yn * c)
    if C >= -1e-9:
        return float("inf")
    return float((cfg.cam_height + drop_depth) * A / (-C))


def perceive(world, cfg, pose, cm, w=640, h=360):
    """One full cycle: ray-cast -> costmap -> plan -> drive command."""
    depth, sem, rgb = look(world, cfg, pose, w, h)
    X, Y, Z, _ = backproject(depth, cfg_scaled(cfg, w, h))
    grid = cm.build(X, Y, Z, sem, np.zeros((0, 4)), depth)
    goal = goal_cell(cfg, cm.nx, cm.ny)
    path, reached = astar(grid, cfg, goal=goal)
    v, omega, aim = drive_command(path, cfg)
    return dict(depth=depth, sem=sem, rgb=rgb, grid=grid, goal=goal,
                path=path, reached=reached, v=v, omega=omega, aim=aim)


def cfg_scaled(cfg, w, h):
    """The same rig, described for a w*h image instead of proc_w*proc_h."""
    k = type("C", (), {a: getattr(cfg, a) for a in dir(cfg)
                       if not a.startswith("__")})()
    k.fx = cfg.fx * w / cfg.proc_w
    k.fy = cfg.fy * h / cfg.proc_h
    k.cx = cfg.cx * w / cfg.proc_w
    k.cy = cfg.cy * h / cfg.proc_h
    return k


# ----------------------------------------------------------------------------
# validation: does a known obstacle land where it actually is?
# ----------------------------------------------------------------------------

def validate():
    cfg = Cfg()
    cfg.cam_pitch = np.deg2rad(12.0)
    cfg.cam_height = 0.60
    cm = Costmap(cfg)
    fails = []

    def check(name, got, ok):
        if not ok:
            fails.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<48} {got}")

    print("\nS1. EMPTY ROAD")
    w = World(cylinders=[], trenches=[], road_half=3.5)
    r = perceive(w, cfg, (0.0, 0.0, 0.0), cm)
    lethal = int((r["grid"] == cfg.LETHAL).sum())
    check("clear road has no lethal cells", f"{lethal} lethal", lethal == 0)
    check("clear road reaches the 8 m goal",
          f"reached={r['reached']} len={len(r['path'])}", r["reached"])
    check("clear road drives straight at full speed",
          f"v={r['v']:.2f} m/s w={r['omega']:+.3f} rad/s",
          abs(r["v"] - cfg.v_max) < 1e-6 and abs(r["omega"]) < 1e-3)

    print("\nS2. POSITIVE OBSTACLE, LETHAL BY HEIGHT NOT BY LABEL")
    w = World(cylinders=[(4.0, 0.0, 0.4, 0.9)], trenches=[], road_half=3.5)
    r = perceive(w, cfg, (0.0, 0.0, 0.0), cm)
    li = np.where(r["grid"] == cfg.LETHAL)
    fx_ = cfg.x_min + li[0].mean() * cfg.res
    fy_ = cfg.y_min + li[1].mean() * cfg.res
    check("rock (cost 100, never lethal by label) is lethal",
          f"{len(li[0])} lethal cells", len(li[0]) > 0)
    # A costmap can only report the surface it can see. The near face of a
    # 0.4 m-radius cylinder centred at 4.0 m is at 3.6 m, so that - not 4.0 -
    # is the correct answer, and asserting the centre would hide a real bias.
    check("rock lands at its visible near face",
          f"X={fx_:.2f} m (near face 3.60, centre 4.00)", abs(fx_ - 3.6) < 0.25)
    check("rock lands on the centre line", f"Y={fy_:+.2f} m (true 0.00)",
          abs(fy_) < 0.4)

    print("\nS3. NEGATIVE OBSTACLE: A STEP DOWN THAT LOOKS EXACTLY LIKE ROAD")
    # The whole safety argument of the project: this surface is labelled `road`
    # (cost 0) and is indistinguishable from road in pixels. Only the height
    # channel can catch it.
    drop_x, drop_d = 3.5, 0.35
    w = World(cylinders=[], drops=[(drop_x, drop_d)], road_half=3.5)
    r = perceive(w, cfg, (0.0, 0.0, 0.0), cm)
    li = np.where(r["grid"] == cfg.LETHAL)
    check("step-down labelled `road` is lethal by geometry",
          f"{len(li[0])} lethal cells", len(li[0]) > 0)
    shadow = w.shadow_end(cfg)
    tx = cfg.x_min + li[0].min() * cfg.res
    check("lethal cells start where the lower ground becomes visible",
          f"X={tx:.2f} m (lip {drop_x:.1f} m, shadow ends {shadow:.2f} m)",
          abs(tx - shadow) < 1.0)
    check("path never crosses a lethal cell",
          f"{sum(1 for ix, iy in r['path'] if r['grid'][ix, iy] == cfg.LETHAL)} on path",
          not any(r["grid"][ix, iy] == cfg.LETHAL for ix, iy in r["path"]))
    # At 5.6 m the hazard is still far away, so full speed is correct here; what
    # matters is that the PLAN refuses to cross it. The approach is checked
    # closed-loop in S7.
    far = max(cfg.x_min + ix * cfg.res for ix, _ in r["path"])
    check("plan stops short of the edge instead of crossing it",
          f"plan reaches X={far:.2f} m, reached={r['reached']}",
          not r["reached"] and far < shadow + 0.3)

    print("\nS3b. A NARROW TRENCH IS UNKNOWN, NOT LETHAL (honest limitation)")
    # A 0.7 m wide, 0.6 m deep trench at 5 m: from a 0.60 m camera the floor is
    # hidden behind the trench's own far wall, so no ray ever measures it. The
    # grid says UNKNOWN, which is correct and is why UNKNOWN must stay expensive.
    w = World(cylinders=[], trenches=[(5.0, 5.7, 0.6)], road_half=3.5)
    r = perceive(w, cfg, (0.0, 0.0, 0.0), cm)
    band = r["grid"][int((5.0 - cfg.x_min) / cfg.res):int((5.7 - cfg.x_min) / cfg.res), :]
    frac_unk = float((band == cfg.UNKNOWN).mean())
    check("occluded trench reads UNKNOWN, never free",
          f"{100*frac_unk:.0f}% of the trench band is UNKNOWN", frac_unk > 0.8)
    check("no cell in the trench band is reported free",
          f"{int((band < 100).sum())} free cells", (band < 100).sum() == 0)

    print("\nS4. STEERING AROUND AN OBSTACLE")
    # The obstacle must straddle the centre line, or going straight is already
    # legal and the correct command is w = 0.
    for y_obs, want, lbl in ((0.55, "right", "obstacle LEFT  -> steer right"),
                             (-0.55, "left", "obstacle RIGHT -> steer left")):
        w = World(cylinders=[(3.5, y_obs, 0.7, 1.0)], road_half=3.5)
        r = perceive(w, cfg, (0.0, 0.0, 0.0), cm)
        got = "left" if r["omega"] > 0.01 else ("right" if r["omega"] < -0.01 else "straight")
        check(lbl, f"w={r['omega']:+.3f} rad/s -> {got}", got == want)

    print("\nS5. ROAD CORRIDOR IS PREFERRED OVER GRASS")
    w = World(cylinders=[], trenches=[], road_half=1.6)
    r = perceive(w, cfg, (0.0, 0.0, 0.0), cm)
    off = [abs(cfg.y_min + iy * cfg.res) for _, iy in r["path"]]
    check("path stays inside the road corridor",
          f"max |Y| = {max(off):.2f} m (road half-width 1.60)",
          max(off) <= 1.6)

    print("\nS6. CLOSED-LOOP DRIVE TO A FIXED GOAL")
    # Swerve around a rock at 4 m and a post at 7.5 m, then arrive at (9.5, 0),
    # which sits 1.5 m short of the kerb drop.
    course = World.default()
    goal = (9.5, 0.0)
    res = drive(course, cfg, steps=300, dt=0.15, quiet=True, goal_world=goal)
    check("robot arrives at the goal",
          f"arrived={res['arrived']} at ({res['x']:.2f},{res['y']:+.2f}) "
          f"in {res['steps']} steps", res["arrived"])
    check("robot clears both obstacles without touching them",
          f"{res['collisions']} collisions", res["collisions"] == 0)
    check("robot never crosses the kerb drop",
          f"final X={res['x']:.2f} m (lip at {course.drops[0][0]:.1f} m)",
          res["x"] < course.drops[0][0])
    check("route is not wildly longer than the straight line",
          f"drove {res['dist']:.1f} m for a {np.hypot(*goal):.1f} m hop",
          res["dist"] < 2.5 * np.hypot(*goal))
    check("commands stay inside the actuator limits",
          f"v<={res['vmax']:.2f} |w|<={res['wmax']:.2f}",
          res["vmax"] <= cfg.v_max + 1e-6 and res["wmax"] <= cfg.w_max + 1e-6)

    print("\nS7. COSTMAP ACCURACY THROUGH A FULL APPROACH")
    # Regression guard for the plane-anchor bug. Walking up to a step-down used
    # to make the ground fit tilt onto the lower surface (a swung to -0.49,
    # c to -0.45), the step measured zero relative height, and the costmap
    # reported the drop as FREE at point-blank range. Every row below must keep
    # the plane anchored and the hazard lethal.
    edge, depth_d = 11.0, 0.45
    w1 = World(cylinders=[], drops=[(edge, depth_d)], road_half=3.5)
    cm1 = Costmap(cfg)
    worst_a = worst_c = 0.0
    range_err = []
    false_free = 0
    for cam_x in (6.0, 7.0, 8.0, 9.0, 9.5, 10.0, 10.4):
        import costmap_prototype as _cp
        _cp._PLANE_STATE.update({"a": 0, "b": 0, "c": 0, "valid": False})
        d, sm, _ = look(w1, cfg, (cam_x, 0.0, 0.0), 640, 360)
        k = cfg_scaled(cfg, 640, 360)
        Xa, Ya, Za, _ = backproject(d, k)
        sub = (slice(None, None, 2), slice(None, None, 2))
        ds = d[sub].ravel()
        a_, b_, c_ = fit_ground_plane(
            Xa[sub].ravel(), Ya[sub].ravel(), Za[sub].ravel(), sm[sub].ravel(),
            near_x=cfg.plane_near_x, valid=(ds > 0.3) & (ds < cfg.max_depth),
            max_c=cfg.plane_max_c, gate=cfg.plane_gate)
        worst_a = max(worst_a, abs(a_)); worst_c = max(worst_c, abs(c_))

        _cp._PLANE_STATE.update({"a": 0, "b": 0, "c": 0, "valid": False})
        g = cm1.build(Xa, Ya, Za, sm, np.zeros((0, 4)), d)
        li = np.where(g == cfg.LETHAL)
        occl = (edge - cam_x) * (cfg.cam_height + depth_d) / cfg.cam_height
        shadow_r = max(occl, fov_floor_range(cfg, 360, depth_d))
        if len(li[0]) == 0:
            false_free += 1
            continue
        first = cfg.x_min + li[0].min() * cfg.res
        if cfg.x_min < shadow_r < cfg.x_max:
            # One cell of quantisation plus 3% of range: a cell 9 m out is
            # thinner in pixels than a cell 3 m out, so a fixed tolerance would
            # be slack near the robot and unfair far from it.
            tol = cfg.res + 0.03 * shadow_r
            range_err.append((abs(first - shadow_r), shadow_r, tol))
        # nothing beyond the lip may ever read as drivable
        beyond = g[int((shadow_r - cfg.x_min) / cfg.res):, :] if shadow_r < cfg.x_max else g[:0]
        false_free += int((beyond < 100).sum() > 0)

    check("ground plane stays anchored at every approach range",
          f"max |a| = {worst_a:.3f} (was -0.49), max |c| = {worst_c:.3f} m",
          worst_a < 0.05 and worst_c < 0.05)
    check("step-down is lethal at every approach range",
          f"{false_free} frames reported the drop as drivable", false_free == 0)
    worst = max(range_err, key=lambda t: t[0] - t[2])
    check("lethal band lands within one cell + 3% of range, at every range",
          f"worst {worst[0]:.2f} m error at {worst[1]:.2f} m "
          f"(allowed {worst[2]:.2f}); sensor floor {fov_floor_range(cfg, 360, depth_d):.2f} m",
          all(e <= t for e, _, t in range_err))

    print("\nS8. NO FALSE READINGS ON CLEAN GROUND")
    import costmap_prototype as _cp
    _cp._PLANE_STATE.update({"a": 0, "b": 0, "c": 0, "valid": False})
    flat = World(cylinders=[], road_half=4.5)
    gflat = perceive(flat, cfg, (0.0, 0.0, 0.0), Costmap(cfg))["grid"]
    check("flat ground produces no lethal cell",
          f"{int((gflat == cfg.LETHAL).sum())} lethal of {gflat.size}",
          (gflat == cfg.LETHAL).sum() == 0)
    seen = gflat != cfg.UNKNOWN
    check("everything the camera measured reads drivable",
          f"{int((gflat[seen] < 100).sum())}/{int(seen.sum())} measured cells low-cost",
          (gflat[seen] < 100).all())
    check("everything outside the field of view stays UNKNOWN",
          f"{int((~seen).sum())} unmeasured cells, none free",
          not (gflat[~seen] < 100).any())

    print("\n" + ("ALL SIM CHECKS PASSED" if not fails
                  else f"{len(fails)} FAILED: {fails}"))
    return 1 if fails else 0


# ----------------------------------------------------------------------------
# closed-loop drive
# ----------------------------------------------------------------------------

def drive(world, cfg, steps=200, dt=0.15, quiet=False, writer=None, show=False,
          goal_world=None, goal_tol=0.6):
    """
    Integrate the robot forward under its own commands. Returns a summary.

    The robot is a point at (x, y, theta) obeying a unicycle model. Each step it
    re-perceives from scratch: single-frame grids only, no temporal fusion.

    `goal_world` is a FIXED destination in world coordinates, re-expressed in
    the robot frame every step. This is the one place the simulator uses its
    privileged ground-truth pose: the live monocular rig has no pose source, so
    costmap_prototype.py keeps the spec's robot-frame carrot (a goal 8 m dead
    ahead) instead. Without a fixed goal the robot has nothing to converge to -
    it will happily drive along a hazard edge forever, which is safe but is not
    navigation. Give the sim a world goal and the same planner terminates.
    """
    cm = Costmap(cfg)
    x = y = th = 0.0
    dist, stalls, collisions = 0.0, 0, 0
    vmax = wmax = 0.0
    arrived = False
    steps_taken = 0

    for k in range(steps):
        steps_taken = k + 1
        if goal_world is not None:
            dx, dy = goal_world[0] - x, goal_world[1] - y
            if np.hypot(dx, dy) <= goal_tol:
                arrived = True
                break
            # goal into the robot frame, clamped into the grid
            fwd = dx * np.cos(th) + dy * np.sin(th)
            lat = -dx * np.sin(th) + dy * np.cos(th)
            cfg.goal_x = float(np.clip(fwd, cfg.x_min + cfg.res, cfg.x_max - cfg.res))
            cfg.goal_y = float(np.clip(lat, cfg.y_min + cfg.res, cfg.y_max - cfg.res))
        r = perceive(world, cfg, (x, y, th), cm)
        v, omega = r["v"], r["omega"]
        vmax = max(vmax, v); wmax = max(wmax, abs(omega))
        if v <= 1e-6:
            stalls += 1

        # collision test against ground truth, not against our own costmap
        for cxw, cyw, rad, _ in world.cylinders:
            if np.hypot(cxw - x, cyw - y) < rad:
                collisions += 1
        for x_lo, x_hi, _ in world.trenches:
            if x_lo <= x <= x_hi:
                collisions += 1
        for xe, _ in world.drops:
            if x >= xe:
                collisions += 1

        if writer is not None or show:
            frame = compose(r, cfg, (x, y, th), k)
            if writer is not None:
                writer.write(frame)
            if show:
                cv2.imshow("sim: camera | costmap", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th += omega * dt
        dist += v * dt
        if not quiet and k % 20 == 0:
            print(f"  t={k*dt:5.1f}s  pos=({x:5.2f},{y:+5.2f}) th={np.rad2deg(th):+6.1f}deg "
                  f"v={v:.2f} w={omega:+.2f} path={len(r['path']):3d} reached={r['reached']}")
    return dict(dist=dist, stalls=stalls, collisions=collisions,
                vmax=vmax, wmax=wmax, x=x, y=y, th=th,
                arrived=arrived, steps=steps_taken)


def compose(r, cfg, pose, k):
    """camera view | costmap+plan, side by side at a fixed size."""
    cam = cv2.resize(r["rgb"], (640, 360))
    cv2.putText(cam, f"t={k}  pos=({pose[0]:.1f},{pose[1]:+.1f})m "
                     f"hdg={np.rad2deg(pose[2]):+.0f}deg", (10, 28), 0, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cmap = render(r["grid"], cfg, 4, path=r["path"], goal=r["goal"],
                  aim=r["aim"], cmd=(r["v"], r["omega"]), reached=r["reached"])
    pad = np.zeros((360, cmap.shape[1], 3), np.uint8)
    hh = min(360, cmap.shape[0])
    pad[:hh] = cmap[:hh]
    return np.hstack([cam, pad])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="score the costmap and planner against known geometry")
    ap.add_argument("--demo", default=None, help="write a closed-loop drive to mp4")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--goal", default="9.5,0", help="world goal as X,Y in metres, or 'carrot' for the robot-frame 8 m goal")
    ap.add_argument("--pitch", type=float, default=12.0)
    ap.add_argument("--height", type=float, default=0.60)
    a = ap.parse_args()

    if a.validate:
        raise SystemExit(validate())

    cfg = Cfg()
    cfg.cam_pitch = np.deg2rad(a.pitch)
    cfg.cam_height = a.height
    world = World.default()

    writer = None
    if a.demo:
        writer = cv2.VideoWriter(a.demo, cv2.VideoWriter_fourcc(*"mp4v"),
                                 15.0, (640 + 80 * 4, 360))
    gw = None
    if a.goal != "carrot":
        gx, gy = (float(t) for t in a.goal.split(","))
        gw = (gx, gy)
        print(f"world goal ({gx:.1f}, {gy:+.1f}) m")
    try:
        res = drive(world, cfg, steps=a.steps, writer=writer, show=not a.demo,
                    goal_world=gw)
        print(f"\narrived={res['arrived']} in {res['steps']} steps; "
              f"travelled {res['dist']:.1f} m, {res['stalls']} stalled steps, "
              f"{res['collisions']} collisions, final pos "
              f"({res['x']:.2f}, {res['y']:+.2f}) hdg {np.rad2deg(res['th']):+.1f} deg")
        if a.demo:
            print(f"wrote {a.demo}")
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
