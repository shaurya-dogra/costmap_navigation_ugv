"""
synth_scene.py - analytic ray-cast scenes for testing the perception front-end.

A camera at a chosen height / pitch / roll looks at flat ground with optional
boxes ("rocks", labelled 100 so only geometry can catch them), a step-down
(labelled road, only geometry can catch it) and a water band (flat, only
semantics can catch it). Returns depth along the optical axis (0 = sky) and a
per-pixel semantic cost. Used by test_perception_core.py and by the fake
simulator client that exercises perception_server.py without a browser.
"""
import numpy as np
import perception_core as pc

def synth(cfg, height, pitch, roll, boxes=(), drop=None, sem_ground=0.0, water=None, trench=None):
    """
    Ray-cast an analytic scene from a camera at `height` with `pitch`/`roll`.
    boxes : (x0, x1, y0, y1, z0, z1) in the ROBOT frame, labelled 100 (rock)
    drop  : (x_edge, depth) ground steps down beyond x_edge, still labelled road
    water : (x0, x1) band of ground labelled water (254) - flat, only semantics
    Returns depth (H, W) along the optical axis (0 = sky) and semantic cost.
    """
    u, v = np.meshgrid(np.arange(cfg.w, dtype=np.float64), np.arange(cfg.h, dtype=np.float64))
    xn = (u - cfg.cx) / cfg.fx
    yn = (v - cfg.cy) / cfg.fy
    R = pc.rotation_from_normal(pc.normal_from_angles(pitch, roll))
    dx = R[0, 0] * xn + R[0, 1] * yn + R[0, 2]
    dy = R[1, 0] * xn + R[1, 1] * yn + R[1, 2]
    dz = R[2, 0] * xn + R[2, 1] * yn + R[2, 2]
    INF = 1e9
    t_best = np.full(xn.shape, INF)
    sem = np.full(xn.shape, -1.0)

    def commit(t, mask, cost):
        nonlocal t_best, sem
        take = mask & (t > 0.05) & (t < t_best)
        t_best = np.where(take, t, t_best)
        sem = np.where(take, cost, sem)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_g = np.where(dz < -1e-9, -height / dz, INF)
    Xg, Yg = t_g * dx, t_g * dy
    gmask = t_g < INF
    if drop is not None:
        xe, dd = drop
        gmask &= Xg < xe
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = np.where(dz < -1e-9, -(height + dd) / dz, INF)
        Xlo = t_lo * dx
        commit(t_lo, (t_lo < INF) & (Xlo >= xe), sem_ground)
    if trench is not None:
        x0, x1, dd = trench
        gmask &= ~((Xg >= x0) & (Xg <= x1))                  # the ground is missing here
        with np.errstate(divide="ignore", invalid="ignore"):
            t_f = np.where(dz < -1e-9, -(height + dd) / dz, INF)
        Xf = t_f * dx
        commit(t_f, (t_f < INF) & (Xf >= x0) & (Xf <= x1), sem_ground)   # floor, where visible
        # far wall: vertical plane at x = x1 from z=-dd to 0
        with np.errstate(divide="ignore", invalid="ignore"):
            t_w = np.where(dx > 1e-9, x1 / dx, INF)
        Zw = height + t_w * dz
        commit(t_w, (t_w < INF) & (Zw <= 0.0) & (Zw >= -dd), sem_ground)
    commit(t_g, gmask, sem_ground)
    if water is not None:
        x0, x1 = water
        wm = gmask & (Xg >= x0) & (Xg <= x1)
        sem = np.where(wm & (t_best == t_g), 254.0, sem)
    for (x0, x1, y0, y1, z0, z1) in boxes:
        o = np.array([0.0, 0.0, height])
        t_lo = np.full(xn.shape, -INF); t_hi = np.full(xn.shape, INF)
        for d_, lo, hi in ((dx, x0, x1), (dy, y0, y1), (dz, z0, z1)):
            oi = o[[dx, dy, dz].index(d_) if False else 0]  # placeholder, replaced below
        # slab method
        for axis, (d_, lo, hi) in enumerate(((dx, x0, x1), (dy, y0, y1), (dz, z0, z1))):
            oi = o[axis]
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (lo - oi) / d_
                t2 = (hi - oi) / d_
            tmin = np.minimum(t1, t2); tmax = np.maximum(t1, t2)
            par = np.abs(d_) < 1e-12
            inside = (oi >= lo) & (oi <= hi)
            tmin = np.where(par, np.where(inside, -INF, INF), tmin)
            tmax = np.where(par, np.where(inside, INF, -INF), tmax)
            t_lo = np.maximum(t_lo, tmin); t_hi = np.minimum(t_hi, tmax)
        hit = (t_hi >= t_lo) & (t_lo > 0.05)
        commit(np.where(hit, t_lo, INF), hit, 100.0)
    depth = np.where(t_best < INF, t_best, 0.0).astype(np.float32)
    return depth, sem.astype(np.float32)


