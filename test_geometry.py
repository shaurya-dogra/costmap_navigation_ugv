"""
Synthetic-scene verification of the geometry + costmap stages.
Needs only numpy + opencv. No models, no camera, no GPU.

    python3 test_geometry.py

Every line must say PASS before you trust a live run.
"""
import numpy as np, importlib.util, sys, types

# stub torch so the module imports without the DL stack installed
_t = types.ModuleType("torch")
_t.cuda = types.SimpleNamespace(is_available=lambda: False)
_t.backends = types.SimpleNamespace(mps=None)
_t.inference_mode = lambda: (lambda f: f)
sys.modules.setdefault("torch", _t)

spec = importlib.util.spec_from_file_location("cp", "costmap_prototype.py")
cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)

# ---------------------------------------------------------------- fixtures --
cfg = cp.Cfg(); cfg.fx = cfg.fy = 470.0; cfg.cx, cfg.cy = 320.0, 180.0
H, W = 360, 640
NB = np.zeros((0, 4))

u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
c, s = np.cos(cfg.cam_pitch), np.sin(cfg.cam_pitch)
dz = -np.ones_like(u) * s + (-(v - cfg.cy) / cfg.fy) * c
depth = np.where(dz < -1e-4, cfg.cam_height / (-dz), 0.0)     # exact flat ground
depth = np.where((depth > 0.3) & (depth < 30), depth, 0.0)
sem = np.zeros((H, W), np.float32)                            # all "road", cost 0
X0, Y0, Z0, raw0 = cp.backproject(depth, cfg)

FAILS = []
def check(name, got, ok):
    tag = "PASS" if ok else "FAIL"
    if not ok: FAILS.append(name)
    print(f"  {tag}  {name:<46} {got}")

def build(d, sm=sem, bx=NB):
    X, Y, Z, _ = cp.backproject(d, cfg)
    cm = cp.Costmap(cfg)
    return cm, cm.build(X, Y, Z, sm, bx, d), Z

def lethal_centroid(mask, factor=0.45):
    """Raise terrain inside mask, return where the LETHAL cells land, in metres."""
    d = depth.copy(); d[mask] *= factor
    cm, g, _ = build(d)
    i = np.where(g == cfg.LETHAL)
    if len(i[0]) == 0: return None, None, 0
    return (cfg.x_min + i[0].mean() * cfg.res,
            cfg.y_min + i[1].mean() * cfg.res, len(i[0]))

# -------------------------------------------------------------------- tests --
print("\n1. BACK-PROJECTION")
cm, g0, _ = build(depth)
rms = float(np.sqrt(np.mean(Z0[depth > 0] ** 2)))
check("flat ground lands at Z = 0", f"{rms:.5f} m rms", rms < 0.01)
check("flat ground costs nothing", int(g0[g0 != cfg.UNKNOWN].max()),
      g0[g0 != cfg.UNKNOWN].max() == 0)
check("grid is (forward, lateral)", f"{g0.shape} nx={cm.nx} ny={cm.ny}",
      g0.shape == (cm.nx, cm.ny))

print("\n2. MONOCULAR SCALE RECOVERY")
sc = cp.recover_scale(cp.backproject(depth * 1.4, cfg)[3], sem, cfg)
check("recovers 1/1.4 from camera height", f"{sc:.3f} (want 0.714)", abs(sc - 1 / 1.4) < 0.05)
sc2 = cp.recover_scale(cp.backproject(depth * 0.7, cfg)[3], sem, cfg)
check("recovers 1/0.7 from camera height", f"{sc2:.3f} (want 1.429)", abs(sc2 - 1 / 0.7) < 0.08)

print("\n3. NEGATIVE OBSTACLE (the ditch case)")
dm = (X0 > 3) & (X0 < 4) & (np.abs(Y0) < 1) & (depth > 0)
d2 = depth.copy(); d2[dm] *= 1.35
_, g2, Z2 = build(d2)
check("ground drops below the ditch threshold", f"{Z2[dm].min():.2f} m", Z2[dm].min() < cfg.ditch_h)
check("ditch produces lethal cells", int((g2 == cfg.LETHAL).sum()), (g2 == cfg.LETHAL).sum() > 0)

print("\n4. ORIENTATION (obstacle lands where it actually is)")
f, l, n = lethal_centroid((Y0 > 1.0) & (X0 > 2) & (X0 < 5) & (depth > 0))
check("obstacle on the LEFT  -> +Y", f"X={f:.2f} Y={l:+.2f} n={n}", l > 0.3)
f, l, n = lethal_centroid((Y0 < -1.0) & (X0 > 2) & (X0 < 5) & (depth > 0))
check("obstacle on the RIGHT -> -Y", f"X={f:.2f} Y={l:+.2f} n={n}", l < -0.3)
f, l, n = lethal_centroid((X0 > 3.5) & (X0 < 4.5) & (np.abs(Y0) < 0.6) & (depth > 0))
check("obstacle AHEAD       -> Y~0", f"X={f:.2f} Y={l:+.2f} n={n}", abs(l) < 0.3)

print("\n5. RENDER ORIENTATION (what you see matches what is there)")
d = depth.copy(); d[(Y0 > 1.0) & (X0 > 2) & (X0 < 5) & (depth > 0)] *= 0.45
_, gl, _ = build(d)
img = cp.render(gl, cfg, 4); w_ = img.shape[1]
mag = img[:, :, 2].astype(int) - img[:, :, 1].astype(int)
L, R = mag[:, :w_ // 2].sum(), mag[:, w_ // 2:].sum()
check("left obstacle draws on screen LEFT", f"L={L} R={R}", L > R)
d = depth.copy(); d[(X0 > 2.0) & (X0 < 2.6) & (np.abs(Y0) < 0.6) & (depth > 0)] *= 0.45
_, ga, _ = build(d)
img = cp.render(ga, cfg, 4); w_ = img.shape[1]
mag = img[:, :, 2].astype(int) - img[:, :, 1].astype(int)
L, R = mag[:, :w_ // 2].sum(), mag[:, w_ // 2:].sum()
sym = abs(L - R) / max(abs(L), abs(R), 1)
check("obstacle ahead draws symmetric", f"asym {sym*100:.2f}%", sym < 0.05)
check("render output is uint8 BGR", f"{img.dtype} {img.shape}", img.dtype == np.uint8)

print("\n6. FAIL-SAFE BEHAVIOUR")
_, g5, _ = build(np.zeros_like(depth))
check("no depth -> everything UNKNOWN, never free", bool((g5 == cfg.UNKNOWN).all()),
      (g5 == cfg.UNKNOWN).all())
_, g6, _ = build(depth, np.full_like(sem, 254.0))
seen6 = g6 != cfg.UNKNOWN
check("all-water semantics -> all lethal", bool((g6[seen6] >= 250).all()),
      (g6[seen6] >= 250).all())
_, g7, _ = build(depth, sem, np.array([[200, 100, 400, 355]], np.float32))
check("a detection stamps lethal cells", int((g7 == cfg.LETHAL).sum()),
      (g7 == cfg.LETHAL).sum() > 0)

print("\n7. THROUGHPUT (costmap stage only, CPU)")
import time
big = np.random.uniform(1, 10, (720, 1280)).astype(np.float32)
bsem = np.random.choice([0, 40, 254], (720, 1280)).astype(np.float32)
cmb = cp.Costmap(cfg)
X, Y, Z, raw = cp.backproject(big, cfg); cmb.build(X, Y, Z, bsem, NB, big)
t = time.time(); N = 10
for _ in range(N):
    X, Y, Z, raw = cp.backproject(big, cfg)
    cp.recover_scale(raw, bsem, cfg)
    cmb.build(X, Y, Z, bsem, NB, big)
ms = (time.time() - t) / N * 1000
check("1280x720 costmap stage under 150 ms", f"{ms:.1f} ms/frame ({1000/ms:.0f} Hz)", ms < 150)

print("\n8. CONFIGURATION & CLI OPTIONS")
default_cfg = cp.Cfg()
check("default fx/fy are 940.0", f"fx={default_cfg.fx} fy={default_cfg.fy}",
      default_cfg.fx == 940.0 and default_cfg.fy == 940.0)
check("default cx/cy are 640/360", f"cx={default_cfg.cx} cy={default_cfg.cy}",
      default_cfg.cx == 640.0 and default_cfg.cy == 360.0)

# ----------------------------------------------------------------------- plan --
print("\n9. A* PLANNER OVER THE COSTMAP")
NX, NY = cm.nx, cm.ny
FREE = np.zeros((NX, NY), np.uint8)
CTR  = int(round((0.0 - cfg.y_min) / cfg.res))     # grid column of Y = 0

def wall(row, thick=4, gap=None):
    """Lethal band across the grid at `row`, optional free gap columns (a, b)."""
    g = np.zeros((NX, NY), np.uint8)
    g[row:row + thick, :] = cfg.LETHAL
    if gap:
        g[row:row + thick, gap[0]:gap[1]] = 0
    return g

check("start cell is dead centre, nearest row", f"{cp.start_cell(cfg, NX, NY)}",
      cp.start_cell(cfg, NX, NY) == (0, CTR))
gc = cp.goal_cell(cfg, NX, NY)
check("goal cell is 8 m ahead on the centre line",
      f"{gc} = X {cfg.x_min + gc[0]*cfg.res:.1f} m Y {cfg.y_min + gc[1]*cfg.res:+.1f} m",
      abs((cfg.x_min + gc[0] * cfg.res) - cfg.goal_x) < cfg.res
      and abs(cfg.y_min + gc[1] * cfg.res) < cfg.res)

p, reached = cp.astar(FREE, cfg)
check("free grid reaches the goal", f"reached={reached} len={len(p)}",
      reached and p[-1] == gc)
check("free grid path runs dead straight",
      f"max |Y| = {max(abs(iy - CTR) for _, iy in p) * cfg.res:.2f} m",
      all(iy == CTR for _, iy in p))
check("path starts at the start cell", f"{p[0]}", p[0] == cp.start_cell(cfg, NX, NY))
check("path is 8-connected (no jumps)",
      f"max step {max(max(abs(b[0]-a[0]), abs(b[1]-a[1])) for a, b in zip(p, p[1:]))} cell",
      all(max(abs(b[0]-a[0]), abs(b[1]-a[1])) == 1 for a, b in zip(p, p[1:])))

# --- the UNKNOWN contract: expensive, but never blocked and never free -------
gu = np.full((NX, NY), cfg.UNKNOWN, np.uint8)
pu, ru = cp.astar(gu, cfg)
check("all-UNKNOWN grid still plans (not blocked)", f"reached={ru} len={len(pu)}",
      ru and len(pu) > 0)
gband = np.zeros((NX, NY), np.uint8)
gband[40:60, :] = cfg.UNKNOWN
gband[40:60, 55:60] = 0                            # free detour to the left
pb, rb = cp.astar(gband, cfg)
n_unk = sum(1 for ix, iy in pb if gband[ix, iy] == cfg.UNKNOWN)
check("detours around UNKNOWN when free ground exists (not free)",
      f"{n_unk} UNKNOWN cells crossed, via cols {sorted({iy for ix, iy in pb if 40 <= ix < 60})}",
      rb and n_unk == 0)

# --- lethal is hard blocked -------------------------------------------------
pg, rg = cp.astar(wall(40, gap=(55, 62)), cfg)
gapw = wall(40, gap=(55, 62))
check("routes through a gap in a wall",
      f"reached={rg} via cols {sorted({iy for ix, iy in pg if 40 <= ix < 44})}",
      rg and all(55 <= iy < 62 for ix, iy in pg if 40 <= ix < 44))
check("path never enters a LETHAL cell",
      f"{sum(1 for ix, iy in pg if gapw[ix, iy] == cfg.LETHAL)} lethal cells on path",
      not any(gapw[ix, iy] == cfg.LETHAL for ix, iy in pg))

solid = wall(40)
pw, rw = cp.astar(solid, cfg)
check("sealed wall -> reached=False, partial plan returned",
      f"reached={rw} len={len(pw)} ends at ix={pw[-1][0] if pw else None}",
      rw is False and len(pw) > 0 and pw[-1][0] < 40)

blocked_start = np.zeros((NX, NY), np.uint8)
blocked_start[0, CTR] = cfg.LETHAL
pz, rz = cp.astar(blocked_start, cfg)
check("lethal dead ahead -> no path at all", f"reached={rz} len={len(pz)}",
      rz is False and len(pz) == 0)

# ------------------------------------------------------------- drive command --
print("\n10. PURE-PURSUIT DRIVE COMMAND")
v, w, aim = cp.drive_command(cp.astar(FREE, cfg)[0], cfg)
check("clear path -> full speed, no steering", f"v={v:.2f} m/s w={w:+.3f} rad/s",
      abs(v - cfg.v_max) < 1e-6 and abs(w) < 1e-6)

v0, w0, aim0 = cp.drive_command([], cfg)
check("no path -> full stop", f"v={v0:.2f} w={w0:+.2f}", v0 == 0.0 and w0 == 0.0)

gl = np.zeros((NX, NY), np.uint8); gl[5:25, CTR:CTR + 22] = cfg.LETHAL
_, wl, _ = cp.drive_command(cp.astar(gl, cfg)[0], cfg)
check("obstacle on the LEFT  -> steers RIGHT (w < 0)", f"w={wl:+.3f} rad/s", wl < -0.01)
gr = np.zeros((NX, NY), np.uint8); gr[5:25, CTR - 22:CTR + 1] = cfg.LETHAL
_, wr, _ = cp.drive_command(cp.astar(gr, cfg)[0], cfg)
check("obstacle on the RIGHT -> steers LEFT  (w > 0)", f"w={wr:+.3f} rad/s", wr > 0.01)

speeds = []
for row in (3, 8, 15, 40):
    pv, _ = cp.astar(wall(row), cfg)
    speeds.append(cp.drive_command(pv, cfg)[0])
check("speed de-rates as the plan shortens",
      " -> ".join(f"{s:.2f}" for s in speeds) + " m/s at 0.8/1.3/2.0/4.5 m",
      speeds == sorted(speeds) and speeds[0] < cfg.v_max and speeds[-1] == cfg.v_max)

near = np.zeros((NX, NY), np.uint8); near[1:, :] = cfg.LETHAL
vn, wn, _ = cp.drive_command(cp.astar(near, cfg)[0], cfg)
check("plan shorter than stop_dist -> stop", f"v={vn:.2f} w={wn:+.2f}",
      vn == 0.0 and wn == 0.0)

check("omega never exceeds w_max",
      f"|w| max {max(abs(wl), abs(wr)):.2f} <= {cfg.w_max}",
      max(abs(wl), abs(wr)) <= cfg.w_max + 1e-9)

# ------------------------------------------------------------------- drawing --
print("\n11. PLAN RENDERING")
pp, rr = cp.astar(gl, cfg)
_, _, aimp = cp.drive_command(pp, cfg)
plain = cp.render(gl, cfg, 4)
drawn = cp.render(gl, cfg, 4, path=pp, goal=gc, aim=aimp, cmd=(0.8, -0.1), reached=rr)
check("render(path=...) draws over the bare costmap",
      f"{int((plain != drawn).any(axis=2).sum())} px changed",
      (plain != drawn).any() and drawn.shape == plain.shape)
check("render without a path is unchanged (tests stay valid)",
      f"{plain.dtype} {plain.shape}",
      (plain == cp.render(gl, cfg, 4)).all())
sx_px = cp.cell_to_px(0, CTR, NX, NY, 4)
check("cell_to_px puts the robot bottom-centre",
      f"{sx_px} in a {drawn.shape[1]}x{drawn.shape[0]} image",
      abs(sx_px[0] - drawn.shape[1] // 2) <= 4 and sx_px[1] >= drawn.shape[0] - 4)

print("\n12. PLANNER THROUGHPUT")
import time as _t2
rngp = np.random.default_rng(0)
worst = np.full((NX, NY), cfg.UNKNOWN, np.uint8)     # forces full expansion
cp.astar(worst, cfg)
t = _t2.time(); N = 20
for _ in range(N):
    cp.astar(worst, cfg)
ms = (_t2.time() - t) / N * 1000
check("worst-case A* under 40 ms", f"{ms:.1f} ms/frame ({1000/ms:.0f} Hz)", ms < 40)

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

