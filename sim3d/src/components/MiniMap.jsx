import { useEffect, useRef, useState } from "react";
import { link, useNav } from "../nav/link";
import { toNavWorld, toThree } from "../nav/frames";
import { TRAIL_HALF } from "../nav/world";

// Top-down course map (three.js x right, -z up). Small by default, click ▢ to
// expand. Click anywhere on it to flag the destination. Draws the course from its
// own ground-truth description, the rover, the goal flag and the server's global path.
const X0 = -30, X1 = 30, Z_TOP = -138, Z_BOT = 10;   // world extent shown

export default function MiniMap({ course, telemetry }) {
  const [big, setBig] = useState(false);
  const ref = useRef(null);
  const { nav, goal } = useNav();
  // Drawing resolution; the ON-SCREEN size is set by CSS below from the viewport
  // height (the course is portrait, 60 x 148 m, so height is what runs out first).
  const W = big ? 480 : 240, H = Math.round(W * (Z_BOT - Z_TOP) / (X1 - X0));

  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d");
    const sx = W / (X1 - X0), sz = H / (Z_BOT - Z_TOP);
    const px = (x, z) => [(x - X0) * sx, (z - Z_TOP) * sz];
    const disc = (o, col, r = o.r) => { const [x, y] = px(o.x, o.z); ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, Math.max(1.5, r * sx), 0, 6.283); ctx.fill(); };
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#3a6b30"; ctx.fillRect(0, 0, W, H);                         // meadow
    // trail
    ctx.strokeStyle = "#9a7b55"; ctx.lineWidth = 2 * TRAIL_HALF * sx; ctx.lineCap = "round"; ctx.beginPath();
    course.trail.forEach((p, i) => { const [x, y] = px(p.x, p.z); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    course.sand.forEach((o) => disc(o, "#c9b47a"));
    course.mud.forEach((o) => disc(o, "#5e4127"));
    course.ponds.forEach((o) => disc(o, "#2f6fae", o.R * 0.85));
    for (const d of course.ditches) {                                              // rotated rectangles
      const [x, y] = px(d.x, d.z);
      ctx.save(); ctx.translate(x, y); ctx.rotate(d.yaw);
      ctx.fillStyle = "#1b1410"; ctx.fillRect(-d.len / 2 * sx, -d.w / 2 * sz, d.len * sx, d.w * sz);
      ctx.restore();
    }
    for (const l of course.logs) {
      const c_ = Math.cos(l.yaw), s_ = Math.sin(l.yaw);
      const a = px(l.x - c_ * l.len / 2, l.z - s_ * l.len / 2), b = px(l.x + c_ * l.len / 2, l.z + s_ * l.len / 2);
      ctx.strokeStyle = "#7a5a34"; ctx.lineWidth = Math.max(3, 2 * l.r * sx); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    }
    course.scatter.forEach((o) => disc(o, "#8d877e"));
    course.rocks.forEach((o) => disc(o, "#a19b92"));
    course.bushes.forEach((o) => disc(o, "#2c7a2e"));
    course.trees.forEach((o) => disc({ x: o.x, z: o.z }, "#17491b", 1.3 * o.s));
    course.bgTrees.forEach((o) => disc({ x: o.x, z: o.z }, "#1f4f22", 1.1 * o.s));
    course.poles.forEach((o) => disc(o, "#dddddd", 0.35));
    course.drops.forEach((o) => disc(o, "#d97706", 0.6));
    course.movers.forEach((m) => disc({ x: m.pos.x, z: m.pos.z }, "#f2a516", 0.8));
    for (const p of course.pads) {                                                 // A / B
      const [x, y] = px(p.x, p.z);
      ctx.strokeStyle = "#f5f5f0"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, p.r * sx, 0, 6.283); ctx.stroke();
      ctx.fillStyle = "#f5f5f0"; ctx.font = `bold ${Math.round(11 * W / 240)}px sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(p.id, x, y);
    }
    // server's global path
    if (nav && nav.global && nav.global.path_world && nav.global.path_world.length > 1) {
      ctx.strokeStyle = "#facc15"; ctx.lineWidth = 2; ctx.lineCap = "butt"; ctx.beginPath();
      nav.global.path_world.forEach(([gx, gy], i) => { const t = toThree(gx, gy); const [x, y] = px(t.x, t.z); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }
    // goal flag
    if (goal) {
      const t = toThree(goal.x, goal.y); const [x, y] = px(t.x, t.z);
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y - 14); ctx.stroke();
      ctx.fillStyle = nav && nav.status === "ARRIVED" ? "#22c55e" : "#f59e0b"; ctx.beginPath(); ctx.moveTo(x, y - 14); ctx.lineTo(x + 10, y - 10); ctx.lineTo(x, y - 6); ctx.fill();
    }
    // rover
    const rx = parseFloat(telemetry.x), rz = parseFloat(telemetry.z), hd = parseFloat(telemetry.heading) * Math.PI / 180;
    if (!isNaN(rx)) {
      const [x, y] = px(rx, rz);
      ctx.fillStyle = "#38bdf8"; ctx.beginPath(); ctx.arc(x, y, 5, 0, 6.283); ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - Math.sin(hd) * 12, y - Math.cos(hd) * 12); ctx.stroke();
    }
  }, [course, telemetry, nav, goal, W, H]);

  const onClick = (ev) => {
    const r = ev.currentTarget.getBoundingClientRect();
    const x = X0 + (ev.clientX - r.left) / r.width * (X1 - X0);
    const z = Z_TOP + (ev.clientY - r.top) / r.height * (Z_BOT - Z_TOP);
    const n = toNavWorld(x, z, 0);
    link.setGoalNav(n.x, n.y);
  };

  return (
    <div style={{ ...(big
        ? { position: "fixed", top: "50%", left: "50%", transform: "translate(-50%, -50%)", zIndex: 30 }
        : { position: "relative", flex: "0 0 auto", alignSelf: "flex-start" }),
      pointerEvents: "auto", background: "rgba(10,12,18,0.82)", padding: 6, borderRadius: 8,
      fontFamily: "ui-monospace, Menlo, monospace", fontSize: 11, color: "#cbd5e1" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4, gap: 8 }}>
        <span>course map · click = goal</span>
        <button onClick={() => setBig((b) => !b)} style={{ background: "#1f2937", color: "#fff", border: "1px solid #374151", borderRadius: 4, padding: "1px 8px", cursor: "pointer" }}>{big ? "▣ shrink" : "▢ expand"}</button>
      </div>
      <canvas ref={ref} width={W} height={H} onClick={onClick}
        style={{ display: "block", margin: "0 auto", height: big ? "calc(100vh - 80px)" : "min(36vh, 380px)", width: "auto",
                 aspectRatio: `${W} / ${H}`, cursor: "crosshair", borderRadius: 4 }} />
    </div>
  );
}
