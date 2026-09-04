import { useEffect, useRef, useState } from "react";
import { link, useNav } from "../nav/link";
import { toNavWorld, toThree } from "../nav/frames";

// Top-down course map (three.js x right, -z up). Small by default, click ▢ to
// expand. Click anywhere on it to flag the destination. Draws the known course
// layout, the rover, the goal flag and the server's global path.
const X0 = -28, X1 = 28, Z_TOP = -112, Z_BOT = 12;   // world extent shown

export default function MiniMap({ course, telemetry }) {
  const [big, setBig] = useState(false);
  const ref = useRef(null);
  const { nav, goal } = useNav();
  const W = big ? 560 : 240, H = Math.round(W * (Z_BOT - Z_TOP) / (X1 - X0));

  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d");
    const sx = W / (X1 - X0), sz = H / (Z_BOT - Z_TOP);
    const px = (x, z) => [(x - X0) * sx, (z - Z_TOP) * sz];
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#2f5a2b"; ctx.fillRect(0, 0, W, H);                       // grass
    ctx.fillStyle = "#4b4b4b"; ctx.fillRect(px(-6, 0)[0], 0, 12 * sx, H);       // road
    ctx.strokeStyle = "#c9b23a"; ctx.setLineDash([6, 6]); ctx.beginPath(); ctx.moveTo(px(0, 0)[0], 0); ctx.lineTo(px(0, 0)[0], H); ctx.stroke(); ctx.setLineDash([]);
    const disc = (o, col) => { const [x, y] = px(o.x, o.z); ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, Math.max(1.5, o.r * sx), 0, 6.283); ctx.fill(); };
    course.sand.forEach((o) => disc(o, "#c9b47a"));
    course.mud.forEach((o) => disc(o, "#6b4a2b"));
    course.ponds.forEach((o) => disc(o, "#2f6fae"));
    disc({ ...course.mound }, "#3f7a35");
    for (const t of course.trenches) { const [x, y] = px(t.x0, t.z - t.halfW); ctx.fillStyle = "#1b1410"; ctx.fillRect(x, y, (t.x1 - t.x0) * sx, 2 * t.halfW * sz + 1); }
    for (const f of course.fences) { const [x, y] = px(f.x0, f.z); ctx.fillStyle = "#b08850"; ctx.fillRect(x, y - 1, (f.x1 - f.x0) * sx, 3); }
    for (const l of course.logs) { const c_ = Math.cos(l.yaw), s_ = Math.sin(l.yaw); const a = px(l.x - c_ * l.len / 2, l.z - s_ * l.len / 2), b = px(l.x + c_ * l.len / 2, l.z + s_ * l.len / 2); ctx.strokeStyle = "#7a5a34"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); }
    course.rubble.forEach((o) => disc(o, "#8d877e"));
    course.rocks.forEach((o) => disc(o, "#9a948b"));
    course.bushes.forEach((o) => disc({ ...o, r: o.r * 0.8 }, "#2c7a2e"));
    course.trees.forEach((o) => disc({ x: o.x, z: o.z, r: 1.6 * o.s }, "#1f5c22"));
    course.bgTrees.forEach((o) => disc({ x: o.x, z: o.z, r: 1.2 * o.s }, "#254f27"));
    course.poles.forEach((o) => disc({ ...o, r: 0.3 }, "#dddddd"));
    // server's global path
    if (nav && nav.global && nav.global.path_world && nav.global.path_world.length > 1) {
      ctx.strokeStyle = "#facc15"; ctx.lineWidth = 2; ctx.beginPath();
      nav.global.path_world.forEach(([gx, gy], i) => { const t = toThree(gx, gy); const [x, y] = px(t.x, t.z); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }
    // goal flag
    if (goal) {
      const t = toThree(goal.x, goal.y); const [x, y] = px(t.x, t.z);
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y - 14); ctx.stroke();
      ctx.fillStyle = nav && nav.status === "ARRIVED" ? "#22c55e" : "#f59e0b"; ctx.beginPath(); ctx.moveTo(x, y - 14); ctx.lineTo(x + 10, y - 10); ctx.lineTo(x, y - 6); ctx.fill();
      ctx.strokeStyle = "#f59e0b"; ctx.beginPath(); ctx.arc(x, y, 5, 0, 6.283); ctx.stroke();
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
    <div style={{ position: "absolute", bottom: 12, left: 16, zIndex: 11, background: "rgba(10,12,18,0.72)", padding: 6, borderRadius: 8, fontFamily: "ui-monospace, Menlo, monospace", fontSize: 11, color: "#cbd5e1" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span>course map · click to flag the destination</span>
        <button onClick={() => setBig((b) => !b)} style={{ background: "#1f2937", color: "#fff", border: "1px solid #374151", borderRadius: 4, padding: "1px 8px", cursor: "pointer" }}>{big ? "▣ shrink" : "▢ expand"}</button>
      </div>
      <canvas ref={ref} width={W} height={H} onClick={onClick} style={{ display: "block", width: W, height: H, cursor: "crosshair", borderRadius: 4 }} />
    </div>
  );
}
