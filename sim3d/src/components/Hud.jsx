import { useState } from "react";
import { NAV } from "../nav/config";
import { link, useNav } from "../nav/link";
import { toThree } from "../nav/frames";

const box = {
  position: "absolute", color: "#fff", zIndex: 10, fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12,
  background: "rgba(10,12,18,0.72)", padding: "10px 12px", borderRadius: 8, backdropFilter: "blur(4px)",
};
const btn = (bg = "#4a7c59") => ({ background: bg, color: "#fff", border: "none", padding: "6px 10px", borderRadius: 4, cursor: "pointer", fontWeight: 600, fontSize: 12 });
const statusColor = { DRIVING: "#4ade80", ARRIVED: "#38bdf8", BLOCKED: "#f87171", STOPPED: "#f87171", ERROR: "#f87171", TURNING: "#f59e0b", PLANNING: "#f59e0b", NO_GOAL: "#9ca3af" };

export default function Hud({ telemetry, cameraMode, setCameraMode, auto, setAuto, autoRef, collisions, pipRef, course }) {
  const st = useNav();
  const nav = st.nav;
  const [gx, setGx] = useState(String(course.goalPreset.navX));
  const [gy, setGy] = useState(String(course.goalPreset.navY));

  const toggleAuto = () => { const n = !auto; autoRef.current = n; setAuto(n); link.setMode(n); };
  const setGoal = () => { const x = parseFloat(gx), y = parseFloat(gy); if (!isNaN(x) && !isNaN(y)) link.setGoalNav(x, y); };
  const clickGlobal = (ev) => {
    if (!nav || !nav.global || !nav.global.meta) return;
    const img = ev.currentTarget, r = img.getBoundingClientRect(), mt = nav.global.meta;
    const px = (ev.clientX - r.left) * img.naturalWidth / r.width, py = (ev.clientY - r.top) * img.naturalHeight / r.height;
    const wx = mt.origin_x + (px / mt.scale) * mt.res, wy = mt.origin_y + ((img.naturalHeight - py) / mt.scale) * mt.res;
    setGx(wx.toFixed(1)); setGy(wy.toFixed(1)); link.setGoalNav(wx, wy);
  };
  const plane = nav && nav.plane;
  const gThree = st.goal ? toThree(st.goal.x, st.goal.y) : null;

  return (
    <>
      {/* ---- left: controls + telemetry ------------------------------------------ */}
      <div style={{ ...box, top: 16, left: 16, width: 300 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
          <strong>SLAM3D · autonomous rover</strong>
          <span style={{ color: st.connected ? "#4ade80" : "#f87171" }}>● {st.connected ? "perception" : "no server"}</span>
        </div>
        <div style={{ fontSize: 22, fontWeight: 700, color: statusColor[nav ? nav.status : "NO_GOAL"] || "#fff" }}>
          {nav ? nav.status : (st.connected ? "waiting for frames" : "offline")}
        </div>
        {nav && nav.note && <div style={{ color: "#9ca3af" }}>{nav.note}</div>}
        <div style={{ margin: "6px 0" }}>
          <button style={btn(auto ? "#b45309" : "#1d4ed8")} onClick={toggleAuto}>{auto ? "AUTO ✓  (T → manual)" : "MANUAL  (T → auto)"}</button>
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse" }}><tbody>
          <tr><td style={td}>command</td><td>{nav && nav.cmd ? (nav.cmd.v > 0 ? `v ${nav.cmd.v.toFixed(2)} m/s  ω ${nav.cmd.omega >= 0 ? "+" : ""}${nav.cmd.omega.toFixed(2)}` : `STOP${nav.cmd.omega ? `  ω ${nav.cmd.omega.toFixed(2)}` : ""}`) : "–"}</td></tr>
          <tr><td style={td}>goal (nav x,y)</td><td>{st.goal ? `${st.goal.x.toFixed(1)}, ${st.goal.y.toFixed(1)}  ·  ${nav && nav.dist_to_goal != null ? nav.dist_to_goal.toFixed(1) + " m away" : ""}` : "click the ground / map"}</td></tr>
          <tr><td style={td}>speed</td><td>{telemetry.speed} m/s · accel {telemetry.acceleration} m/s²</td></tr>
          <tr><td style={td}>pose (three)</td><td>x {telemetry.x} z {telemetry.z} hdg {telemetry.heading}°</td></tr>
          <tr><td style={td}>perception</td><td>{nav ? `${nav.fps ?? "–"} fps · depth ${nav.depth_mode}` : "–"}{st.sentFrames ? ` · sent ${st.sentFrames}` : ""}</td></tr>
          <tr><td style={td}>collisions</td><td style={{ color: collisions.count ? "#f87171" : "#4ade80" }}>{collisions.count} {collisions.last ? `(last: ${collisions.last})` : ""}</td></tr>
        </tbody></table>
        {plane && (
          <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px solid rgba(255,255,255,0.15)" }}>
            <div style={{ color: "#9ca3af" }}>ground plane · measured per frame ({plane.source})</div>
            <div>h <b>{plane.ok ? plane.height.toFixed(3) : "–"} m</b> · pitch <b>{plane.ok ? plane.pitch_deg.toFixed(1) : "–"}°</b> · roll <b>{plane.ok ? plane.roll_deg.toFixed(1) : "–"}°</b> · conf {(plane.confidence * 100).toFixed(0)}%</div>
            {plane.mount_err && <div style={{ color: "#9ca3af" }}>vs mount ({NAV.mount.height} m, {NAV.mount.pitchDeg}°): Δh {(plane.mount_err.height * 100).toFixed(1)} cm · Δpitch {plane.mount_err.pitch_deg.toFixed(2)}°</div>}
          </div>
        )}
        {nav && nav.warnings && nav.warnings.length > 0 && <div style={{ color: "#f59e0b", marginTop: 4 }}>{nav.warnings.join(" · ")}</div>}
        <div style={{ marginTop: 8, display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ color: "#9ca3af" }}>goal</span>
          <input value={gx} onChange={(e) => setGx(e.target.value)} style={inp} />
          <input value={gy} onChange={(e) => setGy(e.target.value)} style={inp} />
          <button style={btn("#1d4ed8")} onClick={setGoal}>Set</button>
          <button style={btn("#374151")} onClick={() => link.clearGoal()}>Clear</button>
          <button style={btn("#374151")} onClick={() => link.resetMap()}>Reset map</button>
        </div>
        <div style={{ marginTop: 8, display: "flex", gap: 6, alignItems: "center" }}>
          <button style={btn()} onClick={() => setCameraMode((p) => (p + 1) % 3)}>Camera {cameraMode + 1}/3 (C)</button>
          <label style={{ color: "#9ca3af" }}>depth&nbsp;
            <select value={nav ? nav.depth_mode : "sim"} onChange={(e) => link.setDepth(e.target.value)} style={inp}>
              {(st.config ? st.config.depth_modes : ["sim", "metric", "relative"]).map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </label>
        </div>
        <div style={{ color: "#9ca3af", marginTop: 6 }}>WASD drive (takes over) · Space brake · T auto · click ground = goal · seed {course.seed}{gThree ? ` · goal three (${gThree.x.toFixed(0)}, ${gThree.z.toFixed(0)})` : ""}</div>
      </div>

      {/* ---- right: what the robot sees + maps ---------------------------------- */}
      <div style={{ ...box, top: 16, right: 16, width: 336, padding: 8 }}>
        <div style={lbl}>robot camera (streamed)</div>
        <canvas ref={pipRef} width={320} height={180} style={{ width: 320, height: 180, display: "block", background: "#000", borderRadius: 4 }} />
        <div style={{ ...lbl, marginTop: 6 }}>local costmap · forward = up</div>
        {nav && nav.images && nav.images.costmap ? <img src={nav.images.costmap} alt="costmap" style={{ width: 320, display: "block", borderRadius: 4 }} /> : <div style={ph}>waiting…</div>}
        <div style={{ ...lbl, marginTop: 6 }}>global costmap · north up · click to set goal</div>
        {nav && nav.images && nav.images.global ? <img src={nav.images.global} alt="global" onClick={clickGlobal} style={{ width: 320, display: "block", borderRadius: 4, cursor: "crosshair" }} /> : <div style={ph}>waiting…</div>}
        {nav && nav.images && nav.images.depth && <img src={nav.images.depth} alt="depth" style={{ width: 320, display: "block", borderRadius: 4, marginTop: 6, opacity: 0.9 }} />}
      </div>
    </>
  );
}

const td = { color: "#9ca3af", paddingRight: 8, whiteSpace: "nowrap" };
const inp = { width: 64, background: "#0f1115", color: "#fff", border: "1px solid #374151", borderRadius: 4, padding: "3px 6px", fontFamily: "inherit", fontSize: 12 };
const lbl = { color: "#9ca3af", fontSize: 11, marginBottom: 3 };
const ph = { width: 320, height: 120, background: "#111", borderRadius: 4, color: "#555", display: "flex", alignItems: "center", justifyContent: "center" };
