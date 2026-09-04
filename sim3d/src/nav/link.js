// WebSocket link to perception_server.py (see ../../sih_prototype_costmap/PROTOCOL.md).
//
// One singleton `link` owns the socket; React components read a snapshot via
// useNav(). Frames go out with sendFrame() - at most one in flight, so a slow
// server is never flooded - and the latest drive command is read by the vehicle
// every physics step through link.command, which returns STOP when the last
// message is older than the watchdog.
import { useEffect, useState } from "react";
import { NAV } from "./config";

const WATCHDOG_MS = 1000;
const INFLIGHT_TIMEOUT_MS = 1500;

class NavLink {
  constructor() {
    this.ws = null;
    this.state = { connected: false, config: null, nav: null, goal: null, mode: NAV.autoAtStart ? "auto" : "manual", lastNavAt: 0, sentFrames: 0 };
    this.listeners = new Set();
    this.modeListeners = new Set();
    this.inFlight = false;
    this.sentAt = 0;
    this.pendingSeq = 0;
    this.retry = null;
  }

  connect() {
    if (this.ws && (this.ws.readyState === 0 || this.ws.readyState === 1)) return;
    const ws = new WebSocket(NAV.wsUrl);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => {
      this.state.connected = true;
      this.inFlight = false;
      this.send({ type: "hello", role: "sim", client: "SLAM3D" });
      this.emit();
    };
    ws.onclose = () => {
      this.state.connected = false;
      this.inFlight = false;
      this.emit();
      clearTimeout(this.retry);
      this.retry = setTimeout(() => this.connect(), 1000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }
      if (m.type === "config") {
        this.state.config = m;
        if (m.mode) this.setModeFromServer(m.mode);
      } else if (m.type === "nav") {
        this.state.nav = m;
        this.state.lastNavAt = performance.now();
        if (m.seq === this.pendingSeq) this.inFlight = false;
        if ("goal" in m) this.state.goal = m.goal;
        if (m.mode && m.mode !== this.state.mode) this.setModeFromServer(m.mode);
      } else if (m.type === "goal") {
        this.state.goal = m.goal;
      } else if (m.type === "mode") {
        this.setModeFromServer(m.mode);
      }
      this.emit();
    };
  }

  setModeFromServer(mode) {
    this.state.mode = mode;
    for (const fn of this.modeListeners) fn(mode);
  }

  send(obj) {
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj));
  }

  /** binary frame, see capture.packFrame(); returns false if the link is busy */
  sendFrame(buf, seq) {
    if (!this.ws || this.ws.readyState !== 1) return false;
    const now = performance.now();
    if (this.inFlight && now - this.sentAt < INFLIGHT_TIMEOUT_MS) return false;
    this.inFlight = true;
    this.sentAt = now;
    this.pendingSeq = seq;
    this.state.sentFrames++;
    this.ws.send(buf);
    return true;
  }

  get canSend() {
    return !!this.ws && this.ws.readyState === 1 && (!this.inFlight || performance.now() - this.sentAt >= INFLIGHT_TIMEOUT_MS);
  }

  /** latest drive command with a watchdog: silence -> stop */
  get command() {
    const n = this.state.nav;
    if (!n || !n.cmd || performance.now() - this.state.lastNavAt > WATCHDOG_MS) return { v: 0, omega: 0, stale: true };
    return { v: n.cmd.v, omega: n.cmd.omega, stale: false };
  }

  setGoalNav(x, y) { this.send({ type: "set_goal", x, y }); }
  clearGoal() { this.send({ type: "clear_goal" }); }
  setMode(auto) { this.state.mode = auto ? "auto" : "manual"; this.send({ type: "set_mode", auto }); this.emit(); }
  resetMap() { this.send({ type: "reset" }); }
  setDepth(mode) { this.send({ type: "set_depth", mode }); }

  subscribe(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  onMode(fn) { this.modeListeners.add(fn); return () => this.modeListeners.delete(fn); }
  emit() { for (const fn of this.listeners) fn(this.state); }
}

export const link = new NavLink();
if (typeof window !== "undefined") window.__nav = link;   // debugging / scripted demos

/** React snapshot of the link state; re-renders at the server's frame rate (~5-8 Hz). */
export function useNav() {
  const [s, setS] = useState(() => ({ ...link.state }));
  useEffect(() => link.subscribe((st) => setS({ ...st })), []);
  return s;
}
