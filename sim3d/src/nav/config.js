// Runtime configuration for the autonomy link. Override with URL params:
//   ?ws=ws://host:8790/ws   perception server
//   ?seed=12                course randomisation (background trees, rock jitter)
//   ?auto=1                 start in AUTO mode
const q = new URLSearchParams(window.location.search);

export const NAV = {
  wsUrl: q.get("ws") || `ws://${window.location.hostname || "localhost"}:8790/ws`,
  seed: parseInt(q.get("seed") || "7", 10),
  autoAtStart: q.get("auto") === "1",

  // What the robot's camera streams to the perception server.
  capture: {
    w: 640, h: 360,           // RGB JPEG
    depthW: 320, depthH: 180, // true depth, u16 millimetres
    vfovDeg: 60,              // three.js PerspectiveCamera.fov is the VERTICAL fov
    near: 0.1, far: 100,
    maxHz: 6,                 // never faster than this; also never more than 1 frame in flight
    jpegQuality: 0.8,
  },

  // Camera mount on the rover, metres, relative to the rover base on the ground.
  // The server never trusts these: it measures height/pitch/roll from the depth
  // and reports the difference (plane.mount_err) - a live accuracy check.
  mount: { height: 1.0, forward: 0.7, pitchDeg: 15 },

  rover: { scale: 0.5, radius: 0.6 },          // ~1.2 m wide UGV footprint
  manual: { speed: 6, reverse: 3 },            // m/s for keyboard driving
};
