// The demo course, in three.js coordinates (x right, z toward the camera; the rover
// starts at the origin heading -z). Also the GROUND TRUTH used to score a run: the
// perception stack never sees this file, it only sees pixels and depth.
//
// Layout - a 130 m dirt trail from A to B through open meadow, one challenge per zone,
// each zone separated from the next and sized for the real robot: the server inflates
// every obstacle by a 1 m robot radius, so every detour leaves >= 3.5 m of clear width.
//
//   d =   0      A  start pad
//   d =  14-32   boulder field           positive obstacles, lethal by HEIGHT
//   d =  42      fallen tree on the trail positive obstacle across the way, detour left
//   d =  56      washed-out drainage ditch negative obstacle, lethal by DEPTH, go round its end
//   d =  68-80   pond beside the trail + a puddle ON it   water: a real depression, so
//                                                        DEPTH and SEMANTICS both see it
//   d =  90      a person crossing the trail           DYNAMIC obstacle
//   d = 100-116  woodland: trees and bushes close to the trail
//   d = 120      mud across the trail    drivable, costly by semantics
//   d = 130      B  goal pad
//
// (d = distance along the course = -z.) Hills rise beyond ~24 m either side of the
// trail: scenery, past the 20 m depth horizon, never on the route.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- the trail ---------------------------------------------------------------------
export const TRAIL_HALF = 1.9;                       // a 3.8 m dirt track
export const COURSE_LEN = 130;
/** Trail centreline x at z (gentle S-bends, straight through the start pad). */
export function trailX(z) {
  const d = -z;
  const ramp = Math.min(1, Math.max(0, (d - 4) / 12));          // straight out of A
  return ramp * (4.2 * Math.sin((2 * Math.PI * d) / 95) + 1.2 * Math.sin((2 * Math.PI * d) / 37));
}
/** Point `off` metres to the LEFT of the trail centre (rover's left when driving -z). */
const at = (d, off) => ({ x: trailX(-d) - off, z: -d });

// ---- terrain -------------------------------------------------------------------------
export const TERRAIN = { x0: -38, x1: 38, z0: -150, z1: 22, res: 0.3 };  // fine heightfield

const smooth = (e0, e1, v) => { const t = Math.min(1, Math.max(0, (v - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };

function ditchLocal(dt, x, z) {
  const c = Math.cos(dt.yaw), s = Math.sin(dt.yaw);
  const dx = x - dt.x, dz = z - dt.z;
  return { u: dx * c + dz * s, v: -dx * s + dz * c };     // u along the ditch, v across
}

export function makeHeight(course) {
  const { ditches, ponds } = course;
  return function height(x, z) {
    // meadow: a few centimetres of relief - real, but far inside the plane-fit gate
    let h = 0.025 * Math.sin(x * 0.61 + 1.3) * Math.cos(z * 0.47) + 0.015 * Math.sin(x * 1.7 + z * 1.3);
    // hills well away from the route (scenery beyond the depth horizon)
    const off = Math.abs(x - trailX(z));
    const hill = smooth(24, 40, off);
    if (hill > 0) h += hill * (3.0 + 1.8 * Math.sin(z / 19 + x / 23) + 1.2 * Math.cos(z / 11 - x / 17));
    const back = smooth(8, 30, z) + smooth(-(COURSE_LEN + 6), -(COURSE_LEN + 28), z);  // behind A, beyond B
    if (back > 0) h += back * (2.5 + 1.5 * Math.sin(x / 13));
    for (const dt of ditches) {
      const { u, v } = ditchLocal(dt, x, z);
      const inLen = smooth(dt.len / 2 + 0.3, dt.len / 2, Math.abs(u));   // rounded ends
      const inW = smooth(dt.w / 2 + 0.15, dt.w / 2 - 0.15, Math.abs(v));  // steep banks
      h -= dt.depth * inLen * inW;
    }
    for (const p of ponds) {
      const r = Math.hypot(x - p.x, z - p.z);
      if (r < p.R + 0.5) h -= p.depth * smooth(p.R + 0.4, p.R * 0.55, r);
    }
    return h;
  };
}

// ---- the course ------------------------------------------------------------------
export function buildCourse(seed = 7) {
  const rnd = mulberry32(seed);
  const j = (a) => (rnd() - 0.5) * 2 * a;

  // zone 1: boulder field - staggered so there is always a lane >= 3.5 m wide
  const rocks = [
    [15, 0.4, 1.0], [19.5, -3.4, 0.7], [23, 3.3, 0.9], [27, -0.6, 1.1], [31, 3.8, 0.7], [31.5, -4.6, 0.8],
  ].map(([d, off, r], i) => ({ id: `boulder${i + 1}`, ...at(d + j(0.3), off + j(0.2)), r }));

  // zone 2: a fallen tree across the trail; its root end blocks the right, go left
  const lg = at(42, -1.6);
  const logs = [{ id: "fallen_tree", x: lg.x, z: lg.z, yaw: 0.12, len: 8.5, r: 0.32 }];

  // zone 3: a washed-out drainage ditch cutting the trail; go round its left end
  const dc = at(56, -2.0);
  const ditches = [{ id: "ditch", x: dc.x, z: dc.z, yaw: 0.10, len: 15, w: 1.7, depth: 0.6 }];

  // zone 4: a pond to the left, and a puddle sitting on the trail itself
  const pd = at(73, 8.2), pu = at(77, -0.2);
  const ponds = [
    { id: "pond", x: pd.x, z: pd.z, R: 5.2, depth: 0.75, water: 0.32 },
    { id: "puddle", x: pu.x, z: pu.z, R: 1.8, depth: 0.45, water: 0.26 },
  ];

  // zone 5: a person walking back and forth across the trail
  const w0 = at(90, 7.5), w1 = at(90, -7.5);
  const movers = [{ id: "walker", type: "person", a: w0, b: w1, speed: 0.9, pause: 1.6, r: 0.35,
                    pos: { x: w0.x, z: w0.z }, heading: 0 }];

  // zone 6: woodland - trunks close to the trail on both sides, bushes between
  const trees = [];
  [[100, 5.2], [102.5, -5.6], [105, 6.4], [107.5, -4.9], [110, 5.4], [112.5, -6.2], [115, 5.0],
   [101, 9.5], [104, -9.8], [109, 10.2], [113, -10.5]]
    .forEach(([d, off], i) => trees.push({ id: `tree${i + 1}`, ...at(d + j(0.4), off + j(0.3)), s: 0.9 + rnd() * 0.3 }));
  const bushes = [[103.5, 2.9, 0.6], [108.8, -3.1, 0.55], [106, 0.3, 0.5], [114, 3.2, 0.6]]
    .map(([d, off, r], i) => ({ id: `bush${i + 1}`, ...at(d, off), r }));

  // zone 7: mud across the trail (drivable, expensive), sand beside it
  const mud = [{ id: "mud", ...at(121, 0.6), r: 3.2 }];
  const sand = [{ id: "sand", ...at(9, -4.5), r: 2.2 }, { id: "sand2", ...at(124, 6.0), r: 2.6 }];

  // start / goal pads and the goal flag (the flag pole is off the pad: an obstacle)
  const A = at(0, 0), B = at(COURSE_LEN, 0);
  const pads = [{ id: "A", x: A.x, z: A.z, r: 2.4 }, { id: "B", x: B.x, z: B.z, r: 2.4 }];
  const fb = at(COURSE_LEN, -3.4);
  const poles = [{ id: "flag_B", x: fb.x, z: fb.z, h: 3.2, flag: true, letter: "B" },
                 { id: "sign_A", ...at(1.5, 3.6), h: 1.8, letter: "A" }];

  // scenery: stones and trees scattered well off the route (real obstacles, just far)
  const scatter = [];
  while (scatter.length < 36) {
    const d = 4 + rnd() * (COURSE_LEN + 4), side = rnd() < 0.5 ? -1 : 1, off = side * (11 + rnd() * 12);
    scatter.push({ id: `stone${scatter.length}`, ...at(d, off), r: 0.25 + rnd() * 0.45 });
  }
  const bgTrees = [];
  while (bgTrees.length < 70) {
    const d = -12 + rnd() * (COURSE_LEN + 30), side = rnd() < 0.5 ? -1 : 1, off = side * (14 + rnd() * 26);
    bgTrees.push({ id: `bg${bgTrees.length}`, ...at(d, off), s: 0.8 + rnd() * 0.6 });
  }

  // trail polyline for rendering and the map
  const trail = [];
  for (let d = -2; d <= COURSE_LEN + 3; d += 1) trail.push({ x: trailX(-d), z: -d });

  const course = {
    seed, trail, pads, rocks, logs, ditches, ponds, movers, trees, bushes, mud, sand, poles, scatter, bgTrees,
    drops: [],                                         // obstacles dropped at run time ("sudden obstacle")
    goalPreset: { navX: -B.z, navY: -B.x },            // nav frame: X = -three.z, Y = -three.x
  };
  course.height = makeHeight(course);
  return course;
}

/** Is (x, z) inside a ditch footprint? */
export function inDitch(dt, x, z, pad = 0) {
  const { u, v } = ditchLocal(dt, x, z);
  return Math.abs(u) < dt.len / 2 + pad && Math.abs(v) < dt.w / 2 + pad;
}

/**
 * Ground-truth contacts of the rover disc (x, z, radius r) with body height y.
 * `moving`: a person walking INTO a parked rover is not the rover's collision.
 */
export function groundTruthHits(x, z, y, course, r, moving = true) {
  const hits = [];
  const disc = (list, type, rr) => {
    for (const o of list) if (Math.hypot(o.x - x, o.z - z) < (rr ? rr(o) : o.r) + r) hits.push({ id: o.id, type });
  };
  disc(course.rocks, "rock");
  disc(course.scatter, "rock");
  disc(course.trees, "tree", (t) => 0.45 * t.s);
  disc(course.bushes, "bush");
  disc(course.poles, "pole", () => 0.1);
  disc(course.drops, "dropped obstacle");
  if (moving) for (const m of course.movers) if (Math.hypot(m.pos.x - x, m.pos.z - z) < m.r + r) hits.push({ id: m.id, type: m.type });
  for (const p of course.ponds) if (Math.hypot(p.x - x, p.z - z) < p.R * 0.8) hits.push({ id: p.id, type: "water" });
  for (const l of course.logs) {
    const c = Math.cos(l.yaw), s = Math.sin(l.yaw);
    const dx = x - l.x, dz = z - l.z;
    const along = Math.max(-l.len / 2, Math.min(l.len / 2, dx * c + dz * s));
    const px = l.x + along * c, pz = l.z + along * s;
    if (Math.hypot(x - px, z - pz) < l.r + r) hits.push({ id: l.id, type: "log" });
  }
  for (const dt of course.ditches) if (inDitch(dt, x, z, r * 0.3) && y < -0.15) hits.push({ id: dt.id, type: "ditch" });
  return hits;
}
