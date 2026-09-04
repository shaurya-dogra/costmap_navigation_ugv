// The outdoor course, in three.js coordinates (x right, z toward the camera;
// the rover starts at the origin heading -z). Also the GROUND TRUTH used to
// score the run: the costmap never sees this list, it only sees pixels.
//
// Every hazard type exercises a different perception channel:
//   rocks / trees / logs / poles / fences / rubble -> positive obstacles, lethal by HEIGHT
//   trenches                                       -> negative obstacles, lethal by DEPTH only
//   ponds / puddles                                -> flat, lethal by SEMANTICS only
//   mud / sand                                     -> flat, drivable but COSTLY by semantics
//   bushes                                         -> low positive obstacles (height ~0.5 m)
//   mound                                          -> a slope the plane fit must not mistake for a wall

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

export const ROAD_HALF = 6.0;   // road.glb is ~12 m wide about x = 0

export function buildCourse(seed = 7) {
  const rnd = mulberry32(seed);
  const j = (a) => (rnd() - 0.5) * 2 * a;   // jitter +-a

  // ---- positive obstacles on and beside the road ------------------------------
  const rocks = [
    { id: "rock1", x: 1.5 + j(0.4), z: -12 + j(0.5), r: 0.9 },
    { id: "rock2", x: -2.5 + j(0.4), z: -20 + j(0.5), r: 0.7 },
    { id: "rock3", x: 0.6 + j(0.4), z: -31 + j(0.5), r: 0.8 },
    { id: "rock4", x: -1.2 + j(0.4), z: -47 + j(0.5), r: 1.0 },
    { id: "rock5", x: 4.5 + j(0.4), z: -55 + j(0.5), r: 0.6 },
    { id: "rock6", x: -4.0 + j(0.4), z: -64 + j(0.5), r: 1.2 },
    { id: "rock7", x: 2.0 + j(0.4), z: -72 + j(0.5), r: 0.7 },
    { id: "rock8", x: -0.5 + j(0.4), z: -86 + j(0.5), r: 0.9 },
  ];
  // rubble: clusters of small stones (individually small, together a wall)
  const rubble = [];
  const clusters = [{ x: -3.5, z: -9 }, { x: 3.0, z: -44 }, { x: -2.0, z: -78 }, { x: 5.5, z: -35 }];
  clusters.forEach((c, ci) => {
    for (let k = 0; k < 7; k++) rubble.push({ id: `rubble${ci}_${k}`, x: c.x + j(1.4), z: c.z + j(1.4), r: 0.25 + rnd() * 0.25 });
  });
  const trees = [
    { id: "tree1", x: -3.2, z: -16, s: 1.0 },
    { id: "tree2", x: 3.6, z: -27, s: 0.9 },
    { id: "tree3", x: -1.6, z: -41, s: 1.1 },
    { id: "tree4", x: 2.4, z: -58, s: 1.0 },
    { id: "tree5", x: -4.5, z: -70, s: 1.2 },
    { id: "tree6", x: 1.0, z: -82, s: 0.9 },
    { id: "tree7", x: 5.0, z: -90, s: 1.0 },
    { id: "tree8", x: -7.5, z: -30, s: 1.1 },
    { id: "tree9", x: 8.0, z: -66, s: 0.9 },
  ];
  // fallen logs: (x, z) centre, yaw, length
  const logs = [
    { id: "log1", x: -4.0, z: -25, yaw: 0.35, len: 4.5, r: 0.28 },
    { id: "log2", x: 3.5, z: -50, yaw: -0.2, len: 3.5, r: 0.25 },
    { id: "log3", x: -1.0, z: -75, yaw: 1.2, len: 5.0, r: 0.3 },
  ];
  // bushes: low, round, drivable-looking but 0.5 m high
  const bushes = [];
  for (let k = 0; k < 18; k++) {
    const side = rnd() < 0.5 ? -1 : 1;
    bushes.push({ id: `bush${k}`, x: side * (2.0 + rnd() * 6.5), z: -6 - rnd() * 88, r: 0.5 + rnd() * 0.5 });
  }
  // poles / signposts: thin, tall
  const poles = [
    { id: "pole1", x: -5.5, z: -14, h: 2.4, sign: true },
    { id: "pole2", x: 5.8, z: -40, h: 3.0, sign: false },
    { id: "pole3", x: -0.4, z: -60, h: 2.6, sign: true },
    { id: "pole4", x: 6.0, z: -80, h: 3.2, sign: false },
  ];

  // ---- negative obstacles ----------------------------------------------------
  const trenches = [
    { id: "trench1", z: -38, halfW: 0.7, x0: -7.0, x1: 2.5, depth: 0.5 },   // way round on the right
    { id: "trench2", z: -68, halfW: 0.6, x0: -2.0, x1: 8.0, depth: 0.6 },   // way round on the left
  ];

  // ---- flat hazards & terrain -------------------------------------------------
  const ponds = [
    { id: "pond1", x: 5.0, z: -24, r: 3.5 },
    { id: "pond2", x: -5.5, z: -56, r: 2.8 },
    { id: "puddle1", x: 1.5, z: -84, r: 1.4 },
  ];
  const mud = [
    { id: "mud1", x: -2.0, z: -33, r: 3.0 },
    { id: "mud2", x: 3.0, z: -62, r: 2.5 },
  ];
  const sand = [
    { id: "sand1", x: 4.0, z: -12, r: 2.6 },
    { id: "sand2", x: -4.0, z: -88, r: 3.2 },
  ];
  // a gentle mound beside the road: slope, not an obstacle
  const mound = { id: "mound", x: -9.5, z: -48, r: 6.0, h: 1.6 };

  const fences = [
    { id: "fence1", z: -52, x0: -2.0, x1: 7.0, h: 1.1 },   // way round on the left
    { id: "fence2", z: -92, x0: -8.0, x1: 1.5, h: 1.1 },   // way round on the right
  ];

  const bgTrees = [];
  while (bgTrees.length < 44) {
    const x = (rnd() - 0.5) * 200, z = (rnd() - 0.5) * 220 - 20;
    if (Math.abs(x) < 11) continue;
    bgTrees.push({ id: `bg${bgTrees.length}`, x, z, s: 0.8 + rnd() * 0.5 });
  }
  const goalPreset = { navX: 100, navY: 0 };          // three (0, -100)
  return { seed, rocks, rubble, trees, logs, bushes, poles, trenches, ponds, mud, sand, mound, fences, bgTrees, goalPreset };
}

/** Ground-truth contacts of a rover disc (x, z, radius r) at body height y. */
export function groundTruthHits(x, z, y, course, r) {
  const hits = [];
  const disc = (list, type, rr) => {
    for (const o of list) if (Math.hypot(o.x - x, o.z - z) < (rr ? rr(o) : o.r) + r) hits.push({ id: o.id, type });
  };
  disc(course.rocks, "rock");
  disc(course.rubble, "rubble");
  disc(course.trees, "tree", (t) => 0.45 * t.s);
  disc(course.bushes, "bush");
  disc(course.poles, "pole", () => 0.1);
  for (const p of course.ponds) if (Math.hypot(p.x - x, p.z - z) < p.r) hits.push({ id: p.id, type: "water" });
  for (const l of course.logs) {
    // distance from the disc centre to the log's axis segment
    const c = Math.cos(l.yaw), s = Math.sin(l.yaw);
    const dx = x - l.x, dz = z - l.z;
    const along = Math.max(-l.len / 2, Math.min(l.len / 2, dx * c + dz * s));
    const px = l.x + along * c, pz = l.z + along * s;
    if (Math.hypot(x - px, z - pz) < l.r + r) hits.push({ id: l.id, type: "log" });
  }
  for (const tr of course.trenches)
    if (x >= tr.x0 && x <= tr.x1 && Math.abs(z - tr.z) < tr.halfW + r * 0.5 && y < -0.15) hits.push({ id: tr.id, type: "ditch" });
  for (const f of course.fences)
    if (x >= f.x0 - r && x <= f.x1 + r && Math.abs(z - f.z) < 0.15 + r) hits.push({ id: f.id, type: "fence" });
  return hits;
}
