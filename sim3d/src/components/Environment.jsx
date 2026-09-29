import { useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { RigidBody, CuboidCollider, BallCollider, CylinderCollider, CapsuleCollider, HeightfieldCollider } from "@react-three/rapier";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { mulberry32, TERRAIN, TRAIL_HALF, inDitch, trailX } from "../nav/world";

// ===========================================================================
// procedural textures (no external assets beyond the tree / rover models)
// ===========================================================================
function canvasTex(size, draw, repeat = 1, srgb = true) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  draw(c.getContext("2d"), size);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeat, repeat);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

/** Grass: mottled base, darker clumps, thousands of short blades. Tiles every 2 m. */
function grassTexture(seed) {
  const rnd = mulberry32(seed);
  return canvasTex(512, (g, S) => {
    const img = g.createImageData(S, S);
    for (let i = 0; i < S * S; i++) {
      const n = rnd();
      img.data[i * 4] = 52 + n * 30; img.data[i * 4 + 1] = 92 + n * 42; img.data[i * 4 + 2] = 30 + n * 18; img.data[i * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    for (let k = 0; k < 60; k++) {             // clumps and bare patches
      const x = rnd() * S, y = rnd() * S, r = 10 + rnd() * 34;
      g.fillStyle = rnd() < 0.7 ? `rgba(30,70,22,${0.18 + rnd() * 0.2})` : `rgba(120,105,60,${0.12 + rnd() * 0.12})`;
      g.beginPath(); g.ellipse(x, y, r, r * (0.5 + rnd() * 0.5), rnd() * 3, 0, 6.283); g.fill();
    }
    for (let k = 0; k < 5200; k++) {          // blades
      const x = rnd() * S, y = rnd() * S, l = 3 + rnd() * 7;
      g.strokeStyle = rnd() < 0.5 ? "rgba(28,74,24,0.7)" : "rgba(118,160,64,0.55)";
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + (rnd() - 0.5) * 4, y - l); g.stroke();
    }
  });
}

/** Packed-earth trail: brown base, two worn wheel ruts, pebbles. u across, v along. */
function dirtTexture(seed) {
  const rnd = mulberry32(seed + 5);
  return canvasTex(512, (g, S) => {
    const img = g.createImageData(S, S);
    for (let i = 0; i < S * S; i++) {
      const n = rnd();
      img.data[i * 4] = 112 + n * 38; img.data[i * 4 + 1] = 88 + n * 30; img.data[i * 4 + 2] = 60 + n * 22; img.data[i * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    for (const cx of [0.3 * S, 0.7 * S]) {    // ruts
      const grd = g.createLinearGradient(cx - 40, 0, cx + 40, 0);
      grd.addColorStop(0, "rgba(70,52,34,0)"); grd.addColorStop(0.5, "rgba(70,52,34,0.35)"); grd.addColorStop(1, "rgba(70,52,34,0)");
      g.fillStyle = grd; g.fillRect(cx - 40, 0, 80, S);
    }
    for (let k = 0; k < 900; k++) {           // pebbles
      const x = rnd() * S, y = rnd() * S, r = 0.8 + rnd() * 2.6, v = 120 + rnd() * 90;
      g.fillStyle = `rgb(${v},${v * 0.92},${v * 0.82})`;
      g.beginPath(); g.arc(x, y, r, 0, 6.283); g.fill();
    }
    g.fillStyle = "rgba(60,110,40,0.35)";     // grass creeping in along the edges
    for (let k = 0; k < 500; k++) { const e = rnd() < 0.5 ? rnd() * 0.12 : 1 - rnd() * 0.12; g.fillRect(e * S, rnd() * S, 2, 4 + rnd() * 6); }
  });
}

function earthTexture(seed, base, spread, specks) {
  const rnd = mulberry32(seed);
  return canvasTex(256, (g, S) => {
    const img = g.createImageData(S, S);
    for (let i = 0; i < S * S; i++) {
      const n = rnd();
      for (let c = 0; c < 3; c++) img.data[i * 4 + c] = base[c] + n * spread[c];
      img.data[i * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    for (let k = 0; k < specks; k++) {
      g.fillStyle = `rgba(0,0,0,${0.08 + rnd() * 0.12})`;
      g.beginPath(); g.arc(rnd() * S, rnd() * S, 2 + rnd() * 8, 0, 6.283); g.fill();
    }
  });
}

/** Equirectangular sky: deep zenith, hazy horizon, sun glow, a few soft clouds. */
function skyTexture(seed) {
  const rnd = mulberry32(seed + 99);
  const c = document.createElement("canvas");
  c.width = 2048; c.height = 1024;
  const g = c.getContext("2d");
  const grd = g.createLinearGradient(0, 0, 0, 1024);
  grd.addColorStop(0.0, "#3d74c4"); grd.addColorStop(0.35, "#79aee6"); grd.addColorStop(0.49, "#cfe3f2");
  grd.addColorStop(0.5, "#d9e6ea"); grd.addColorStop(1.0, "#8a9a80");
  g.fillStyle = grd; g.fillRect(0, 0, 2048, 1024);
  const sx = 1500, sy = 300;                               // sun
  const sun = g.createRadialGradient(sx, sy, 0, sx, sy, 260);
  sun.addColorStop(0, "rgba(255,252,235,1)"); sun.addColorStop(0.06, "rgba(255,245,210,0.9)"); sun.addColorStop(1, "rgba(255,240,200,0)");
  g.fillStyle = sun; g.fillRect(0, 0, 2048, 1024);
  for (let k = 0; k < 26; k++) {                           // clouds
    const cx = rnd() * 2048, cy = 180 + rnd() * 280, w = 120 + rnd() * 260;
    for (let b = 0; b < 9; b++) {
      const bx = cx + (rnd() - 0.5) * w, by = cy + (rnd() - 0.5) * w * 0.18, br = w * (0.12 + rnd() * 0.16);
      const cg = g.createRadialGradient(bx, by, 0, bx, by, br);
      cg.addColorStop(0, "rgba(255,255,255,0.55)"); cg.addColorStop(1, "rgba(255,255,255,0)");
      g.fillStyle = cg; g.beginPath(); g.arc(bx, by, br, 0, 6.283); g.fill();
    }
  }
  const t = new THREE.CanvasTexture(c);
  t.mapping = THREE.EquirectangularReflectionMapping;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function Sky({ seed }) {
  const { scene } = useThree();
  useMemo(() => {
    scene.background = skyTexture(seed);
    scene.fog = new THREE.Fog("#c9d9df", 70, 260);
  }, [scene, seed]);
  return null;
}

// ===========================================================================
// terrain: one fine heightfield around the course (physics + render) and a
// coarse skirt out to the horizon (render only)
// ===========================================================================
function gridGeometry(x0, z0, nx, nz, res, height, colorOf) {
  const pos = new Float32Array((nx + 1) * (nz + 1) * 3);
  const uv = new Float32Array((nx + 1) * (nz + 1) * 2);
  const col = colorOf ? new Float32Array((nx + 1) * (nz + 1) * 3) : null;
  let k = 0;
  for (let r = 0; r <= nz; r++) {
    const z = z0 + r * res;
    for (let c = 0; c <= nx; c++, k++) {
      const x = x0 + c * res, y = height(x, z);
      pos.set([x, y, z], k * 3);
      uv.set([x / 2, z / 2], k * 2);                     // grass tiles every 2 m
      if (col) col.set(colorOf(x, z, y), k * 3);
    }
  }
  const idx = new Uint32Array(nx * nz * 6);
  let i = 0;
  for (let r = 0; r < nz; r++) for (let c = 0; c < nx; c++) {
    const a = r * (nx + 1) + c, b = a + 1, d = a + (nx + 1), e = d + 1;
    idx.set([a, d, b, b, d, e], i); i += 6;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  if (col) g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  return g;
}

function Terrain({ course, grass, onGroundClick }) {
  const T = TERRAIN;
  const nx = Math.round((T.x1 - T.x0) / T.res), nz = Math.round((T.z1 - T.z0) / T.res);
  const { geo, heights } = useMemo(() => {
    const rnd = mulberry32(course.seed + 3);
    const patch = Array.from({ length: 40 }, () => [T.x0 + rnd() * (T.x1 - T.x0), T.z0 + rnd() * (T.z1 - T.z0), 3 + rnd() * 7, rnd()]);
    const colorOf = (x, z, y) => {
      // large-scale tint so the 2 m grass tile never reads as a repeat
      let v = 1.0, w = 0;
      for (const [px, pz, pr, t] of patch) { const d = Math.hypot(x - px, z - pz); if (d < pr) w += (1 - d / pr) * (t - 0.5); }
      v += 0.18 * w + 0.04 * Math.sin(x * 0.13 + z * 0.07);
      const bank = y < -0.08 ? Math.min(1, -y / 0.4) : 0;         // ditch / pond banks go to earth
      return [v * (1 - bank) + 0.62 * bank, v * (1 - bank) + 0.5 * bank, v * (1 - bank) + 0.38 * bank];
    };
    const geo = gridGeometry(T.x0, T.z0, nx, nz, T.res, course.height, colorOf);
    // Rapier heightfield: rows along z, columns along x, index = r + c * (nrows + 1)
    // (probed: row 0 at local z = -scale.z/2, column 0 at local x = -scale.x/2)
    const heights = new Array((nz + 1) * (nx + 1));
    for (let c = 0; c <= nx; c++) for (let r = 0; r <= nz; r++)
      heights[r + c * (nz + 1)] = course.height(T.x0 + c * T.res, T.z0 + r * T.res);
    return { geo, heights };
  }, [course, nx, nz]);
  const click = (e) => { e.stopPropagation(); onGroundClick?.(e.point); };
  return (
    <RigidBody type="fixed" colliders={false} friction={2} restitution={0}>
      <HeightfieldCollider args={[nz, nx, heights, { x: T.x1 - T.x0, y: 1, z: T.z1 - T.z0 }]}
        position={[(T.x0 + T.x1) / 2, 0, (T.z0 + T.z1) / 2]} />
      <mesh geometry={geo} receiveShadow onClick={click}>
        <meshStandardMaterial map={grass} vertexColors roughness={0.97} />
      </mesh>
    </RigidBody>
  );
}

function Horizon({ course, grass }) {
  const geo = useMemo(() => {
    const L = 420, res = 6, n = Math.round(2 * L / res);
    const T = TERRAIN;
    const h = (x, z) => {
      const inside = x > T.x0 + 1 && x < T.x1 - 1 && z > T.z0 + 1 && z < T.z1 - 1;
      return course.height(x, z) - (inside ? 0.3 : 0.02);      // tucked under the fine terrain
    };
    return gridGeometry(-L, -L - 65, n, n, res, h, null);
  }, [course]);
  return (
    <mesh geometry={geo} receiveShadow>
      <meshStandardMaterial map={grass} color="#b8c9a0" roughness={1} />
    </mesh>
  );
}

// ===========================================================================
// ground surfaces that follow the terrain: trail ribbon, pads, patches, water
// ===========================================================================
function Trail({ course, dirt }) {
  const geo = useMemo(() => {
    const pts = course.trail, H = course.height;
    const cut = (x, z) => course.ditches.some((d) => inDitch(d, x, z, 0.1)) ||
      course.ponds.some((p) => Math.hypot(p.x - x, p.z - z) < p.R + 0.2);
    const pos = [], uv = [], idx = [];
    const ACROSS = 6;
    let along = 0, prevOk = false;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i], q = pts[Math.min(i + 1, pts.length - 1)], o = pts[Math.max(i - 1, 0)];
      const tx = q.x - o.x, tz = q.z - o.z, tl = Math.hypot(tx, tz) || 1;
      const nx = -tz / tl, nz = tx / tl;
      if (i > 0) along += Math.hypot(p.x - pts[i - 1].x, p.z - pts[i - 1].z);
      // fine sub-steps so the ribbon hugs the ground and the ditch cut is clean
      const sub = i < pts.length - 1 ? 4 : 1;
      for (let s = 0; s < sub; s++) {
        const f = s / sub, px = p.x + (q.x - p.x) * f, pz = p.z + (q.z - p.z) * f;
        const a = along + Math.hypot(q.x - p.x, q.z - p.z) * f;
        const base = pos.length / 3;
        let ok = true;
        for (let k = 0; k <= ACROSS; k++) {
          const w = (k / ACROSS - 0.5) * 2 * TRAIL_HALF, x = px + nx * w, z = pz + nz * w;
          if (cut(x, z)) ok = false;
          pos.push(x, H(x, z) + 0.012, z); uv.push(k / ACROSS, a / 4);
        }
        if (prevOk && ok) for (let k = 0; k < ACROSS; k++) {
          const a0 = base - (ACROSS + 1) + k, b0 = a0 + 1, c0 = base + k, d0 = c0 + 1;
          idx.push(a0, b0, c0, b0, d0, c0);          // CCW seen from above
        }
        prevOk = ok;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  }, [course]);
  const edge = useMemo(() => canvasTex(64, (g, S) => {       // soft edges: grass bleeds into the track
    const grd = g.createLinearGradient(0, 0, S, 0);
    grd.addColorStop(0, "#000"); grd.addColorStop(0.14, "#fff"); grd.addColorStop(0.86, "#fff"); grd.addColorStop(1, "#000");
    g.fillStyle = grd; g.fillRect(0, 0, S, S);
  }, 1, false), []);
  return (
    <mesh geometry={geo} receiveShadow>
      <meshStandardMaterial map={dirt} alphaMap={edge} transparent depthWrite roughness={1} polygonOffset polygonOffsetFactor={-2} />
    </mesh>
  );
}

/** A disc laid on the terrain (pads, mud, sand). */
function GroundDisc({ x, z, r, H, material, lift = 0.02 }) {
  const geo = useMemo(() => {
    const g = new THREE.CircleGeometry(r, 48, 0, Math.PI * 2);
    g.rotateX(-Math.PI / 2);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) p.setY(i, H(x + p.getX(i), z + p.getZ(i)) + lift);
    g.computeVertexNormals();
    return g;
  }, [x, z, r, H, lift]);
  return <mesh geometry={geo} position={[x, 0, z]} receiveShadow>{material}</mesh>;
}

/** A gravel clearing (start / goal area). Deliberately NO paint: a high-contrast painted
 *  ring and letter read as "wall"/"fence" to the ADE20K segmenter (measured: 27 % of the
 *  pad's pixels), which is exactly what a real outdoor site would not have. The letters
 *  live on signposts beside the pads instead. */
function gravelTexture(seed) {
  const rnd = mulberry32(seed + 21);
  return canvasTex(256, (g, S) => {
    g.fillStyle = "#8c8578"; g.fillRect(0, 0, S, S);
    for (let k = 0; k < 5000; k++) {
      const v = 95 + rnd() * 90, r = 0.6 + rnd() * 1.8;
      g.fillStyle = `rgb(${v},${v * 0.96},${v * 0.88})`;
      g.beginPath(); g.arc(rnd() * S, rnd() * S, r, 0, 6.283); g.fill();
    }
  }, 3);
}

function letterTexture(letter) {
  return canvasTex(128, (g, S) => {
    g.fillStyle = "#1f4f8a"; g.fillRect(0, 0, S, S);
    g.fillStyle = "#f5f5f0"; g.font = "bold 96px sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(letter, S / 2, S / 2 + 6);
  });
}

function Pond({ p, H }) {
  const geo = useMemo(() => {
    // water fills the bowl up to `water` below grade: find the shoreline radius, where
    // the terrain comes up to the water level (plus a little, to meet the bank)
    let lo = 0, hi = p.R + 0.5;
    for (let k = 0; k < 30; k++) { const m = (lo + hi) / 2; if (H(p.x + m, p.z) < -p.water) lo = m; else hi = m; }
    const g = new THREE.CircleGeometry(lo + 0.15, 56);
    g.rotateX(-Math.PI / 2);
    return g;
  }, [p]);
  return (
    <mesh geometry={geo} position={[p.x, -p.water, p.z]}>
      <meshStandardMaterial color="#3d6c86" roughness={0.08} metalness={0.55} transparent opacity={0.92} />
    </mesh>
  );
}

// ===========================================================================
// obstacles (all rest ON the terrain: y = height(x, z))
// ===========================================================================
function Rock({ x, z, r, y, seed, color = "#77716a" }) {
  const geo = useMemo(() => {
    const g = new THREE.IcosahedronGeometry(r, 3);
    const rnd = mulberry32(seed);
    const p = g.attributes.position;
    const k1 = 0.75 + rnd() * 0.3, k2 = 0.55 + rnd() * 0.25;
    for (let i = 0; i < p.count; i++) {
      const n = 0.88 + 0.22 * Math.sin(p.getX(i) * 5.1 + seed) * Math.cos(p.getZ(i) * 4.3) + rnd() * 0.06;
      p.setXYZ(i, p.getX(i) * n * k1 * 1.15, p.getY(i) * n * k2, p.getZ(i) * n);
    }
    g.computeVertexNormals();
    return g;
  }, [r, seed]);
  return (
    <RigidBody type="fixed" colliders={false} position={[x, y, z]}>
      <BallCollider args={[r * 0.75]} position={[0, r * 0.5, 0]} />
      <mesh geometry={geo} position={[0, r * 0.32, 0]} rotation={[0, seed, 0]} castShadow receiveShadow>
        <meshStandardMaterial color={color} roughness={0.93} flatShading />
      </mesh>
    </RigidBody>
  );
}

function Tree({ x, z, s, y, gltf, withCollider }) {
  const obj = useMemo(() => gltf.scene.clone(), [gltf]);
  const inner = <primitive object={obj} scale={s} castShadow receiveShadow />;
  if (!withCollider) return <group position={[x, y, z]}>{inner}</group>;
  return (
    <RigidBody type="fixed" colliders={false} position={[x, y, z]}>
      <CylinderCollider args={[2.5, 0.45 * s]} position={[0, 2.5, 0]} />
      {inner}
    </RigidBody>
  );
}

function Bush({ x, z, r, y, seed }) {
  const blobs = useMemo(() => {
    const rnd = mulberry32(seed);
    return Array.from({ length: 7 }, () => [(rnd() - 0.5) * r * 1.1, r * 0.4 + rnd() * r * 0.35, (rnd() - 0.5) * r * 1.1, r * (0.42 + rnd() * 0.35)]);
  }, [r, seed]);
  return (
    <RigidBody type="fixed" colliders={false} position={[x, y, z]}>
      <BallCollider args={[r * 0.8]} position={[0, r * 0.5, 0]} />
      {blobs.map(([bx, by, bz, br], i) => (
        <mesh key={i} position={[bx, by, bz]} castShadow receiveShadow>
          <icosahedronGeometry args={[br, 1]} />
          <meshStandardMaterial color={["#2f6b2a", "#3f7f33", "#355f28"][i % 3]} roughness={1} flatShading />
        </mesh>
      ))}
    </RigidBody>
  );
}

function FallenTree({ l, H }) {
  const y = H(l.x, l.z);
  return (
    <RigidBody type="fixed" colliders={false} position={[l.x, y + l.r * 0.9, l.z]} rotation={[0, -l.yaw, 0]}>
      <CuboidCollider args={[l.len / 2, l.r, l.r]} />
      <mesh rotation={[0, 0, Math.PI / 2]} castShadow receiveShadow>
        <cylinderGeometry args={[l.r * 0.8, l.r, l.len, 14]} />
        <meshStandardMaterial color="#5b4430" roughness={1} />
      </mesh>
      {/* root plate at the thick end */}
      <mesh position={[-l.len / 2 - 0.1, 0.25, 0]} rotation={[0, 0, Math.PI / 2]} castShadow>
        <cylinderGeometry args={[1.0, 0.8, 0.35, 10]} />
        <meshStandardMaterial color="#4a3a28" roughness={1} flatShading />
      </mesh>
      {[0.15, 0.45, -0.2].map((t, i) => (
        <mesh key={i} position={[l.len * t, l.r * 0.7, (i - 1) * 0.15]} rotation={[0.4 * (i - 1), 0, 0.5 + i * 0.3]} castShadow>
          <cylinderGeometry args={[0.07, 0.1, 1.2, 6]} />
          <meshStandardMaterial color="#4f3b27" roughness={1} />
        </mesh>
      ))}
    </RigidBody>
  );
}

function Pole({ x, z, y, h, sign, flag, letter }) {
  const lt = useMemo(() => (letter ? letterTexture(letter) : null), [letter]);
  return (
    <RigidBody type="fixed" colliders={false} position={[x, y, z]}>
      <CylinderCollider args={[h / 2, 0.1]} position={[0, h / 2, 0]} />
      <mesh position={[0, h / 2, 0]} castShadow>
        <cylinderGeometry args={[0.06, 0.08, h, 10]} />
        <meshStandardMaterial color="#9a9a9a" metalness={0.5} roughness={0.5} />
      </mesh>
      {sign && (
        <mesh position={[0, h - 0.3, 0]} castShadow>
          <boxGeometry args={[0.8, 0.5, 0.04]} />
          <meshStandardMaterial color="#e3b23c" roughness={0.6} />
        </mesh>
      )}
      {lt && (
        <mesh position={[0, h - 0.45, 0.03]} castShadow>
          <boxGeometry args={[0.7, 0.7, 0.04]} />
          <meshStandardMaterial map={lt} roughness={0.6} />
        </mesh>
      )}
      {flag && (
        <mesh position={[0.55, h - 0.35, 0]} castShadow>
          <boxGeometry args={[1.0, 0.6, 0.02]} />
          <meshStandardMaterial color="#d6452e" roughness={0.7} />
        </mesh>
      )}
    </RigidBody>
  );
}

/** Low grass tufts across the meadow (kept < 0.1 m: under the obstacle threshold). */
function Tufts({ course }) {
  const n = 4500;
  const init = (mesh) => {
    if (!mesh || mesh.userData.done) return;
    const rnd = mulberry32(course.seed + 11), m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler();
    let k = 0, guard = 0;
    while (k < n && guard++ < n * 4) {
      const d = -6 + rnd() * 150, off = (rnd() - 0.5) * 44;
      if (Math.abs(off) < TRAIL_HALF + 0.5) continue;                         // not on the trail
      const z = -d, x = trailX(z) + off;
      if (course.ditches.some((dt) => inDitch(dt, x, z, 0.5))) continue;
      if (course.ponds.some((p) => Math.hypot(p.x - x, p.z - z) < p.R + 0.5)) continue;
      if (course.pads.some((p) => Math.hypot(p.x - x, p.z - z) < p.r + 0.3)) continue;
      const s = 0.6 + rnd() * 0.8;
      e.set(0, rnd() * 6.28, 0); q.setFromEuler(e);
      m.compose(new THREE.Vector3(x, course.height(x, z) + 0.03 * s, z), q, new THREE.Vector3(s, s, s));
      mesh.setMatrixAt(k++, m);
    }
    mesh.count = k;
    mesh.instanceMatrix.needsUpdate = true;
    mesh.userData.done = true;
  };
  return (
    <instancedMesh ref={init} args={[null, null, n]} receiveShadow>
      <coneGeometry args={[0.09, 0.09, 5]} />
      <meshStandardMaterial color="#4f8a36" roughness={1} flatShading />
    </instancedMesh>
  );
}

// ===========================================================================
// the dynamic obstacle: a person in a hi-vis vest walking across the trail
// ===========================================================================
function Walker({ m, H }) {
  const body = useRef(), legL = useRef(), legR = useRef(), armL = useRef(), armR = useRef();
  const st = useRef({ t: 0, dir: 1, wait: 0, s: 0 });
  const L = Math.hypot(m.b.x - m.a.x, m.b.z - m.a.z);
  useFrame((_, dt) => {
    const S = st.current;
    dt = Math.min(dt, 0.1);
    let moving = false;
    if (S.wait > 0) S.wait -= dt;
    else {
      S.s += S.dir * m.speed * dt;
      moving = true;
      if (S.s >= L) { S.s = L; S.dir = -1; S.wait = m.pause; }
      if (S.s <= 0) { S.s = 0; S.dir = 1; S.wait = m.pause; }
    }
    const f = S.s / L, x = m.a.x + (m.b.x - m.a.x) * f, z = m.a.z + (m.b.z - m.a.z) * f;
    const y = H(x, z);
    m.pos.x = x; m.pos.z = z;
    const yaw = Math.atan2(S.dir * (m.b.x - m.a.x), S.dir * (m.b.z - m.a.z));
    if (body.current) {
      body.current.setNextKinematicTranslation({ x, y, z });
      body.current.setNextKinematicRotation(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, yaw, 0)));
    }
    S.t += moving ? dt * m.speed * 5.5 : 0;
    const sw = moving ? Math.sin(S.t) * 0.55 : 0;
    if (legL.current) { legL.current.rotation.x = sw; legR.current.rotation.x = -sw; armL.current.rotation.x = -sw * 0.8; armR.current.rotation.x = sw * 0.8; }
  });
  const skin = <meshStandardMaterial color="#c68e6a" roughness={0.8} />;
  const cloth = <meshStandardMaterial color="#2b3a55" roughness={0.9} />;
  return (
    <RigidBody ref={body} type="kinematicPosition" colliders={false} position={[m.a.x, H(m.a.x, m.a.z), m.a.z]}>
      <CapsuleCollider args={[0.55, 0.3]} position={[0, 0.88, 0]} />
      <group ref={legL} position={[-0.11, 0.9, 0]}><mesh position={[0, -0.44, 0]} castShadow><capsuleGeometry args={[0.075, 0.72, 4, 8]} />{cloth}</mesh></group>
      <group ref={legR} position={[0.11, 0.9, 0]}><mesh position={[0, -0.44, 0]} castShadow><capsuleGeometry args={[0.075, 0.72, 4, 8]} />{cloth}</mesh></group>
      <mesh position={[0, 1.22, 0]} castShadow><capsuleGeometry args={[0.19, 0.42, 4, 10]} /><meshStandardMaterial color="#f2a516" roughness={0.6} /></mesh>
      <group ref={armL} position={[-0.26, 1.42, 0]}><mesh position={[0, -0.3, 0]} castShadow><capsuleGeometry args={[0.055, 0.5, 4, 8]} /><meshStandardMaterial color="#f2a516" roughness={0.6} /></mesh></group>
      <group ref={armR} position={[0.26, 1.42, 0]}><mesh position={[0, -0.3, 0]} castShadow><capsuleGeometry args={[0.055, 0.5, 4, 8]} /><meshStandardMaterial color="#f2a516" roughness={0.6} /></mesh></group>
      <mesh position={[0, 1.66, 0]} castShadow><sphereGeometry args={[0.11, 14, 12]} />{skin}</mesh>
    </RigidBody>
  );
}

/** Obstacles dropped in front of the rover at run time: a crate. */
function Crate({ o, H }) {
  const y = H(o.x, o.z);
  return (
    <RigidBody type="fixed" colliders={false} position={[o.x, y, o.z]} rotation={[0, o.yaw || 0, 0]}>
      <CuboidCollider args={[0.45, 0.45, 0.45]} position={[0, 0.45, 0]} />
      <mesh position={[0, 0.45, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.9, 0.9, 0.9]} />
        <meshStandardMaterial color="#8b6a3e" roughness={0.9} />
      </mesh>
    </RigidBody>
  );
}

// ===========================================================================
export default function Environment({ course, onGroundClick, drops = [] }) {
  const treeGLTF = useGLTF("/tree.glb");
  const H = course.height;
  const grass = useMemo(() => grassTexture(course.seed), [course.seed]);
  const dirt = useMemo(() => dirtTexture(course.seed), [course.seed]);
  const mudTex = useMemo(() => earthTexture(course.seed + 1, [62, 44, 28], [30, 22, 14], 60), [course.seed]);
  const sandTex = useMemo(() => earthTexture(course.seed + 2, [196, 176, 128], [34, 30, 24], 20), [course.seed]);
  const gravel = useMemo(() => gravelTexture(course.seed), [course.seed]);

  return (
    <>
      <Sky seed={course.seed} />
      <Terrain course={course} grass={grass} onGroundClick={onGroundClick} />
      <Horizon course={course} grass={grass} />
      <Trail course={course} dirt={dirt} />
      <Tufts course={course} />

      {course.pads.map((p) => (
        <GroundDisc key={p.id} {...p} H={H} lift={0.025}
          material={<meshStandardMaterial map={gravel} roughness={0.95} polygonOffset polygonOffsetFactor={-4} />} />
      ))}
      {course.sand.map((p) => <GroundDisc key={p.id} {...p} H={H} material={<meshStandardMaterial map={sandTex} roughness={1} polygonOffset polygonOffsetFactor={-3} />} />)}
      {course.mud.map((p) => <GroundDisc key={p.id} {...p} H={H} material={<meshStandardMaterial map={mudTex} roughness={0.55} polygonOffset polygonOffsetFactor={-3} />} />)}
      {course.ponds.map((p) => <Pond key={p.id} p={p} H={H} />)}

      {course.rocks.map((r, i) => <Rock key={r.id} {...r} y={H(r.x, r.z)} seed={course.seed * 31 + i} />)}
      {course.scatter.map((r, i) => <Rock key={r.id} {...r} y={H(r.x, r.z)} seed={course.seed * 17 + i} color="#827c73" />)}
      {course.trees.map((t) => <Tree key={t.id} {...t} y={H(t.x, t.z)} gltf={treeGLTF} withCollider />)}
      {course.bgTrees.map((t) => <Tree key={t.id} {...t} y={H(t.x, t.z)} gltf={treeGLTF} withCollider={false} />)}
      {course.bushes.map((b, i) => <Bush key={b.id} {...b} y={H(b.x, b.z)} seed={course.seed * 53 + i} />)}
      {course.logs.map((l) => <FallenTree key={l.id} l={l} H={H} />)}
      {course.poles.map((p) => <Pole key={p.id} {...p} y={H(p.x, p.z)} />)}
      {course.movers.map((m) => <Walker key={m.id} m={m} H={H} />)}
      {drops.map((o) => <Crate key={o.id} o={o} H={H} />)}
    </>
  );
}

useGLTF.preload("/tree.glb");
