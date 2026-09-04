import { useMemo } from "react";
import { RigidBody, CuboidCollider, BallCollider, CylinderCollider } from "@react-three/rapier";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { mulberry32 } from "../nav/world";

// ---------------------------------------------------------------------------
// procedural textures: the flat green plane of the original scene gave the depth
// model and the segmenter nothing to work with
// ---------------------------------------------------------------------------
function noiseTexture(seed, base, amp, blades, bladeColor, repeat) {
  const size = 256, c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  const rnd = mulberry32(seed);
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const n = rnd();
    img.data[i * 4] = base[0] + n * amp[0]; img.data[i * 4 + 1] = base[1] + n * amp[1];
    img.data[i * 4 + 2] = base[2] + n * amp[2]; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  if (blades) {
    ctx.strokeStyle = bladeColor;
    for (let k = 0; k < blades; k++) {
      const x = rnd() * size, y = rnd() * size;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + (rnd() - 0.5) * 3, y - 3 - rnd() * 5); ctx.stroke();
    }
  }
  ctx.fillStyle = "rgba(120,100,60,0.22)";
  for (let k = 0; k < 40; k++) { ctx.beginPath(); ctx.arc(rnd() * size, rnd() * size, 4 + rnd() * 10, 0, 6.283); ctx.fill(); }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(repeat, repeat);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}
const makeGrass = (seed) => noiseTexture(seed, [40, 95, 25], [40, 60, 25], 900, "rgba(30,70,25,0.55)", 120);
const makeMud = (seed) => noiseTexture(seed + 1, [70, 50, 30], [40, 30, 20], 0, null, 3);
const makeSand = (seed) => noiseTexture(seed + 2, [190, 170, 120], [40, 35, 30], 0, null, 3);

// ---------------------------------------------------------------------------
// obstacles
// ---------------------------------------------------------------------------
function Rock({ x, z, r, seed, color = "#6f6a63" }) {
  const geo = useMemo(() => {
    const g = new THREE.IcosahedronGeometry(r, 2);
    const rnd = mulberry32(seed);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const k = 0.78 + rnd() * 0.42;
      p.setXYZ(i, p.getX(i) * k, p.getY(i) * (0.55 + rnd() * 0.3), p.getZ(i) * k);
    }
    g.computeVertexNormals();
    return g;
  }, [r, seed]);
  return (
    <RigidBody type="fixed" colliders={false} position={[x, 0, z]} userData={{ obstacle: "rock" }}>
      <BallCollider args={[r * 0.75]} position={[0, r * 0.55, 0]} />
      <mesh geometry={geo} position={[0, r * 0.45, 0]} castShadow receiveShadow>
        <meshStandardMaterial color={color} roughness={0.95} flatShading />
      </mesh>
    </RigidBody>
  );
}

function Tree({ x, z, s, gltf, withCollider }) {
  const obj = useMemo(() => gltf.scene.clone(), [gltf]);
  const inner = <primitive object={obj} position={[0, 0, 0]} scale={s} castShadow receiveShadow />;
  if (!withCollider) return <group position={[x, 0, z]}>{inner}</group>;
  return (
    <RigidBody type="fixed" colliders={false} position={[x, 0, z]} userData={{ obstacle: "tree" }}>
      <CylinderCollider args={[2.5, 0.45 * s]} position={[0, 2.5, 0]} />
      {inner}
    </RigidBody>
  );
}

function Bush({ x, z, r, seed }) {
  const blobs = useMemo(() => {
    const rnd = mulberry32(seed);
    return Array.from({ length: 5 }, () => [(rnd() - 0.5) * r, r * 0.45 + rnd() * r * 0.3, (rnd() - 0.5) * r, r * (0.5 + rnd() * 0.4)]);
  }, [r, seed]);
  return (
    <RigidBody type="fixed" colliders={false} position={[x, 0, z]} userData={{ obstacle: "bush" }}>
      <BallCollider args={[r * 0.8]} position={[0, r * 0.5, 0]} />
      {blobs.map(([bx, by, bz, br], i) => (
        <mesh key={i} position={[bx, by, bz]} castShadow receiveShadow>
          <sphereGeometry args={[br, 10, 8]} />
          <meshStandardMaterial color={i % 2 ? "#2f6b2a" : "#3f7f33"} roughness={1} flatShading />
        </mesh>
      ))}
    </RigidBody>
  );
}

function Log({ x, z, yaw, len, r }) {
  return (
    <RigidBody type="fixed" colliders={false} position={[x, r, z]} rotation={[0, -yaw, 0]} userData={{ obstacle: "log" }}>
      <CuboidCollider args={[len / 2, r, r]} />
      <mesh rotation={[0, 0, Math.PI / 2]} castShadow receiveShadow>
        <cylinderGeometry args={[r, r * 0.85, len, 12]} />
        <meshStandardMaterial color="#5a4128" roughness={1} />
      </mesh>
      <mesh position={[len * 0.3, r * 0.6, 0]} rotation={[0, 0, 0.6]} castShadow>
        <cylinderGeometry args={[r * 0.3, r * 0.2, r * 3, 8]} />
        <meshStandardMaterial color="#4a3520" roughness={1} />
      </mesh>
    </RigidBody>
  );
}

function Pole({ x, z, h, sign }) {
  return (
    <RigidBody type="fixed" colliders={false} position={[x, 0, z]} userData={{ obstacle: "pole" }}>
      <CylinderCollider args={[h / 2, 0.1]} position={[0, h / 2, 0]} />
      <mesh position={[0, h / 2, 0]} castShadow>
        <cylinderGeometry args={[0.07, 0.09, h, 10]} />
        <meshStandardMaterial color="#8d8d8d" metalness={0.5} roughness={0.5} />
      </mesh>
      {sign && (
        <mesh position={[0, h - 0.35, 0]} castShadow>
          <boxGeometry args={[0.7, 0.5, 0.04]} />
          <meshStandardMaterial color="#e3b23c" roughness={0.6} />
        </mesh>
      )}
    </RigidBody>
  );
}

function Fence({ f }) {
  const posts = [];
  for (let x = f.x0; x <= f.x1 + 1e-6; x += 1.5) posts.push(x);
  const len = f.x1 - f.x0, cx = (f.x0 + f.x1) / 2;
  return (
    <RigidBody type="fixed" colliders={false} userData={{ obstacle: "fence" }}>
      <CuboidCollider args={[len / 2, f.h / 2, 0.08]} position={[cx, f.h / 2, f.z]} />
      {posts.map((x) => (
        <mesh key={x} position={[x, f.h / 2, f.z]} castShadow>
          <boxGeometry args={[0.12, f.h, 0.12]} />
          <meshStandardMaterial color="#8a6a3f" roughness={0.9} />
        </mesh>
      ))}
      {[0.45, 0.9].map((y) => (
        <mesh key={y} position={[cx, y, f.z]} castShadow>
          <boxGeometry args={[len, 0.08, 0.06]} />
          <meshStandardMaterial color="#9c7a4a" roughness={0.9} />
        </mesh>
      ))}
    </RigidBody>
  );
}

function FlatPatch({ x, z, r, y, material, rim }) {
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[x, y, z]} receiveShadow>
        <circleGeometry args={[r, 48]} />{material}
      </mesh>
      {rim && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[x, y - 0.005, z]}>
          <ringGeometry args={[r, r + 0.5, 48]} />
          <meshStandardMaterial color="#5a4a30" roughness={1} />
        </mesh>
      )}
    </group>
  );
}

function Mound({ m, grass }) {
  // a squashed hemisphere: a real slope in the depth buffer, rover can drive up it
  return (
    <RigidBody type="fixed" colliders="hull" friction={2}>
      <mesh position={[m.x, -m.h * 0.15, m.z]} scale={[m.r, m.h, m.r]} receiveShadow castShadow>
        <sphereGeometry args={[1, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial map={grass} roughness={0.95} />
      </mesh>
    </RigidBody>
  );
}

// Ground with real holes for the trenches: slabs between trenches, sunken floors and
// walls, and floor colliders that leave the gaps open so the rover can fall in.
function Ground({ course, grass, onGroundClick }) {
  const L = 200;
  const click = (e) => { e.stopPropagation(); onGroundClick?.(e.point); };
  const mat = <meshStandardMaterial map={grass} roughness={0.95} />;
  // z-bands not cut by any trench, from +L down to -L
  const trs = [...course.trenches].sort((a, b) => b.z - a.z);   // descending z (nearest first)
  const bands = [];
  let zTop = L;
  for (const tr of trs) { bands.push([tr.z + tr.halfW, zTop]); zTop = tr.z - tr.halfW; }
  bands.push([-L, zTop]);
  return (
    <>
      {bands.map(([z0, z1], i) => (
        <RigidBody key={`band${i}`} type="fixed" friction={2} restitution={0}>
          <CuboidCollider args={[L, 1, (z1 - z0) / 2]} position={[0, -1, (z0 + z1) / 2]} />
          <mesh receiveShadow rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, (z0 + z1) / 2]} onClick={click}>
            <planeGeometry args={[2 * L, z1 - z0]} />{mat}
          </mesh>
        </RigidBody>
      ))}
      {trs.map((tr) => (
        <group key={tr.id}>
          {/* the strip of ground the trench does NOT cut */}
          <RigidBody type="fixed" friction={2} restitution={0}>
            <CuboidCollider args={[(tr.x0 + L) / 2, 1, tr.halfW]} position={[(tr.x0 - L) / 2, -1, tr.z]} />
            <CuboidCollider args={[(L - tr.x1) / 2, 1, tr.halfW]} position={[(L + tr.x1) / 2, -1, tr.z]} />
            <mesh receiveShadow rotation={[-Math.PI / 2, 0, 0]} position={[(tr.x0 - L) / 2, 0, tr.z]} onClick={click}>
              <planeGeometry args={[tr.x0 + L, 2 * tr.halfW]} />{mat}
            </mesh>
            <mesh receiveShadow rotation={[-Math.PI / 2, 0, 0]} position={[(L + tr.x1) / 2, 0, tr.z]} onClick={click}>
              <planeGeometry args={[L - tr.x1, 2 * tr.halfW]} />{mat}
            </mesh>
          </RigidBody>
          {/* floor (with a collider, so a rover that drops in lands in the trench
              instead of falling out of the world) + walls; falling in is the failure */}
          <RigidBody type="fixed" friction={2} restitution={0}>
            <CuboidCollider args={[(tr.x1 - tr.x0) / 2, 0.5, tr.halfW]} position={[(tr.x0 + tr.x1) / 2, -tr.depth - 0.5, tr.z]} />
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[(tr.x0 + tr.x1) / 2, -tr.depth, tr.z]} receiveShadow>
              <planeGeometry args={[tr.x1 - tr.x0, 2 * tr.halfW]} />
              <meshStandardMaterial color="#3b2f22" roughness={1} />
            </mesh>
          </RigidBody>
          {[tr.z + tr.halfW, tr.z - tr.halfW].map((zz, i) => (
            <mesh key={i} position={[(tr.x0 + tr.x1) / 2, -tr.depth / 2, zz]}>
              <boxGeometry args={[tr.x1 - tr.x0, tr.depth, 0.02]} />
              <meshStandardMaterial color="#4a3a2a" roughness={1} side={THREE.DoubleSide} />
            </mesh>
          ))}
          {[tr.x0, tr.x1].map((xx, i) => (
            <mesh key={i} position={[xx, -tr.depth / 2, tr.z]}>
              <boxGeometry args={[0.02, tr.depth, 2 * tr.halfW]} />
              <meshStandardMaterial color="#4a3a2a" roughness={1} side={THREE.DoubleSide} />
            </mesh>
          ))}
        </group>
      ))}
    </>
  );
}

export default function Environment({ course, onGroundClick }) {
  const roadGLTF = useGLTF("/road.glb");
  const treeGLTF = useGLTF("/tree.glb");
  const grass = useMemo(() => makeGrass(course.seed), [course.seed]);
  const mudTex = useMemo(() => makeMud(course.seed), [course.seed]);
  const sandTex = useMemo(() => makeSand(course.seed), [course.seed]);
  const click = (e) => { e.stopPropagation(); onGroundClick?.(e.point); };

  return (
    <>
      <Ground course={course} grass={grass} onGroundClick={onGroundClick} />

      {/* road: visual only. Was at y = 0.2, which put the grass exactly at the
          ditch threshold (-0.20 m) relative to the road surface. */}
      <primitive object={roadGLTF.scene} position={[0, 0.02, 0]} receiveShadow onClick={click} />

      {/* terrain patches: semantics decides how expensive these are */}
      {course.sand.map((p) => <FlatPatch key={p.id} {...p} y={0.035} material={<meshStandardMaterial map={sandTex} roughness={1} />} />)}
      {course.mud.map((p) => <FlatPatch key={p.id} {...p} y={0.035} material={<meshStandardMaterial map={mudTex} roughness={0.7} />} />)}
      {course.ponds.map((p) => <FlatPatch key={p.id} {...p} y={0.04} rim material={<meshStandardMaterial color="#2f6fae" roughness={0.15} metalness={0.35} />} />)}
      <Mound m={course.mound} grass={grass} />

      {/* positive obstacles */}
      {course.rocks.map((r, i) => <Rock key={r.id} {...r} seed={course.seed * 31 + i} />)}
      {course.rubble.map((r, i) => <Rock key={r.id} {...r} seed={course.seed * 17 + i} color="#7d776e" />)}
      {course.trees.map((t) => <Tree key={t.id} {...t} gltf={treeGLTF} withCollider />)}
      {course.bgTrees.map((t) => <Tree key={t.id} {...t} gltf={treeGLTF} withCollider={false} />)}
      {course.bushes.map((b, i) => <Bush key={b.id} {...b} seed={course.seed * 53 + i} />)}
      {course.logs.map((l) => <Log key={l.id} {...l} />)}
      {course.poles.map((p) => <Pole key={p.id} {...p} />)}
      {course.fences.map((f) => <Fence key={f.id} f={f} />)}
    </>
  );
}

useGLTF.preload("/road.glb");
useGLTF.preload("/tree.glb");
