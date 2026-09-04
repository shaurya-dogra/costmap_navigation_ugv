import { useMemo } from "react";
import { Line } from "@react-three/drei";
import * as THREE from "three";
import { useNav } from "../nav/link";
import { toThree, robotToNavWorld } from "../nav/frames";

/** Flag + ring at the server's goal; green once ARRIVED. */
export function GoalMarker() {
  const { goal, nav } = useNav();
  if (!goal) return null;
  const p = toThree(goal.x, goal.y);
  const arrived = nav && nav.status === "ARRIVED";
  const col = arrived ? "#22c55e" : "#f59e0b";
  return (
    <group position={[p.x, 0, p.z]}>
      <mesh position={[0, 1.6, 0]} castShadow>
        <cylinderGeometry args={[0.05, 0.05, 3.2, 12]} />
        <meshStandardMaterial color="#eeeeee" />
      </mesh>
      <mesh position={[0.5, 2.9, 0]}>
        <planeGeometry args={[1.0, 0.6]} />
        <meshStandardMaterial color={col} side={THREE.DoubleSide} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.06, 0]}>
        <ringGeometry args={[0.9, 1.25, 48]} />
        <meshBasicMaterial color={col} transparent opacity={0.85} />
      </mesh>
    </group>
  );
}

/** Global path (world) and local plan (robot frame -> world) drawn on the ground. */
export function PathLines() {
  const { nav } = useNav();
  const gp = nav?.global?.path_world, lp = nav?.local?.path_m, pose = nav?.pose;
  const globalPts = useMemo(() => (gp && gp.length >= 2)
    ? gp.map(([x, y]) => { const t = toThree(x, y); return [t.x, 0.12, t.z]; }) : null, [gp]);
  const localPts = useMemo(() => (lp && lp.length >= 2 && pose)
    ? lp.map(([rx, ry]) => { const w = robotToNavWorld(rx, ry, pose); const t = toThree(w.x, w.y); return [t.x, 0.16, t.z]; }) : null, [lp, pose]);
  return (
    <>
      {globalPts && <Line points={globalPts} color="#facc15" lineWidth={2} transparent opacity={0.9} />}
      {localPts && <Line points={localPts} color="#38bdf8" lineWidth={4} />}
    </>
  );
}
