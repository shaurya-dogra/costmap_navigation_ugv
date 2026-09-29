import { useRef, useState, useMemo, useEffect, useCallback, Suspense } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Physics } from "@react-three/rapier";
import { KeyboardControls } from "@react-three/drei";
import { NAV } from "./nav/config";
import { link } from "./nav/link";
import { toNavWorld } from "./nav/frames";
import { buildCourse } from "./nav/world";
import Environment from "./components/Environment";
import Vehicle from "./components/Vehicle";
import Hud from "./components/Hud";
import MiniMap from "./components/MiniMap";
import { GoalMarker, PathLines } from "./components/GoalMarker";

const keyboardMap = [
  { name: "forward", keys: ["ArrowUp", "KeyW"] },
  { name: "backward", keys: ["ArrowDown", "KeyS"] },
  { name: "left", keys: ["ArrowLeft", "KeyA"] },
  { name: "right", keys: ["ArrowRight", "KeyD"] },
  { name: "brake", keys: ["Space"] },
  { name: "cameraToggle", keys: ["KeyC"] },
  { name: "autoToggle", keys: ["KeyT"] },
];

/** Late-morning sun from the south-east. It follows the rover so a small shadow box
 *  (sharp shadows where the cameras look) covers whatever is around it. */
function SunLight({ poseRef }) {
  const ref = useRef();
  useFrame(() => {
    const l = ref.current, p = poseRef.current;
    if (!l) return;
    l.position.set(p.x + 40, 65, p.z + 25);
    l.target.position.set(p.x, 0, p.z - 10);
    l.target.updateMatrixWorld();
  });
  return (
    <directionalLight ref={ref} castShadow intensity={2.6} color={"#fff4e0"}
      shadow-mapSize={[2048, 2048]} shadow-bias={-0.0004} shadow-normalBias={0.03}
      shadow-camera-left={-45} shadow-camera-right={45} shadow-camera-top={45} shadow-camera-bottom={-45}
      shadow-camera-near={1} shadow-camera-far={200} />
  );
}

export default function App() {
  const [telemetry, setTelemetry] = useState({ speed: "0.00", acceleration: "0.00", force: "0.0", x: "0", z: "0", heading: "0" });
  const [cameraMode, setCameraMode] = useState(0);
  const [auto, setAuto] = useState(NAV.autoAtStart);
  const autoRef = useRef(NAV.autoAtStart);
  const poseRef = useRef({ x: 0, z: 0, heading: 0, y: 0 });
  const pipRef = useRef(null);
  const [collisions, setCollisions] = useState({ count: 0, last: null, log: [] });
  const course = useMemo(() => buildCourse(NAV.seed), []);
  const [drops, setDrops] = useState([]);

  // "sudden obstacle": drop a crate 6 m in front of the rover (button or the O key)
  const dropObstacle = useCallback(() => {
    const p = poseRef.current, d = 6.0;
    const o = { id: `crate${Date.now() % 100000}`, x: p.x - Math.sin(p.heading) * d, z: p.z - Math.cos(p.heading) * d,
                r: 0.64, yaw: p.heading };
    course.drops.push(o);                      // ground truth scoring reads this list
    setDrops([...course.drops]);
  }, [course]);
  const clearDrops = useCallback(() => { course.drops.length = 0; setDrops([]); }, [course]);
  useEffect(() => {
    const onKey = (e) => { if (e.code === "KeyO" && !(e.target instanceof HTMLInputElement)) dropObstacle(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dropObstacle]);

  useEffect(() => {
    link.connect();
    // the dashboard (or the server) may switch mode too
    return link.onMode((m) => { const a = m === "auto"; autoRef.current = a; setAuto(a); });
  }, []);

  const onGroundClick = useCallback((point) => {
    const n = toNavWorld(point.x, point.z, 0);
    link.setGoalNav(n.x, n.y);
  }, []);

  const onCollision = useCallback((hit) => {
    setCollisions((c) => ({ count: c.count + 1, last: `${hit.type} ${hit.id}`, log: [...c.log, { t: Date.now(), ...hit }].slice(-50) }));
    console.warn("[ground truth] contact:", hit);
    link.reportEvent({ kind: "contact", ...hit });
  }, []);

  return (
    <KeyboardControls map={keyboardMap}>
      <Hud telemetry={telemetry} cameraMode={cameraMode} setCameraMode={setCameraMode} auto={auto} setAuto={setAuto}
        autoRef={autoRef} collisions={collisions} pipRef={pipRef} course={course}
        dropObstacle={dropObstacle} clearDrops={clearDrops} nDrops={drops.length}>
        <MiniMap course={course} telemetry={telemetry} />
      </Hud>

      <Canvas shadows camera={{ position: [0, 8, 14], fov: 50, far: 900 }}>
        <hemisphereLight skyColor={"#dbe9f5"} groundColor={"#5a5f3a"} intensity={0.9} />
        <ambientLight intensity={0.25} />
        <SunLight poseRef={poseRef} />

        <Suspense fallback={null}>
          <Physics gravity={[0, -9.81, 0]}>
            <Vehicle setTelemetry={setTelemetry} cameraMode={cameraMode} setCameraMode={setCameraMode}
              setAuto={setAuto} autoRef={autoRef} poseRef={poseRef} pipRef={pipRef}
              course={course} onCollision={onCollision} />
            <Environment course={course} onGroundClick={onGroundClick} drops={drops} />
          </Physics>
          <GoalMarker />
          <PathLines />
        </Suspense>
      </Canvas>
    </KeyboardControls>
  );
}
