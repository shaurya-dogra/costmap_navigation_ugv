import { useRef, useState, useMemo, useEffect, useCallback, Suspense } from "react";
import { Canvas } from "@react-three/fiber";
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

export default function App() {
  const [telemetry, setTelemetry] = useState({ speed: "0.00", acceleration: "0.00", force: "0.0", x: "0", z: "0", heading: "0" });
  const [cameraMode, setCameraMode] = useState(0);
  const [auto, setAuto] = useState(NAV.autoAtStart);
  const autoRef = useRef(NAV.autoAtStart);
  const poseRef = useRef({ x: 0, z: 0, heading: 0, y: 0 });
  const pipRef = useRef(null);
  const [collisions, setCollisions] = useState({ count: 0, last: null, log: [] });
  const course = useMemo(() => buildCourse(NAV.seed), []);

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
  }, []);

  return (
    <KeyboardControls map={keyboardMap}>
      <Hud telemetry={telemetry} cameraMode={cameraMode} setCameraMode={setCameraMode} auto={auto} setAuto={setAuto}
        autoRef={autoRef} collisions={collisions} pipRef={pipRef} course={course} />
      <MiniMap course={course} telemetry={telemetry} />

      <Canvas shadows camera={{ position: [0, 8, 14], fov: 50 }}>
        <ambientLight intensity={0.7} />
        <directionalLight position={[30, 50, 20]} castShadow intensity={2.5} shadow-mapSize={[1024, 1024]}
          shadow-camera-left={-60} shadow-camera-right={60} shadow-camera-top={60} shadow-camera-bottom={-60} />
        <hemisphereLight skyColor={"#ffffff"} groundColor={"#444444"} intensity={1.0} />
        <color attach="background" args={["#87ceeb"]} />
        <fog attach="fog" args={["#87ceeb", 60, 160]} />

        <Suspense fallback={null}>
          <Physics gravity={[0, -9.81, 0]}>
            <Vehicle setTelemetry={setTelemetry} cameraMode={cameraMode} setCameraMode={setCameraMode}
              setAuto={setAuto} autoRef={autoRef} poseRef={poseRef} pipRef={pipRef}
              course={course} onCollision={onCollision} />
            <Environment course={course} onGroundClick={onGroundClick} />
          </Physics>
          <GoalMarker />
          <PathLines />
        </Suspense>
      </Canvas>
    </KeyboardControls>
  );
}
