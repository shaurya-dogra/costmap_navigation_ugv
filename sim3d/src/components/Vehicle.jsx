import { useRef, useMemo, useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { RigidBody, BallCollider } from "@react-three/rapier";
import { useKeyboardControls, useGLTF, useAnimations } from "@react-three/drei";
import * as THREE from "three";
import { NAV } from "../nav/config";
import { link } from "../nav/link";
import { groundTruthHits } from "../nav/world";
import RobotCamera from "./RobotCamera";

export default function Vehicle({ setTelemetry, cameraMode, setCameraMode, setAuto, autoRef, poseRef, pipRef, course, onCollision }) {
  const chassisRef = useRef(null);
  const visualGroupRef = useRef(null);
  const prevLinvel = useRef(new THREE.Vector3());
  const mass = 45;
  const { camera } = useThree();

  const roverGLTF = useGLTF("/rover.glb");

  // the rover lives on layer 1 so its own camera (layer 0 only) never sees it
  useEffect(() => {
    roverGLTF.scene.rotation.y = Math.PI;
    roverGLTF.scene.traverse((o) => o.layers.set(1));
    camera.layers.enable(1);
  }, [roverGLTF.scene, camera]);

  const slicedAnimations = useMemo(() => {
    if (!roverGLTF.animations.length) return [];
    const fps = 30;
    return [THREE.AnimationUtils.subclip(roverGLTF.animations[0], "Startup", 0, 9 * fps, fps)];
  }, [roverGLTF.animations]);
  const { actions } = useAnimations(slicedAnimations, visualGroupRef);
  const [, getKeys] = useKeyboardControls();
  useEffect(() => {
    const a = actions["Startup"];
    if (a) { a.setLoop(THREE.LoopOnce, 1); a.clampWhenFinished = true; a.play(); }
  }, [actions]);

  const lastToggle = useRef(0);
  const headingAngle = useRef(0);
  const steerAngle = useRef(0);
  const currentCamPos = useRef(new THREE.Vector3());
  const currentTargetPos = useRef(new THREE.Vector3());
  const activeHits = useRef(new Set());
  const smoothV = useRef(0);
  const telemetryTick = useRef(0);
  const MAX_STEER = 1;

  useFrame((state, delta) => {
    if (!chassisRef.current || !visualGroupRef.current) return;
    const { forward, backward, left, right, brake, cameraToggle, autoToggle } = getKeys();
    const now = state.clock.getElapsedTime();

    if (cameraToggle && now - lastToggle.current > 0.3) { setCameraMode((p) => (p + 1) % 3); lastToggle.current = now; }
    if (autoToggle && now - lastToggle.current > 0.3) {
      const next = !autoRef.current; autoRef.current = next; setAuto(next); link.setMode(next); lastToggle.current = now;
    }
    // a human touching the wheel always wins
    if (autoRef.current && (forward || backward || left || right)) { autoRef.current = false; setAuto(false); link.setMode(false); }

    const linvel = chassisRef.current.linvel();
    const currentVel = new THREE.Vector3(linvel.x, linvel.y, linvel.z);
    const speedVal = new THREE.Vector3(linvel.x, 0, linvel.z).length();
    const accelVal = currentVel.clone().sub(prevLinvel.current).length() / Math.max(delta, 1e-3);
    prevLinvel.current.copy(currentVel);

    let targetVx, targetVz;
    if (autoRef.current) {
      // ---- AUTONOMY: apply the perception server's (v, omega) ------------------
      const { v, omega } = link.command;                 // STOP when stale (watchdog)
      // The server plans at ~5 Hz. Integrating omega for the WHOLE gap between
      // commands overshoots the intended heading change, and the next command then
      // corrects back: that is the zig-zag. Apply omega only for one control
      // period, then hold the heading until the next command arrives.
      const age = (performance.now() - link.state.lastNavAt) / 1000;
      const period = Math.max(0.15, 1 / Math.max(link.state.nav?.fps || 4, 1));
      const effOmega = age < period ? omega : 0;
      headingAngle.current += effOmega * delta;          // omega > 0 = left = heading increases
      smoothV.current = THREE.MathUtils.damp(smoothV.current, v, 3, delta);   // no speed steps
      const dir = new THREE.Vector3(-Math.sin(headingAngle.current), 0, -Math.cos(headingAngle.current));
      targetVx = dir.x * smoothV.current; targetVz = dir.z * smoothV.current;
    } else {
      // ---- MANUAL: original keyboard model --------------------------------------
      let targetSteer = 0;
      if (left) targetSteer += MAX_STEER;
      if (right) targetSteer -= MAX_STEER;
      steerAngle.current = THREE.MathUtils.lerp(steerAngle.current, targetSteer, 10 * delta);
      if (forward || backward || speedVal > 0.05) {
        const sign = forward ? 1 : backward ? -1 : 1;
        headingAngle.current += steerAngle.current * 1.6 * delta * sign * Math.min(1, Math.abs(smoothV.current) / 2 + 0.3);
      }
      const dir = new THREE.Vector3(-Math.sin(headingAngle.current), 0, -Math.cos(headingAngle.current));
      // accelerate / brake smoothly instead of snapping to the target speed
      const want = forward ? NAV.manual.speed : backward ? -NAV.manual.reverse : 0;
      const rate = brake ? 8 : (want === 0 ? 2.5 : 3.5);
      smoothV.current = THREE.MathUtils.damp(smoothV.current, want, rate, delta);
      if (Math.abs(smoothV.current) < 0.02) smoothV.current = 0;
      targetVx = dir.x * smoothV.current; targetVz = dir.z * smoothV.current;
    }
    chassisRef.current.setLinvel({ x: targetVx, y: linvel.y, z: targetVz }, true);
    visualGroupRef.current.rotation.y = headingAngle.current;

    const driveDir = new THREE.Vector3(-Math.sin(headingAngle.current), 0, -Math.cos(headingAngle.current));
    const carPos = chassisRef.current.translation();
    const posVec = new THREE.Vector3(carPos.x, carPos.y, carPos.z);

    // pose reported to the server = ground point under the LENS, not the body centre
    poseRef.current = {
      x: carPos.x + driveDir.x * NAV.mount.forward,
      z: carPos.z + driveDir.z * NAV.mount.forward,
      heading: headingAngle.current, y: carPos.y,
    };

    // ground-truth scoring (the costmap never sees this)
    const hits = groundTruthHits(carPos.x, carPos.z, carPos.y, course, NAV.rover.radius);
    const ids = new Set(hits.map((h) => h.id));
    for (const h of hits) if (!activeHits.current.has(h.id)) onCollision?.(h);
    activeHits.current = ids;

    telemetryTick.current += delta;
    if (telemetryTick.current > 0.1) {
      telemetryTick.current = 0;
      setTelemetry({ speed: speedVal.toFixed(2), acceleration: accelVal.toFixed(2), force: (mass * accelVal).toFixed(1),
        x: carPos.x.toFixed(1), z: carPos.z.toFixed(1), heading: (headingAngle.current * 180 / Math.PI).toFixed(0) });
    }

    // chase / top-down / driver cameras (unchanged behaviour)
    let targetCamPos = new THREE.Vector3();
    let lookAtTarget = posVec.clone().add(new THREE.Vector3(0, 1.2, 0));
    if (cameraMode === 0) targetCamPos = posVec.clone().add(driveDir.clone().negate().multiplyScalar(9)).add(new THREE.Vector3(0, 4.5, 0));
    else if (cameraMode === 1) targetCamPos = posVec.clone().add(new THREE.Vector3(0, 35, 0.1));
    else { targetCamPos = posVec.clone().add(driveDir.clone().multiplyScalar(0.5)).add(new THREE.Vector3(0, 2.5, 0)); lookAtTarget = posVec.clone().add(driveDir.clone().multiplyScalar(10)); }
    currentCamPos.current.lerp(targetCamPos, 0.1);
    currentTargetPos.current.lerp(lookAtTarget, 0.1);
    state.camera.position.copy(currentCamPos.current);
    state.camera.lookAt(currentTargetPos.current);
  });

  return (
    <RigidBody ref={chassisRef} type="dynamic" position={[0, 2, 0]} mass={mass} friction={1.5} canSleep={false}
      linearDamping={0.2} angularDamping={0.2} lockRotations colliders={false}>
      {/* a ball: the heading is applied to the visual group only, so a box would never rotate */}
      <BallCollider args={[NAV.rover.radius]} position={[0, NAV.rover.radius, 0]} />
      <group ref={visualGroupRef}>
        <primitive object={roverGLTF.scene} position={[0, 0, 0]} scale={NAV.rover.scale} castShadow receiveShadow />
        <RobotCamera poseRef={poseRef} pipRef={pipRef} autoRef={autoRef} />
      </group>
    </RigidBody>
  );
}

useGLTF.preload("/rover.glb");
