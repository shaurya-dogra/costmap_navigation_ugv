import { useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { NAV } from "../nav/config";
import { link } from "../nav/link";
import { PovCapture, packFrame } from "../nav/capture";
import { toNavWorld } from "../nav/frames";

// The robot's eye. A child of the rover's heading group, so it turns with the
// rover; NOT a child of the animated GLTF (the startup clip would move it).
// Streams RGB + true depth to the perception server at <= maxHz with at most
// one frame in flight, inside the default-priority useFrame so the chase view
// keeps rendering normally.
export default function RobotCamera({ poseRef, pipRef, autoRef }) {
  const camRef = useRef();
  const capRef = useRef(null);
  const busy = useRef(false);
  const lastT = useRef(0);
  const seq = useRef(0);
  const { gl, scene } = useThree();
  const { w, h } = NAV.capture;

  useEffect(() => {
    const cam = camRef.current;
    if (!cam) return;
    cam.layers.set(0);                 // never see the rover itself (it is on layer 1)
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
    return () => { capRef.current?.dispose(); capRef.current = null; };
  }, [w, h]);

  useFrame(() => {
    const cam = camRef.current;
    if (!cam || !link.state.connected) return;
    const now = performance.now();
    if (busy.current || now - lastT.current < 1000 / NAV.capture.maxHz || !link.canSend) return;
    if (!capRef.current) { capRef.current = new PovCapture(gl, scene, cam, NAV.capture); window.__cap = capRef.current; }
    busy.current = true;
    lastT.current = now;
    const pose = { ...poseRef.current };            // snapshot at capture time
    const mode = autoRef.current ? "auto" : "manual";
    capRef.current.grab().then(({ jpeg, depth, canvas, depthValid }) => {
      const s = ++seq.current;
      const nav = toNavWorld(pose.x, pose.z, pose.heading);
      const { fx, fy, cx, cy } = capRef.current.intrinsics;
      const header = {
        type: "frame", seq: s, t: Date.now(), w, h, fx, fy, cx, cy,
        cam_height: NAV.mount.height, cam_pitch: NAV.mount.pitchDeg * Math.PI / 180,
        pose: nav, mode,
        depth: { w: NAV.capture.depthW, h: NAV.capture.depthH, unit: "mm" },
        depth_valid: Math.round(depthValid * 1000) / 1000,
      };
      link.sendFrame(packFrame(header, jpeg, depth), s);
      const pip = pipRef && pipRef.current;
      if (pip) {
        const ctx = pip.getContext("2d");
        ctx.drawImage(canvas, 0, 0, pip.width, pip.height);
      }
    }).catch((e) => console.warn("capture failed", e)).finally(() => { busy.current = false; });
  });

  return (
    <perspectiveCamera
      ref={camRef}
      fov={NAV.capture.vfovDeg}
      near={NAV.capture.near}
      far={NAV.capture.far}
      position={[0, NAV.mount.height, -NAV.mount.forward]}
      rotation={[-NAV.mount.pitchDeg * Math.PI / 180, 0, 0]}
    />
  );
}
