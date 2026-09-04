// Offscreen capture of the robot's point-of-view camera: an RGB JPEG for the
// neural models and the renderer's TRUE depth (u16 millimetres) as the stand-in
// for a stereo camera. Packs both into one binary WebSocket frame.
//
// Two things that are easy to get wrong here:
//  * three.js renders into a render target in LINEAR colour with no tone
//    mapping, so the raw bytes are dark; a linear->sRGB LUT fixes that before
//    the JPEG is encoded (the depth/semantic models expect a normal photo).
//  * the depth pass uses MeshDepthMaterial with RGBADepthPacking (works for
//    skinned, instanced and morph meshes, unlike a bare ShaderMaterial). It packs
//    the NON-LINEAR fragment depth; we unpack and linearise on the CPU. The
//    target is cleared to white, which unpacks to ~1.0 = "nothing here" = 0 mm.
import * as THREE from "three";

const LUT = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  const x = i / 255;
  const y = x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
  LUT[i] = Math.round(Math.min(1, Math.max(0, y)) * 255);
}

export function intrinsicsFromVfov(w, h, vfovDeg) {
  const fy = (h / 2) / Math.tan((vfovDeg * Math.PI / 180) / 2);
  return { fx: fy, fy, cx: w / 2, cy: h / 2 };
}

export class PovCapture {
  constructor(gl, scene, camera, cfg) {
    this.gl = gl; this.scene = scene; this.camera = camera; this.cfg = cfg;
    const { w, h, depthW: dw, depthH: dh } = cfg;
    this.rgbRT = new THREE.WebGLRenderTarget(w, h, { depthBuffer: true, stencilBuffer: false });
    this.depthRT = new THREE.WebGLRenderTarget(dw, dh, {
      minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
      type: THREE.UnsignedByteType, depthBuffer: true, stencilBuffer: false,
    });
    this.depthMat = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
    this.rgb = new Uint8Array(w * h * 4);
    this.dep = new Uint8Array(dw * dh * 4);
    this.flipped = new Uint8ClampedArray(w * h * 4);
    this.u16 = new Uint16Array(dw * dh);
    this.canvas = new OffscreenCanvas(w, h);
    this.ctx = this.canvas.getContext("2d");
    this.intrinsics = intrinsicsFromVfov(w, h, cfg.vfovDeg);
    this._clear = new THREE.Color();
  }

  /** Render both passes and read them back. Resolves to { jpeg: Uint8Array, depth: Uint16Array }. */
  async grab() {
    const { gl, scene, camera, cfg } = this;
    const { w, h, depthW: dw, depthH: dh } = cfg;

    const prevTarget = gl.getRenderTarget();
    gl.getClearColor(this._clear);
    const prevAlpha = gl.getClearAlpha();
    const prevBg = scene.background;
    const prevOverride = scene.overrideMaterial;
    const prevAutoClear = gl.autoClear;

    // pass 1: colour
    gl.autoClear = true;
    gl.setRenderTarget(this.rgbRT);
    gl.render(scene, camera);

    // pass 2: depth, background off, cleared to white (= no measurement)
    scene.background = null;
    scene.overrideMaterial = this.depthMat;
    gl.setClearColor(0xffffff, 1);
    gl.setRenderTarget(this.depthRT);
    gl.clear();
    gl.render(scene, camera);

    scene.overrideMaterial = prevOverride;
    scene.background = prevBg;
    gl.setClearColor(this._clear, prevAlpha);
    gl.setRenderTarget(prevTarget);
    gl.autoClear = prevAutoClear;

    await Promise.all([
      gl.readRenderTargetPixelsAsync(this.rgbRT, 0, 0, w, h, this.rgb),
      gl.readRenderTargetPixelsAsync(this.depthRT, 0, 0, dw, dh, this.dep),
    ]);

    // readPixels is bottom-up: flip rows, apply the sRGB LUT, force alpha 255
    const src = this.rgb, dst = this.flipped, row = w * 4;
    for (let y = 0; y < h; y++) {
      const s = (h - 1 - y) * row, d = y * row;
      for (let x = 0; x < row; x += 4) {
        dst[d + x] = LUT[src[s + x]];
        dst[d + x + 1] = LUT[src[s + x + 1]];
        dst[d + x + 2] = LUT[src[s + x + 2]];
        dst[d + x + 3] = 255;
      }
    }
    this.ctx.putImageData(new ImageData(dst, w, h), 0, 0);

    // unpack RGBADepthPacking -> ndc depth in [0,1] -> linear metres -> u16 mm
    // three r0.185 packing.glsl.js: R is the MOST significant byte, A the least:
    //   v = dot(rgba, vec4(255/256, 255/256/256, 255/256/65536, 1/16777216))
    const near = camera.near, far = camera.far, k = 255 / 256;
    const dp = this.dep, out = this.u16;
    let valid = 0;
    for (let y = 0; y < dh; y++) {
      const s = (dh - 1 - y) * dw * 4, d = y * dw;
      for (let x = 0; x < dw; x++) {
        const i = s + x * 4;
        const v = k * (dp[i] / 255) + k * (dp[i + 1] / 255) / 256 + k * (dp[i + 2] / 255) / 65536 + (dp[i + 3] / 255) / 16777216;
        if (v > 0.9999) { out[d + x] = 0; continue; }
        const viewZ = (near * far) / ((far - near) * v - far);   // negative
        const mm = Math.round(-viewZ * 1000);
        if (mm > 0 && mm < 65535) { out[d + x] = mm; valid++; } else out[d + x] = 0;
      }
    }
    this.depthValid = valid / (dw * dh);

    const blob = await this.canvas.convertToBlob({ type: "image/jpeg", quality: cfg.jpegQuality });
    const jpeg = new Uint8Array(await blob.arrayBuffer());
    return { jpeg, depth: this.u16, canvas: this.canvas, depthValid: this.depthValid };
  }

  dispose() {
    this.rgbRT.dispose(); this.depthRT.dispose(); this.depthMat.dispose();
  }
}

/** u32 LE header length | header JSON | JPEG | u16 LE depth  (PROTOCOL.md §2) */
export function packFrame(header, jpeg, depthU16) {
  const hb = new TextEncoder().encode(JSON.stringify({ ...header, jpeg_len: jpeg.byteLength }));
  const depthBytes = depthU16 ? depthU16.byteLength : 0;
  const buf = new ArrayBuffer(4 + hb.byteLength + jpeg.byteLength + depthBytes);
  const view = new DataView(buf);
  view.setUint32(0, hb.byteLength, true);
  const u8 = new Uint8Array(buf);
  u8.set(hb, 4);
  u8.set(jpeg, 4 + hb.byteLength);
  if (depthU16) u8.set(new Uint8Array(depthU16.buffer, depthU16.byteOffset, depthBytes), 4 + hb.byteLength + jpeg.byteLength);
  return buf;
}
