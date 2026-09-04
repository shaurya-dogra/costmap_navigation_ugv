#!/usr/bin/env python3
"""
Camera intrinsic calibration. Run this ONCE per phone, per resolution.
Print a 9x6 chessboard, tape it flat to a board, hold it at many angles.

  python calibrate.py --source http://192.168.1.7:8080/video

Press SPACE to capture a view (aim for 20), ESC when done.
Paste the printed numbers into Cfg in costmap_prototype.py.
"""
import argparse, numpy as np, cv2

CB = (9, 6)          # inner corners
SQ = 0.025           # square size in metres (measure yours)

ap = argparse.ArgumentParser(); ap.add_argument("--source", default="0")
a = ap.parse_args()
src = int(a.source) if a.source.isdigit() else a.source
cap = cv2.VideoCapture(src)

objp = np.zeros((CB[0] * CB[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CB[0], 0:CB[1]].T.reshape(-1, 2) * SQ
objpts, imgpts, shape = [], [], None

while True:
    ok, f = cap.read()
    if not ok: break
    f = cv2.resize(f, (1280, 720))
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); shape = g.shape[::-1]
    found, corners = cv2.findChessboardCorners(g, CB, None)
    if found:
        cv2.drawChessboardCorners(f, CB, corners, found)
    cv2.putText(f, f"captured {len(objpts)}  SPACE=keep  ESC=done", (10, 30),
                0, 0.8, (0, 255, 0) if found else (0, 0, 255), 2)
    cv2.imshow("calibrate", f)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break
    if k == 32 and found:
        corners = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1),
                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        objpts.append(objp); imgpts.append(corners)

cap.release(); cv2.destroyAllWindows()
if len(objpts) < 8:
    raise SystemExit("need at least 8 good views")

rms, K, dist, _, _ = cv2.calibrateCamera(objpts, imgpts, shape, None, None)
print(f"\nRMS reprojection error: {rms:.3f} px   (want < 0.5)")
print(f"fx = {K[0,0]:.1f}")
print(f"fy = {K[1,1]:.1f}")
print(f"cx = {K[0,2]:.1f}")
print(f"cy = {K[1,2]:.1f}")
print("dist =", np.round(dist.ravel(), 5).tolist())
