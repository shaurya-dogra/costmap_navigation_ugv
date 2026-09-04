#!/usr/bin/env bash
# Run the full 3D demo: perception server (sim mode) + the SLAM3D rover in Vite.
#
#   ./run_sim.sh                 # ground-truth depth from the renderer (robust demo)
#   ./run_sim.sh --depth metric  # neural depth (Depth Anything V2 metric) instead
#   SLAM3D_DIR=/path/to/SLAM3D ./run_sim.sh
#
# Then the browser opens http://localhost:5173/?auto=1 . Click the ground, the
# course map (bottom-left) or the global map to set a destination.
set -euo pipefail
cd "$(dirname "$0")"

# the sim lives in ./sim3d (a copy of https://github.com/Klick07/SLAM3D with the
# autonomy additions); SLAM3D_DIR can point at a separate checkout instead
SLAM3D_DIR="${SLAM3D_DIR:-$([[ -d sim3d ]] && echo sim3d || echo ../SLAM3D)}"
PORT="${PORT:-8790}"
# pick the next free port if the default is busy (e.g. a sim server still running)
port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
while port_busy "$PORT"; do echo "port $PORT busy, trying $((PORT+1))"; PORT=$((PORT+1)); done
VITE_PORT="${VITE_PORT:-5173}"
DEPTH="sim"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --depth) DEPTH="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ ! -d .venv ]]; then
  echo "no .venv - run ./setup_mac.sh first"; exit 1
fi
if [[ ! -d "$SLAM3D_DIR" ]]; then
  echo "SLAM3D not found at $SLAM3D_DIR (set SLAM3D_DIR)"; exit 1
fi
if [[ ! -d "$SLAM3D_DIR/node_modules" ]]; then
  echo "installing SLAM3D dependencies..."; (cd "$SLAM3D_DIR" && npm install)
fi
for f in rover.glb road.glb tree.glb; do
  if [[ ! -f "$SLAM3D_DIR/public/$f" ]]; then
    if [[ -f "$f" ]]; then cp "$f" "$SLAM3D_DIR/public/"; else echo "missing $SLAM3D_DIR/public/$f"; exit 1; fi
  fi
done

export PYTORCH_ENABLE_MPS_FALLBACK=1
cleanup() { echo; echo "stopping..."; kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[1/2] perception server  ->  http://localhost:$PORT   (depth: $DEPTH)"
.venv/bin/python perception_server.py --source sim --depth "$DEPTH" --auto --port "$PORT" --profile "${EXTRA[@]}" &

echo "[2/2] SLAM3D rover       ->  http://localhost:$VITE_PORT"
(cd "$SLAM3D_DIR" && npx vite --port "$VITE_PORT" --strictPort) &

# wait for the server to load its models, then open the sim
for _ in $(seq 1 60); do curl -s "localhost:$PORT/status" >/dev/null && break; sleep 2; done
URL="http://localhost:$VITE_PORT/?auto=1&ws=ws://localhost:$PORT/ws"
echo; echo "open  $URL"; echo "dashboard  http://localhost:$PORT"; echo "Ctrl-C stops both."
command -v open >/dev/null && open "$URL" || true
wait
