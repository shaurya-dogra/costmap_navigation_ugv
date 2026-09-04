#!/usr/bin/env bash
# Live webcam -> self-calibrating costmap, shown in the browser dashboard.
#
#   ./run_webcam.sh                  # MacBook camera (index 0), metric depth
#   ./run_webcam.sh --source 1       # another camera index or a video file
#   ./run_webcam.sh --hfov 70        # if you know the camera's horizontal FOV
#   ./run_webcam.sh --windows        # also show OpenCV windows
#
# Put the laptop on the ground with the lid at ~90 degrees. Camera height, pitch
# and roll are MEASURED every frame and shown on the dashboard - nothing to set.
# Turn off Center Stage / auto-framing (it changes the focal length).
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8790}"
# pick the next free port if the default is busy (e.g. a sim server still running)
port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
while port_busy "$PORT"; do echo "port $PORT busy, trying $((PORT+1))"; PORT=$((PORT+1)); done

if [[ ! -d .venv ]]; then
  echo "no .venv - run ./setup_mac.sh first"; exit 1
fi

ARGS=(--source 0 --rig macbook --depth metric --profile --port "$PORT")
if [[ $# -gt 0 ]]; then
  # user flags override / extend the defaults (argparse takes the last value)
  ARGS+=("$@")
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "perception server  ->  http://localhost:$PORT"
echo "  ${ARGS[*]}"
.venv/bin/python perception_server.py "${ARGS[@]}" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT INT TERM

echo "loading models (first run downloads Depth Anything, ~100 MB)..."
for _ in $(seq 1 90); do
  kill -0 $SERVER 2>/dev/null || { echo "server exited - see the error above (camera permission? try: --source 1)"; exit 1; }
  curl -s "localhost:$PORT/status" >/dev/null && break; sleep 2
done
echo; echo "dashboard  http://localhost:$PORT   (Ctrl-C stops)"
command -v open >/dev/null && open "http://localhost:$PORT" || true
wait $SERVER
