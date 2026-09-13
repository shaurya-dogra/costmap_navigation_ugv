#!/usr/bin/env bash
# Run the whole real-rover demo with one command: perception server here, camera
# agent on the Pi, browser open on the dashboard.
#
#   ./run_rover.sh                       # everything, defaults below
#   ./run_rover.sh --depth metric        # outdoors (default is metric-indoor)
#   ./run_rover.sh --rotation 90         # if the camera is remounted
#   ./run_rover.sh --no-open             # don't open a browser
#   PI=rikshaw@10.218.135.132 ./run_rover.sh
#
# Ctrl-C stops the server here AND the agent on the Pi.
set -uo pipefail
cd "$(dirname "$0")"

# ---- settings (override with env or flags) ---------------------------------------
PI="${PI:-rikshaw@10.218.135.132}"
PORT="${PORT:-8790}"
DEPTH="metric-indoor"      # the outdoor metric model reads a 2 m indoor wall as 5-9 m
ROTATION="180"             # measured for the current mount; see ROVER_PLAN.md
FPS="12"
OPEN_BROWSER=1
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --depth)    DEPTH="$2"; shift 2 ;;
    --rotation) ROTATION="$2"; shift 2 ;;
    --fps)      FPS="$2"; shift 2 ;;
    --pi)       PI="$2"; shift 2 ;;
    --no-open)  OPEN_BROWSER=0; shift ;;
    -h|--help)  sed -n '2,12p' "$0"; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

fail() { echo "  !! $*" >&2; exit 1; }

# ---- preflight: fail loudly HERE rather than halfway through -----------------------
echo "[preflight]"
[[ -d .venv ]] || fail "no .venv - run ./setup_mac.sh first"

# The Pi must reach US, so hand it the address of the interface that routes to it
# rather than whatever `hostname -I` happens to list first.
PI_HOST="${PI#*@}"
MAC_IP="$(route -n get "$PI_HOST" 2>/dev/null | awk '/interface:/{print $2}' \
          | xargs -I{} ipconfig getifaddr {} 2>/dev/null || true)"
[[ -n "$MAC_IP" ]] || MAC_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
[[ -n "$MAC_IP" ]] || fail "cannot work out this Mac's IP - is the network up?"
echo "  this Mac : $MAC_IP"

ssh -o BatchMode=yes -o ConnectTimeout=8 "$PI" true 2>/dev/null \
  || fail "cannot ssh to $PI (key auth). Try: ssh-copy-id $PI"
echo "  pi       : $PI  reachable"

ssh -o BatchMode=yes "$PI" 'test -f rover_agent.py' 2>/dev/null || {
  echo "  agent    : not on the Pi, copying"
  scp -q -o BatchMode=yes rover_agent.py "$PI:~/rover_agent.py" || fail "scp failed"
}
# Always refresh it: a stale agent on the Pi against a newer server is a confusing
# class of bug (wrong header fields, wrong colour order) that costs an hour to find.
scp -q -o BatchMode=yes rover_agent.py "$PI:~/rover_agent.py" || fail "scp failed"
echo "  agent    : up to date on the Pi"

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
while port_busy "$PORT"; do echo "  port $PORT busy, trying $((PORT+1))"; PORT=$((PORT+1)); done

# ---- shutdown: both ends, however we exit ------------------------------------------
cleanup() {
  trap - EXIT INT TERM
  echo; echo "stopping..."
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$PI" 'pkill -f "[r]over_agent" || true' 2>/dev/null || true
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export PYTORCH_ENABLE_MPS_FALLBACK=1

echo
echo "[1/2] perception server  ->  http://localhost:$PORT   (depth: $DEPTH)"
.venv/bin/python perception_server.py --source rover --depth "$DEPTH" \
    --depth-res 280 --profile --port "$PORT" ${EXTRA[@]+"${EXTRA[@]}"} &
SERVER=$!

# Models load for ~20-40 s on first run. Start the agent only once the socket is up,
# so its first frames are not dropped and its reconnect backoff never kicks in.
echo -n "      loading models"
for _ in $(seq 1 90); do
  kill -0 $SERVER 2>/dev/null || { echo; fail "server exited - see the error above"; }
  curl -s "localhost:$PORT/status" >/dev/null 2>&1 && break
  echo -n "."; sleep 2
done
echo " ready"

echo "[2/2] rover agent on the Pi  (fps $FPS, rotation $ROTATION)"
# Kill and launch MUST be two separate ssh calls. `pkill -f` matches against the whole
# command line, and the launch command line necessarily contains "rover_agent.py" - so a
# combined call has pkill kill its own remote shell, and ssh returns 255. The [r] trick
# only stops the *pattern* self-matching, which does not help here.
ssh -o BatchMode=yes -o ConnectTimeout=8 "$PI" 'pkill -f "[r]over_agent" || true' >/dev/null 2>&1 || true
sleep 1
ssh -o BatchMode=yes -o ConnectTimeout=8 "$PI" \
  "cd ~ && setsid nohup python3 rover_agent.py \
     --server ws://$MAC_IP:$PORT/ws --fps $FPS --rotation $ROTATION \
     >/tmp/rover_agent.log 2>&1 </dev/null & echo ok" >/dev/null 2>&1 \
  || fail "could not start the agent on the Pi"

sleep 3
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$PI" 'pgrep -f "[r]over_agent" >/dev/null'; then
  echo "  !! agent died on startup. Its log:"
  ssh -o BatchMode=yes "$PI" 'tail -20 /tmp/rover_agent.log' 2>/dev/null || true
  fail "agent not running"
fi
echo "      agent up"

cat <<EOF

  dashboard   http://localhost:$PORT
  costmap     http://localhost:$PORT/costmap     (the final output)
  camera      http://localhost:$PORT/camera      (live feed)
  overlay     http://localhost:$PORT/overlay     (+ semantic classes)
  depth       http://localhost:$PORT/depth
  all streams http://localhost:$PORT/streams

  Click the costmap or the global map to set a destination.
  Agent log on the Pi:  ssh $PI 'tail -f /tmp/rover_agent.log'
  Ctrl-C stops both ends.

EOF
[[ $OPEN_BROWSER -eq 1 ]] && command -v open >/dev/null && open "http://localhost:$PORT" || true
wait $SERVER
