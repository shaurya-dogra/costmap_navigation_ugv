#!/usr/bin/env bash
# Stop everything the launchers start: perception servers, the Vite rover sim,
# and any headless Chrome test browser. Frees the default ports (8790.., 5173).
#
#   ./stop_all.sh
set -uo pipefail

stop() {   # stop <label> <pgrep pattern>
  local label="$1" pat="$2"
  local pids
  pids=$(pgrep -f "$pat" || true)
  if [[ -z "$pids" ]]; then
    echo "  $label: not running"
    return
  fi
  echo "  $label: stopping $(echo "$pids" | wc -l | tr -d ' ') process(es)"
  kill $pids 2>/dev/null || true
  sleep 1
  pids=$(pgrep -f "$pat" || true)
  [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
}

echo "stopping the demo..."
stop "perception server"      "perception_server.py"
stop "rover sim (vite)"        "vite --port|vite.js --port|node_modules/.bin/vite"
stop "legacy costmap windows"  "costmap_prototype.py"
stop "headless chrome (tests)" "remote-debugging-port=9222"

# anything still listening on our ports
for port in 8790 8791 8792 8793 8794 8795 5173; do
  pid=$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [[ -n "$pid" ]]; then
    cmd=$(ps -o comm= -p "$pid" 2>/dev/null || true)
    case "$cmd" in
      *python*|*node*|*Chrome*) echo "  port $port: killing $cmd ($pid)"; kill "$pid" 2>/dev/null || true ;;
      *) echo "  port $port: left alone ($cmd, not ours)" ;;
    esac
  fi
done
echo "done."
