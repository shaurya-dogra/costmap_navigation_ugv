#!/usr/bin/env bash
# Run this in macOS Terminal, NOT from the Claude device shell.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo
echo "done. now:"
echo "  source .venv/bin/activate"
echo "  export PYTORCH_ENABLE_MPS_FALLBACK=1"
echo "  python test_geometry.py          # must be ALL CHECKS PASSED"
echo "  python costmap_prototype.py --source 0"
