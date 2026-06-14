#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "Local venv not found; creating/installing first..."
  ./install_linux.sh
fi
. .venv/bin/activate
python -m dailystonks.live_terminal "$@"
