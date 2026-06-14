#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$ENGINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m dailystonks.live_terminal "$@"
