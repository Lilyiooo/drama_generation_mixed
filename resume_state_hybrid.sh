#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" tools/resume_state_hybrid.py "$@"
