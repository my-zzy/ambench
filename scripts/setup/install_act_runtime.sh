#!/usr/bin/env bash
set -euo pipefail

# Activate the simulator's Python 3.11 environment before running this installer.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_DIR="$PROJECT_ROOT/outputs/act_runtime/python_deps"
export UV_CACHE_DIR="$PROJECT_ROOT/outputs/act_runtime/cache/uv"
cd "$PROJECT_ROOT"

uv pip install --python "$(command -v python)" --target "$TARGET_DIR" \
  --no-deps --link-mode copy -r experiments/pressbutton/requirements-act.txt
