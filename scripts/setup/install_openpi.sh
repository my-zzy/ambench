#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${1:?Pass a dedicated artifact/environment directory, e.g. outputs/openpi}"
cd "$PROJECT_ROOT"
WORK_DIR="$(realpath -m "$WORK_DIR")"
export UV_PROJECT_ENVIRONMENT="$WORK_DIR/.venv"
export UV_CACHE_DIR="$WORK_DIR/cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORK_DIR/cache/python"
export UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1

uv sync --project ext/openpi --frozen --no-dev --python 3.11
uv pip install --python "$WORK_DIR/.venv/bin/python" --no-deps -e source/ambench_learn
