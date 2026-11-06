#!/usr/bin/env bash
set -euo pipefail

# Create the dedicated Python 3.11 Conda prefix before invoking this installer.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PREFIX="$(realpath "${1:?Pass the dedicated Conda environment prefix}")"
WORK_DIR="$(realpath -m "${2:?Pass the artifact directory}")"
# Direct mirror for this server workflow; the frozen lock's hashes remain authoritative.
PYPI_PACKAGES_URL="https://mirrors.aliyun.com/pypi/packages/"
export UV_CACHE_DIR="$WORK_DIR/cache/uv"
export UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1
export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=

test -d "$ENV_PREFIX/conda-meta"
"$ENV_PREFIX/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
mkdir -p "$WORK_DIR/server_setup"
cd "$PROJECT_ROOT/ext/openpi"
# Preserve exact artifact URLs and hashes, including packages missing from indexes.
# Unlike uv sync, this does not replace the Conda environment with a virtualenv.
uv export --frozen --no-dev --format pylock.toml \
    --output-file "$WORK_DIR/server_setup/pylock.toml" > "$WORK_DIR/server_setup/export-pylock.log"
"$ENV_PREFIX/bin/python" - "$WORK_DIR/server_setup" "$PYPI_PACKAGES_URL" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
text = (root / "pylock.toml").read_text()
(root / "pylock.direct.toml").write_text(text.replace("https://files.pythonhosted.org/packages/", sys.argv[2]))
PY
uv pip install --python "$ENV_PREFIX/bin/python" --no-deps \
    -r "$WORK_DIR/server_setup/pylock.direct.toml"
# The distribution metadata targets Isaac/ACT and requires a different Gymnasium.
# Register the learning source for this training-only environment instead.
"$ENV_PREFIX/bin/python" - "$PROJECT_ROOT/source/ambench_learn" <<'PY'
import sys
import sysconfig
from pathlib import Path

(Path(sysconfig.get_path("purelib")) / "ambench_learn_source.pth").write_text(sys.argv[1] + "\n")
PY
uv pip check --python "$ENV_PREFIX/bin/python"
uv pip freeze --python "$ENV_PREFIX/bin/python" > "$WORK_DIR/server_setup/installed-requirements.txt"
conda list --prefix "$ENV_PREFIX" --explicit > "$WORK_DIR/server_setup/conda-explicit.txt"
