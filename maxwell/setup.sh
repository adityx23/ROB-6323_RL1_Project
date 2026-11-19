#!/usr/bin/env bash
# One-time setup on Maxwell (8x RTX A6000 48 GB), mirroring the Torch setup (torch-setup/setup.sbatch +
# prefetch_weights.sbatch): uv 0.9.5 -> CPython 3.11 -> `uv sync --frozen`, then the training data and norm stats
# from Hugging Face (adityx23/ocb-stack-3k, private) and the pi05_base weights + PaliGemma tokenizer from openpi's
# public bucket. Every step is skipped when its output already exists, so rerunning is safe.
# Needs: internet, git, python3 with pip, ~45 GB free in RLFT_ROOT, and an HF token with read access to the data
# repo (export HF_TOKEN=... or `hf auth login` beforehand).
#   source config.sh && bash maxwell/setup.sh
: "${PROJ:?source config.sh first}"
set -euo pipefail
DATA_REPO=${DATA_REPO:-adityx23/ocb-stack-3k}
cd "$RLFT_ROOT"
echo "start=$(date -Is) host=$(hostname) RLFT_ROOT=$RLFT_ROOT"
test "$(git rev-parse HEAD)" = "$RLFT_COMMIT" || { echo "rlft-bench is not at $RLFT_COMMIT" >&2; exit 1; }
test "$(git -C "$LAIR_OPENPI" rev-parse HEAD)" = "$LAIR_COMMIT" || { echo "lair-openpi is not at $LAIR_COMMIT" >&2; exit 1; }
test "$(cd "$LAIR_OPENPI" && pwd)" = "$(cd ../lair-openpi && pwd)" || { echo "lair-openpi must be ../lair-openpi" >&2; exit 1; }

export XDG_CACHE_HOME=$RLFT_ROOT/cache UV_CACHE_DIR=$RLFT_ROOT/cache/uv
export UV_PYTHON_INSTALL_DIR=$RLFT_ROOT/environments/python UV_PROJECT_ENVIRONMENT=$RLFT_ROOT/.venv
export PIP_CACHE_DIR=$RLFT_ROOT/cache/pip TMPDIR=$RLFT_ROOT/tmp GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$TMPDIR" logs

# 1. uv 0.9.5 + the frozen venv (same lockfile as Torch; never run an unfrozen uv sync / pip install in it)
bootstrap=$RLFT_ROOT/environments/uv-0.9.5
[ -x "$bootstrap/bin/uv" ] || python3 -m pip --isolated install --index-url https://pypi.org/simple \
  --cache-dir "$PIP_CACHE_DIR" --target "$bootstrap" 'uv==0.9.5'
export PATH=$bootstrap/bin:$PATH
uv --version
if [ ! -e .venv ]; then
  lock_before=$(sha256sum uv.lock)
  uv sync --frozen --python 3.11
  test "$(sha256sum uv.lock)" = "$lock_before"
fi
bash "$PROJ/torch-setup/check_env.sh"   # must print IDENTICAL and the expected commits

# 2. training data (LeRobot local/ocb_stack_3k) + the ladder's norm stats
source "$PROJ/overnight/env.sh"
export HF_HUB_OFFLINE=0
D=$HF_LEROBOT_HOME/local/ocb_stack_3k
NS=$RLFT_ROOT/runs/ladder/assets/pi05_ocbs3k/local/ocb_stack_3k
$PY - "$DATA_REPO" "$D" <<'E'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], repo_type="dataset", local_dir=sys.argv[2], max_workers=8))
E
n=$(find "$D/data" -name 'episode_*.parquet' | wc -l)
test "$n" -eq 3000 || { echo "expected 3000 episode files, found $n: rerun to resume the download" >&2; exit 1; }
mkdir -p "$NS"; cp "$D/norm_stats/pi05_ocbs3k/local/ocb_stack_3k/norm_stats.json" "$NS/"
echo "data OK: $n episodes in $D; norm stats in $NS"

# 3. pi05_base weights + PaliGemma tokenizer -> $OPENPI_DATA_HOME
JAX_PLATFORMS=cpu $PY - <<'E'
from openpi.shared import download
print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params"))
print(download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))
E

# 4. GPU check (on a GPU node: expect 8 CUDA devices)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "no nvidia-smi here: run step 4 on a GPU node"
$PY -c "import jax; print('jax devices:', jax.devices())" || true
echo "end=$(date -Is)"
