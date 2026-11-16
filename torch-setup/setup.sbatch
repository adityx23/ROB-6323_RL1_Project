#!/usr/bin/env bash
#SBATCH --job-name=rlft-setup
#SBATCH --account=torch_pr_50_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
printf 'job_id=%s\njob_name=%s\naccount=%s\nhost=%s\nnodes=%s\nstart=%s\nworking_directory=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NAME" "${SLURM_JOB_ACCOUNT:-unknown}" "$(hostname)" \
  "$SLURM_JOB_NODELIST" "$(date --iso-8601=seconds)" "$PWD"
case $(hostname) in torch-login-*) echo 'Refusing installation on a login node' >&2; exit 1;; esac
: "${SLURM_JOB_ID:?Compute allocation required}"
test "${SLURMD_NODENAME:-}" = "$(hostname -s)" || {
  echo 'Hostname does not match the allocated SLURM node; stop for inspection' >&2; exit 1;
}
test "$PWD" = $RLFT_ROOT
test "$(git rev-parse HEAD)" = "$RLFT_COMMIT"
test "$(git -C "$LAIR_OPENPI" rev-parse HEAD)" = "$LAIR_COMMIT"
test "$(cd "$LAIR_OPENPI" && pwd)" = "$(cd ../lair-openpi && pwd)"
test ! -e .venv || { echo 'Existing .venv: inspect before installing' >&2; exit 1; }

export RLFT_ROOT=$RLFT_ROOT
export XDG_CACHE_HOME=$RLFT_ROOT/cache
export UV_CACHE_DIR=$RLFT_ROOT/cache/uv
export UV_PYTHON_INSTALL_DIR=$RLFT_ROOT/environments/python
export UV_PROJECT_ENVIRONMENT=$RLFT_ROOT/.venv
export PIP_CACHE_DIR=$RLFT_ROOT/cache/pip
export HF_HOME=$RLFT_ROOT/cache/huggingface
export TORCH_HOME=$RLFT_ROOT/cache/torch
export OPENPI_DATA_HOME=$RLFT_ROOT/cache/openpi
export MPLCONFIGDIR=$RLFT_ROOT/cache/matplotlib
export TMPDIR=$RLFT_ROOT/tmp
export GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= WANDB_MODE=disabled
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$HF_HOME" \
  "$TORCH_HOME" "$OPENPI_DATA_HOME" "$MPLCONFIGDIR" "$TMPDIR"
myquota
module load anaconda3/2025.06
module list
python --version
bootstrap=$RLFT_ROOT/environments/uv-0.9.5
test ! -e "$bootstrap" || { echo 'Existing uv bootstrap: inspect before installing' >&2; exit 1; }
python -m pip --isolated install --index-url https://pypi.org/simple \
  --cache-dir "$PIP_CACHE_DIR" --target "$bootstrap" 'uv==0.9.5'
export PATH=$bootstrap/bin:$PATH
uv --version
git rev-parse HEAD
git -C "$LAIR_OPENPI" rev-parse HEAD
sha256sum uv.lock pyproject.toml ../lair-openpi/pyproject.toml
lock_before=$(sha256sum uv.lock)
printf '%s\n' 'command=GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11' 'inputs=uv.lock; sibling lair-openpi; no dataset or model weights' 'seed=not applicable (installation/import checks)'
uv sync --frozen --python 3.11
test "$(sha256sum uv.lock)" = "$lock_before"
uv run --frozen --no-sync python - <<'PY'
from importlib.metadata import version
import jax
import torch
import mujoco
import mujoco_warp
import ocbench
import openpi
for package in ('jax', 'torch', 'mujoco', 'mujoco-warp', 'ocbench', 'openpi'):
    print(f'{package}={version(package)}')
print('CPU devices:', jax.devices())
print('PASS: CPU import check; GPU simulation and rendering remain untested')
PY
printf 'end=%s\n' "$(date --iso-8601=seconds)"
