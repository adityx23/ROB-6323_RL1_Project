#!/usr/bin/env bash
# Run inside an existing, approved compute allocation; does not allocate resources.
set -euo pipefail
case $(hostname) in torch-login-*) echo 'Run workloads in an approved compute allocation' >&2; exit 1;; esac
: "${SLURM_JOB_ID:?Compute allocation required}"
test "${SLURMD_NODENAME:-}" = "$(hostname -s)" || {
  echo 'Hostname does not match the allocated SLURM node; stop for inspection' >&2; exit 1;
}
test "$#" -gt 0 || { echo 'Usage: bash run.sh python SCRIPT [ARGS...]' >&2; exit 1; }
export RLFT_ROOT=/scratch/aap9348/rlft-bench
cd "$RLFT_ROOT"
export UV_PROJECT_ENVIRONMENT=$RLFT_ROOT/.venv
export UV_CACHE_DIR=$RLFT_ROOT/cache/uv UV_PYTHON_INSTALL_DIR=$RLFT_ROOT/environments/python
export XDG_CACHE_HOME=$RLFT_ROOT/cache PIP_CACHE_DIR=$RLFT_ROOT/cache/pip
export HF_HOME=$RLFT_ROOT/cache/huggingface TORCH_HOME=$RLFT_ROOT/cache/torch
export OPENPI_DATA_HOME=$RLFT_ROOT/cache/openpi MPLCONFIGDIR=$RLFT_ROOT/cache/matplotlib
export JAX_COMPILATION_CACHE_DIR=$RLFT_ROOT/cache/jax TMPDIR=$RLFT_ROOT/tmp
export RLFT_DATA=$RLFT_ROOT/data/ocbench RLFT_RUNS=$RLFT_ROOT/runs WANDB_DIR=$RLFT_ROOT/runs/wandb
threads=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads
export NUMEXPR_NUM_THREADS=$threads VECLIB_MAXIMUM_THREADS=$threads
export TF_NUM_INTRAOP_THREADS=$threads TF_NUM_INTEROP_THREADS=1 RLFT_NUM_THREADS=$threads
export RLFT_LOADER_WORKERS=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$threads ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GIT_LFS_SKIP_SMUDGE=1
export PATH=$RLFT_ROOT/environments/uv-0.9.5/bin:$PATH
exec uv run --frozen --no-sync "$@"
