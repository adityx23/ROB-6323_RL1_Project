# Shared env for overnight jobs (source inside a compute allocation).
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
R=$RLFT_ROOT
export PY=$R/.venv/bin/python OVN=$PROJ/overnight RUNS=$R/runs/overnight
export HF_HOME=$R/cache/huggingface HF_LEROBOT_HOME=$R/cache/huggingface/lerobot HF_HUB_OFFLINE=1
export OPENPI_DATA_HOME=$RLFT_ROOT/cache/openpi TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache
export MPLCONFIGDIR=$R/cache/matplotlib JAX_COMPILATION_CACHE_DIR=$R/cache/jax TMPDIR=$R/tmp
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mkdir -p $RUNS
