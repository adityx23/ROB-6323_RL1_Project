# Shared env for overnight jobs (source inside a compute allocation).
R=/scratch/aap9348/rlft-bench
export PY=$R/.venv/bin/python OVN=/home/aap9348/rlft-bench-torch/overnight RUNS=$R/runs/overnight
export HF_HOME=$R/cache/huggingface HF_LEROBOT_HOME=$R/cache/huggingface/lerobot HF_HUB_OFFLINE=1
export OPENPI_DATA_HOME=/scratch/aap9348/openpi/cache TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache
export MPLCONFIGDIR=$R/cache/matplotlib JAX_COMPILATION_CACHE_DIR=$R/cache/jax TMPDIR=$R/tmp
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_PY=$R/environments/libero-venv/bin/python LIBERO_SRC=/scratch/aap9348/lair-openpi/third_party/libero
mkdir -p $RUNS
# LIBERO dataset must be complete (1699 files incl. meta) before training on it.
libero_ready() { [ "$(find $HF_LEROBOT_HOME/physical-intelligence/libero -type f -not -path '*/.cache/*' | wc -l)" -ge 1690 ]; }
