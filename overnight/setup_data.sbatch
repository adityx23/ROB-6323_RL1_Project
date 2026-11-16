#!/usr/bin/env bash
# Overnight run, step 0 (CPU): download LIBERO LeRobot dataset + released pi05_libero checkpoint,
# check out the LIBERO sim and build its py3.8 venv. Each part logs DONE/FAILED so later steps can check.
#SBATCH --job-name=ovn-setup
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err

set -uo pipefail
R=/scratch/aap9348/rlft-bench
LO=/scratch/aap9348/lair-openpi
PY=$R/.venv/bin/python
export HF_HOME=$R/cache/huggingface HF_LEROBOT_HOME=$R/cache/huggingface/lerobot
export OPENPI_DATA_HOME=/scratch/aap9348/openpi/cache
export UV_CACHE_DIR=$R/cache/uv UV_PYTHON_INSTALL_DIR=$R/environments/python TMPDIR=$R/tmp
export PATH=$R/environments/uv-0.9.5/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=0
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID"

( $PY - <<'E'
from huggingface_hub import snapshot_download
import os
p = snapshot_download("physical-intelligence/libero", repo_type="dataset",
                      local_dir=os.environ["HF_LEROBOT_HOME"] + "/physical-intelligence/libero", max_workers=8)
print("libero dataset at", p)
E
) && echo "LIBERO_DATA DONE" || echo "LIBERO_DATA FAILED" &

( $PY -c "from openpi.shared import download as d; print(d.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))" \
  && echo "PI05_LIBERO_CKPT DONE" || echo "PI05_LIBERO_CKPT FAILED" ) &

( set -e
  cd $LO
  git submodule update --init --depth 1 third_party/libero
  V=$R/environments/libero-venv
  [ -x $V/bin/python ] || uv venv --python 3.8 $V
  uv pip install --python $V/bin/python -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
  uv pip install --python $V/bin/python -e packages/openpi-client -e third_party/libero
  MUJOCO_GL=egl PYTHONPATH=$LO/third_party/libero $V/bin/python -c "from libero.libero import benchmark; print(list(benchmark.get_benchmark_dict()))"
) && echo "LIBERO_SIM DONE" || echo "LIBERO_SIM FAILED" &

wait
du -sh $HF_LEROBOT_HOME/physical-intelligence/libero $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero 2>/dev/null
echo "end=$(date -Is)"
