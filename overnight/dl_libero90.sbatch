#!/usr/bin/env bash
# Download the 4 LIBERO-90 stacking demo files (50 demos each, HDF5).
#SBATCH --job-name=ovn-dl-l90
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err
set -uo pipefail
R=/scratch/aap9348/rlft-bench
export HF_HOME=$R/cache/huggingface
for i in $(seq 1 20); do
  $R/.venv/bin/python - <<'E' && { echo "L90 DONE"; break; }
from huggingface_hub import snapshot_download
snapshot_download("yifengzhu-hf/LIBERO-datasets", repo_type="dataset", allow_patterns=["libero_90/*stack*"],
                  local_dir="/scratch/aap9348/rlft-bench/data/libero90_raw", max_workers=4)
E
  echo "attempt $i failed $(date -Is)"; sleep $((30*i))
done
ls -la /scratch/aap9348/rlft-bench/data/libero90_raw/libero_90
