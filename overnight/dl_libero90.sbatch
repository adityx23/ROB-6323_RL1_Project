#!/usr/bin/env bash
# Download the 4 LIBERO-90 stacking demo files (50 demos each, HDF5).
#SBATCH --job-name=ovn-dl-l90
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -uo pipefail
R=$RLFT_ROOT
export HF_HOME=$R/cache/huggingface
for i in $(seq 1 20); do
  $R/.venv/bin/python - <<'E' && { echo "L90 DONE"; break; }
from huggingface_hub import snapshot_download
snapshot_download("yifengzhu-hf/LIBERO-datasets", repo_type="dataset", allow_patterns=["libero_90/*stack*"],
                  local_dir="$RLFT_ROOT/data/libero90_raw", max_workers=4)
E
  echo "attempt $i failed $(date -Is)"; sleep $((30*i))
done
ls -la $RLFT_ROOT/data/libero90_raw/libero_90
