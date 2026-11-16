#!/usr/bin/env bash
# Retry LIBERO LeRobot dataset download (resumable) with few workers + backoff on HF 429s.
#SBATCH --job-name=ovn-dl-libero
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -uo pipefail
R=$RLFT_ROOT
export HF_HOME=$R/cache/huggingface HF_LEROBOT_HOME=$R/cache/huggingface/lerobot
for f in ~/.cache/huggingface/token ~/.hf_token; do [ -s $f ] && export HF_TOKEN=$(tr -d "[:space:]" < $f) && echo "using token from $f" && break; done
for i in $(seq 1 30); do
  $R/.venv/bin/python - <<'E' && { echo "LIBERO_DATA DONE"; break; }
import os
from huggingface_hub import snapshot_download
snapshot_download("physical-intelligence/libero", repo_type="dataset",
                  local_dir=os.environ["HF_LEROBOT_HOME"] + "/physical-intelligence/libero", max_workers=2)
E
  echo "attempt $i failed $(date -Is); sleeping"; sleep $((60 + 30*i))
done
du -sh $HF_LEROBOT_HOME/physical-intelligence/libero; find $HF_LEROBOT_HOME/physical-intelligence/libero -type f | wc -l
