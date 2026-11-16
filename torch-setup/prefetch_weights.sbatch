#!/usr/bin/env bash
# CPU job: download the pi0.5 base weights (gs://openpi-assets/checkpoints/pi05_base, ~12 GB) and the PaliGemma
# tokenizer into $RLFT_ROOT/cache/openpi (OPENPI_DATA_HOME of overnight/env.sh), so GPU jobs never wait on downloads.
#SBATCH --job-name=rlft-prefetch
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
source $PROJ/overnight/env.sh
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=0
echo "start=$(date -Is) host=$(hostname)"
$PY - <<'E'
from openpi.shared import download
print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params"))
print(download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))
E
du -sh $OPENPI_DATA_HOME/*
echo "end=$(date -Is)"
