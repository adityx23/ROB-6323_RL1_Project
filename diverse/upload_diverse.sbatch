#!/usr/bin/env bash
# CPU job: replace the HF cube-stacking dataset (ocb-stack-3k -> ocb-stack-diverse-1k) with the diverse colored set.
#SBATCH --job-name=dv-upload
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
export HF_TOKEN=$(cat ${HF_TOKEN_FILE:-$HOME/.config/ldr-hf/token}) HF_HUB_OFFLINE=0
export HF_HOME=$RLFT_ROOT/cache/hf-upload HF_HUB_ENABLE_HF_TRANSFER=0
echo "start=$(date -Is) host=$(hostname)"
$RLFT_ROOT/.venv/bin/python -I $PROJ/diverse/upload_diverse_hf.py "$@"
echo "end=$(date -Is)"
