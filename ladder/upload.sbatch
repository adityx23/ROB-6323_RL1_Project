#!/usr/bin/env bash
# CPU job: upload the ladder checkpoints (params + assets) to a public HF repo. Resumable; rerun if it stops.
#SBATCH --job-name=ldr-upload
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
export HF_TOKEN=$(cat ${HF_TOKEN_FILE:-$HOME/.config/ldr-hf/token}) HF_HUB_OFFLINE=0
export HF_HOME=$RLFT_ROOT/cache/hf-upload HF_HUB_ENABLE_HF_TRANSFER=0
echo "start=$(date -Is) host=$(hostname)"
$RLFT_ROOT/.venv/bin/python -I $PROJ/ladder/upload_hf.py "$@"
echo "end=$(date -Is)"
