#!/usr/bin/env bash
# Download released OCBench block-single-task1 data (shard 000: visual + state) from the Hugging Face
# dataset seohongpark/ocbench into $RLFT_ROOT/data/ocbench. CPU only.
#SBATCH --job-name=rlft-download
#SBATCH --account=torch_pr_50_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test "$PWD" = $RLFT_ROOT
printf 'job_id=%s\nhost=%s\nstart=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"
module load anaconda3/2025.06
myquota
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1
data=$RLFT_ROOT/data/ocbench
mkdir -p "$data"
run=$PROJ/torch-setup/run.sh
bash $run python scripts/gen_data.py download --task block-single-task1 --visual --shards 1 --root "$data"
bash $run python scripts/gen_data.py download --task block-single-task1 --shards 1 --root "$data"
du -sh "$data"/visual "$data"/full
find "$data"/visual "$data"/full -type f -exec ls -l {} +
printf 'end=%s\n' "$(date --iso-8601=seconds)"
