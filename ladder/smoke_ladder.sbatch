#!/usr/bin/env bash
# Smoke test of the full ladder config on 4 GPUs: 60 steps at batch 128 (checkpoint at 50 and 59), GPU memory log,
# then load the final smoke checkpoint as a policy and run one inference. Deletes the smoke checkpoints afterwards.
#SBATCH --job-name=ldr-smoke
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100|h100
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
source $PROJ/overnight/env.sh
L=$PROJ/ladder
CK=$RLFT_ROOT/runs/ladder/checkpoints/pi05_ocbs3k/smoke
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 10 > $RLFT_ROOT/logs/ldr-smoke-gpu-$SLURM_JOB_ID.csv &
GP=$!
cd $L
LDR_EXP=smoke LDR_STEPS=60 LDR_SAVE=50 LDR_KEEP=50 $PY pi05_ladder.py train
echo "train done $(date -Is)"; ls $CK
LAST=$(ls $CK | grep -E '^[0-9]+$' | sort -n | tail -1)
du -sh $CK/$LAST/* || true
CUDA_VISIBLE_DEVICES=0 $PY pi05_ladder.py infer $CK/$LAST
kill $GP 2>/dev/null || true
echo "peak GPU memory (MiB): $(awk -F', ' '{gsub(/ MiB/,"",$4); if ($4>m) m=$4} END {print m}' $RLFT_ROOT/logs/ldr-smoke-gpu-$SLURM_JOB_ID.csv)"
rm -rf $CK
echo "end=$(date -Is)"
