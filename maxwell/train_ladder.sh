#!/usr/bin/env bash
# pi0.5 ladder on Maxwell (8x RTX A6000 48 GB). Same config as ladder/train_ladder.sbatch on Torch (H=30, RTC,
# batch 128, EMA 0.999, 30k steps, same checkpoints); only the compute profile differs: LDR_CLUSTER=maxwell =
# 8-GPU FSDP, 16 per GPU. Run maxwell/setup.sh first. Works with or without Slurm:
#   source config.sh && nohup bash maxwell/train_ladder.sh > $RLFT_ROOT/logs/ldr-maxwell.log 2>&1 &
#   source config.sh && cd $RLFT_ROOT && sbatch --export=ALL [--account/--partition Maxwell needs] $PROJ/maxwell/train_ladder.sh
# Rerunning resumes from the newest checkpoint; exits at once if the final checkpoint (29999) exists.
# GPU utilization is logged every 30 s to $RLFT_ROOT/logs/ldr-maxwell-gpu.csv.
#SBATCH --job-name=ldr-pi05-mx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?source config.sh first}"
set -euo pipefail
source "$PROJ/overnight/env.sh"
export LDR_CLUSTER=maxwell
L=$PROJ/ladder
CK=$RLFT_ROOT/runs/ladder/checkpoints/pi05_ocbs3k/run
NS=$RLFT_ROOT/runs/ladder/assets/pi05_ocbs3k/local/ocb_stack_3k/norm_stats.json
mkdir -p "$CK" "$RLFT_ROOT/logs"
echo "start=$(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
need=${LDR_FSDP:-8}; have=$($PY -c "import jax; print(jax.device_count())")
[ "$have" -ge "$need" ] && [ $((have % need)) -eq 0 ] || { echo "JAX sees $have GPUs, need a multiple of $need" >&2; exit 1; }
test -f "$NS" || { echo "missing norm stats $NS: run maxwell/setup.sh" >&2; exit 1; }
if [ -d "$CK/29999/params" ]; then echo "final checkpoint 29999 exists: training already finished"; exit 0; fi
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 30 \
  >> "$RLFT_ROOT/logs/ldr-maxwell-gpu.csv" &
bash "$L/prune_ckpts.sh" "$CK" loop >> "$RLFT_ROOT/logs/ldr-prune.log" 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT
RES=""; ls "$CK" 2>/dev/null | grep -qE '^[0-9]+$' && RES=--resume
cd "$L"
$PY pi05_ladder.py train $RES
sleep 150; bash "$L/prune_ckpts.sh" "$CK" >> "$RLFT_ROOT/logs/ldr-prune.log" 2>&1
echo "end=$(date -Is)"
