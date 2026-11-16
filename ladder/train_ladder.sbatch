#!/usr/bin/env bash
# pi0.5 ladder on local/ocb_stack_3k (pi05_ladder.py: H=30, RTC, bs128, EMA, 30k steps), 4-GPU FSDP.
# Checkpoints every 2500 steps; orbax keeps multiples of 5000; prune_ckpts.sh drops train_state from ladder steps.
# Resubmitting resumes from the newest checkpoint.
#SBATCH --job-name=ldr-pi05
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100|h100
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err
set -euo pipefail
source /home/aap9348/rlft-bench-torch/overnight/env.sh
L=/home/aap9348/rlft-bench-torch/ladder
CK=/scratch/aap9348/rlft-bench/runs/ladder/checkpoints/pi05_ocbs3k/run
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p $CK
bash $L/prune_ckpts.sh $CK loop >> /scratch/aap9348/rlft-bench/logs/ldr-prune.log 2>&1 &
RES=""; ls $CK 2>/dev/null | grep -qE '^[0-9]+$' && RES=--resume
cd $L
$PY pi05_ladder.py train $RES
sleep 150; bash $L/prune_ckpts.sh $CK >> /scratch/aap9348/rlft-bench/logs/ldr-prune.log 2>&1
echo "end=$(date -Is)"
