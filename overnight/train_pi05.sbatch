#!/usr/bin/env bash
# pi0.5 fine-tune, shared recipe. sbatch --export=ALL,NAME=pi05_l90|pi05_ocbs train_pi05.sbatch
#SBATCH --job-name=ovn-pi05
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100|h100
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err
set -euo pipefail
source /home/aap9348/rlft-bench-torch/overnight/env.sh
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID name=$NAME gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd $OVN
RES=""; ls $RUNS/checkpoints/$NAME/run 2>/dev/null | grep -q '^[0-9]' && RES=--resume
OVN_FSDP=2 $PY ovn_pi05.py train $NAME $RES
echo "end=$(date -Is)"
