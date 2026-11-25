#!/usr/bin/env bash
# pi0.5 full fine-tune on local/ocb_diverse_1k on ONE H200 (pi05_diverse.py: bs64, 15k steps, ckpts 5k/10k/15k).
# Resubmitting resumes from the newest checkpoint; exits at once if the final checkpoint exists.
#   ./submit.sh diverse/train_diverse.sbatch                       (gh133 reservation, see AGENTS.md)
#SBATCH --job-name=dv-pi05
#SBATCH --account=torch_pr_50_tandon_advanced
#SBATCH --reservation=ka3658-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
source $PROJ/overnight/env.sh
STEPS=${DV_STEPS:-15000}; LAST=$((STEPS - 1))
CK=$RLFT_ROOT/runs/diverse/checkpoints/pi05_ocbdiv1k/${DV_EXP:-run}
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p $CK
if [ -d $CK/$LAST/params ]; then echo "final checkpoint $LAST exists: training already finished"; exit 0; fi
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $RLFT_ROOT/logs/dv-gpu-$SLURM_JOB_ID.csv &
RES=""; ls $CK 2>/dev/null | grep -qE '^[0-9]+$' && RES=--resume
cd $PROJ/diverse
$PY pi05_diverse.py train $RES
echo "end=$(date -Is)"
