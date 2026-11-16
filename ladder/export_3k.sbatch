#!/usr/bin/env bash
# LeRobot export of 3000 successful stacking demos -> local/ocb_stack_3k.
# CPU node (cs): export_3k.py switches MuJoCo to glfw, so no GPU/EGL is needed. (A GPU run, 23152732, was cancelled by
# root after 2 h, most likely for holding an idle GPU.)
# First re-derives the first 3 episodes of the overnight local/ocb_stack and checks they match, then exports.
#SBATCH --job-name=ldr-export
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cs
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_LEROBOT_HOME=$RLFT_ROOT/cache/huggingface/lerobot
E=$PROJ/ladder/export_3k.py
echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
bash $PROJ/torch-setup/run.sh python $E check --episodes 3
echo "check passed $(date -Is)"
bash $PROJ/torch-setup/run.sh python $E export --num ${NUM:-3000}
echo "end=$(date -Is)"
