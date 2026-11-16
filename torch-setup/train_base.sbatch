#!/usr/bin/env bash
# Train one flow-BC base policy: block-single-task1, visual obs, chunk-frame abs actions, H = C = 25
# (the RL action space chosen in experiments.yaml E2/E4).
#   sbatch --export=ALL,NDEMOS=330,STEPS=100000 train_base.sbatch
# Run dir: runs/base/base_block-single-task1_visual_abs-chunk_H25_n<NDEMOS>_<STEPS/1000>k/s0
# Near the time limit Slurm sends USR1; training checkpoints and requeues itself, then resumes.
#SBATCH --job-name=rlft-base
#SBATCH --account=torch_pr_50_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/aap9348/rlft-bench/logs/%x-%j.out
#SBATCH --error=/scratch/aap9348/rlft-bench/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test "$PWD" = /scratch/aap9348/rlft-bench
: "${NDEMOS:?pass NDEMOS via --export}" "${STEPS:?pass STEPS via --export}"
SEED=${SEED:-0}
name=base_block-single-task1_visual_abs-chunk_H25_n${NDEMOS}_$((STEPS / 1000))k
printf 'job_id=%s restart=%s host=%s start=%s\nname=%s seed=%s\n' "$SLURM_JOB_ID" "${SLURM_RESTART_COUNT:-0}" \
  "$(hostname)" "$(date --iso-8601=seconds)" "$name" "$SEED"
module load anaconda3/2025.06
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
git rev-parse HEAD
unset JAX_PLATFORMS
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=offline

# Forward Slurm's USR1 to the training process (uv may not pass it on).
signal_tree() { local p; for p in $(ps -o pid= --ppid "$1"); do signal_tree "$p"; kill -USR1 "$p" 2>/dev/null || true; done; }
trap 'echo "USR1 at $(date --iso-8601=seconds): asking training to checkpoint"; signal_tree $child' USR1

bash /home/aap9348/rlft-bench-torch/run.sh python scripts/train_base.py \
  env=ocb_block_single obs=visual action.space=abs action.abs_frame=chunk action.H=25 action.C=25 \
  data.num_demos="$NDEMOS" train.steps="$STEPS" train.eval_every=10000 env.eval_envs=128 \
  "train.ckpt_steps=[10000,25000,50000,100000]" seed="$SEED" name="$name" &
child=$!
while :; do
  set +e; wait $child; code=$?; set -e
  kill -0 $child 2>/dev/null || break   # wait returns early when the trap fires
done
printf 'exit=%s end=%s\n' "$code" "$(date --iso-8601=seconds)"
exit $code
