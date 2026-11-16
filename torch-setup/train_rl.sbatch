#!/usr/bin/env bash
# Template: one RL fine-tuning run on top of a trained base (not submitted yet; methods still to decide).
#   sbatch --export=ALL,BASE=runs/base/<name>/s0,METHOD=dsrl_na[,SEED=0][,EXTRA="k=v k=v"] train_rl.sbatch
# Run dir: runs/rl/<method>_block-single-task1_visual_<base tag>/s<SEED> (deterministic, so requeues resume).
#SBATCH --job-name=rlft-rl
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
: "${BASE:?pass BASE via --export}" "${METHOD:?pass METHOD via --export}"
test -d "$BASE/policy" || { echo "Base run has no policy/: $BASE" >&2; exit 1; }
SEED=${SEED:-0}
tag=$(basename "$(dirname "$BASE")"); tag=${tag#base_block-single-task1_visual_}
name=${METHOD}_block-single-task1_visual_${tag}
printf 'job_id=%s restart=%s host=%s start=%s\nname=%s seed=%s base=%s extra=%s\n' "$SLURM_JOB_ID" \
  "${SLURM_RESTART_COUNT:-0}" "$(hostname)" "$(date --iso-8601=seconds)" "$name" "$SEED" "$BASE" "${EXTRA:-}"
module load anaconda3/2025.06
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
git rev-parse HEAD
unset JAX_PLATFORMS
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=offline

signal_tree() { local p; for p in $(ps -o pid= --ppid "$1"); do signal_tree "$p"; kill -USR1 "$p" 2>/dev/null || true; done; }
trap 'echo "USR1 at $(date --iso-8601=seconds): asking training to checkpoint"; signal_tree $child' USR1

# shellcheck disable=SC2086  # EXTRA is a space-separated list of Hydra overrides
bash /home/aap9348/rlft-bench-torch/run.sh python scripts/train_rl.py \
  env=ocb_block_single obs=visual action.space=abs action.abs_frame=chunk action.H=25 action.C=25 \
  base.run="$BASE" method="$METHOD" seed="$SEED" name="$name" ${EXTRA:-} &
child=$!
while :; do
  set +e; wait $child; code=$?; set -e
  kill -0 $child 2>/dev/null || break
done
printf 'exit=%s end=%s\n' "$code" "$(date --iso-8601=seconds)"
exit $code
