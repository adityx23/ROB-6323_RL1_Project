#!/usr/bin/env bash
# Test the new binary_advc method (branch binary-advc) and finish GPU verification:
#   1. tests/test_binary_advc.py (new unit tests)
#   2. full unit suite (released data now present, so the dataset tests run instead of skipping)
#   3. RL smoke for residual, rlpd (not reached in job 21746503) and binary_advc, state + visual
# SLURM_JOB_ID is withheld from pytest so no test can ask Slurm to requeue this job.
#SBATCH --job-name=rlft-test-advc
#SBATCH --account=torch_pr_50_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test "$PWD" = $RLFT_ROOT
test "$(git rev-parse --abbrev-ref HEAD)" = binary-advc
printf 'job_id=%s host=%s start=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"
module load anaconda3/2025.06
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
git status --short
unset JAX_PLATFORMS
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
out=$RLFT_ROOT/runs/verification/advc-$SLURM_JOB_ID
mkdir -p "$out"
run=$PROJ/torch-setup/run.sh
py() { bash $run env -u SLURM_JOB_ID python "$@"; }
common=(-m pytest -q -rA -o "cache_dir=$out/pytest-cache")

status=0
stage() {  # stage <label> <args...>: log to $out/<label>.log, keep going, remember failures
  local label=$1; shift
  local t0=$SECONDS
  set +e; py "$@" > "$out/$label.log" 2>&1; local code=$?; set -e
  echo "stage=$label exit=$code seconds=$((SECONDS - t0))" | tee -a "$out/stages.txt"
  tail -3 "$out/$label.log"
  [ $code -eq 0 ] || status=1
}
stage advc-units "${common[@]}" --basetemp="$out/advc-tmp" tests/test_binary_advc.py
stage units "${common[@]}" --basetemp="$out/units-tmp" tests --ignore=tests/smoke
RLFT_SMOKE_METHODS=residual,rlpd,binary_advc stage rl-smoke "${common[@]}" --basetemp="$out/rl-tmp" \
  tests/smoke/test_rl_smoke.py
printf 'status=%s end=%s\n' "$status" "$(date --iso-8601=seconds)"
exit $status
