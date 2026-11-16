#!/usr/bin/env bash
# CPU watcher: submits eval_ladder.sbatch (STEP=s) as soon as each ladder checkpoint is finalized, until the
# training job reaches a terminal state and every present ladder step has been submitted.
#   sbatch --dependency=afterok:<export> eval_watcher.slurm <train_jobid>
#SBATCH --job-name=ldr-watch
#SBATCH --account=torch_pr_50_tandon_priority
#SBATCH --partition=cs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -o pipefail
TRAIN=$1; L=$PROJ/ladder
CK=$RLFT_ROOT/runs/ladder/checkpoints/pi05_ocbs3k/run
declare -A DONE=()
while true; do
  st=$(sacct -n -X -j $TRAIN --format=State%40 | head -1 | tr -d ' ')
  for s in $(cat $L/ladder_steps.txt); do
    [ -n "${DONE[$s]:-}" ] && continue
    [ -d $CK/$s/params ] || continue
    j=$(sbatch --parsable --export=ALL,STEP=$s $L/eval_ladder.sbatch) && DONE[$s]=$j && echo "$(date -Is) step $s -> eval job $j" | tee -a $L/jobs.txt
  done
  case $st in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
    echo "$(date -Is) train $TRAIN $st; submitted ${#DONE[@]} evals"; break;; esac
  sleep 300
done
