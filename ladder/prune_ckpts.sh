#!/usr/bin/env bash
# Keep the checkpoint ladder small: every step except the newest finalized one is either reduced to params+assets
# (ladder steps, enough to serve/eval) or deleted. The newest stays complete so training can resume.
#   prune_ckpts.sh <ckpt_run_dir> [loop]
: "${PROJ:?submit with ./submit.sh (or source config.sh first)}"
set -u
D=$1; LADDER=" $(cat $PROJ/ladder/ladder_steps.txt) "
prune() {
  steps=$(ls "$D" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
  latest=$(echo "$steps" | tail -1)
  for s in $steps; do
    [ "$s" = "$latest" ] && continue
    if [[ "$LADDER" == *" $s "* ]]; then
      [ -d "$D/$s/train_state" ] && { rm -rf "$D/$s/train_state"; echo "$(date -Is) ladder $s: dropped train_state"; }
    else
      rm -rf "$D/$s"; echo "$(date -Is) deleted $s"
    fi
  done
}
if [ "${2:-}" = loop ]; then while true; do prune; sleep 120; done; else prune; fi
