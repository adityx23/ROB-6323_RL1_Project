#!/usr/bin/env bash
# Compare the installed venvs against the frozen references (run anywhere after `source config.sh`).
#   bash torch-setup/check_env.sh            -> prints "IDENTICAL" or a diff per venv, exit 1 on any difference
set -uo pipefail
: "${PROJ:?source config.sh first}"
UV=$RLFT_ROOT/environments/uv-0.9.5/bin/uv
rc=0
norm() { grep -v '^#' | sed "s|file://$LAIR_OPENPI|file://\$LAIR_OPENPI|; s|file://$RLFT_ROOT|file://\$RLFT_ROOT|" | sort; }
cmp_env() {  # name python reference [skip-editable]
  local got; got=$($UV pip freeze --python "$2" 2>/dev/null | norm)
  local want; want=$(norm < "$3")
  [ "${4:-}" = noedit ] && got=$(echo "$got" | grep -v '^-e ')
  if [ "$got" = "$want" ]; then echo "$1: IDENTICAL ($(echo "$want" | wc -l) packages)"
  else echo "$1: DIFFERS"; diff <(echo "$want") <(echo "$got") | sed 's/^/  /'; rc=1; fi
}
cmp_env rlft-bench-venv "$RLFT_ROOT/.venv/bin/python" "$PROJ/torch-setup/rlft_venv_freeze.txt"
echo "rlft-bench  $(git -C "$RLFT_ROOT" rev-parse HEAD)  expected $RLFT_COMMIT"
echo "lair-openpi $(git -C "$LAIR_OPENPI" rev-parse HEAD)  expected $LAIR_COMMIT"
[ "$(git -C "$RLFT_ROOT" rev-parse HEAD)" = "$RLFT_COMMIT" ] && [ "$(git -C "$LAIR_OPENPI" rev-parse HEAD)" = "$LAIR_COMMIT" ] || rc=1
exit $rc
