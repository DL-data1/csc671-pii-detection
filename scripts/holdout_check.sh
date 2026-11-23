#!/usr/bin/env bash
# Issue #8 generalization check: drop prompt style 0/4 (templates 0 and 4 are
# variants of the same prompt) from train and val, then score the best
# checkpoint once on the 628 train+val docs of that style. Used to choose
# between cased and uncased; test is never touched.
#
# Usage (from repo root): bash scripts/holdout_check.sh [python] [log-dir]
PY=${1:-.venv/Scripts/python.exe}
LOGDIR=${2:-logs}
mkdir -p "$LOGDIR"
export PYTHONIOENCODING=utf-8
run() {
  name=$1; shift
  echo "=== $name: $* ($(date +%H:%M:%S))"
  "$PY" scripts/train.py --run-name "$name" --holdout-prompt-ids 0 4 --output "models/holdout/$name" "$@" > "$LOGDIR/$name.log" 2>&1
  echo "    exit $? $(grep -a 'Best val F1' "$LOGDIR/$name.log")"
  "$PY" -c "import json,sys; h=json.load(open('docs/results/$name.json'))['holdout']; print('    holdout F1', h['f1'], 'P', h['precision'], 'R', h['recall'])"
}
run holdout04_cased_ep5
run holdout04_cased_ep8 --epochs 8
run holdout04_uncased --model-name distilbert-base-uncased
echo "=== done ($(date +%H:%M:%S))"
