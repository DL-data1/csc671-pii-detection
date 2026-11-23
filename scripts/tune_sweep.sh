#!/usr/bin/env bash
# Issue #8 tuning sweep, round 1. Runs sequentially on one GPU; a failed run
# does not stop the rest. Each run writes docs/results/<run-name>.json.
# Baseline (seed 42) is docs/results/full_baseline.json from #5.
#
# Usage (from repo root): bash scripts/tune_sweep.sh [python] [log-dir]

PY=${1:-.venv/Scripts/python.exe}
LOGDIR=${2:-logs}
mkdir -p "$LOGDIR"
export PYTHONIOENCODING=utf-8

run() {
  name=$1; shift
  echo "=== $name: $* ($(date +%H:%M:%S))"
  "$PY" scripts/train.py --run-name "$name" "$@" > "$LOGDIR/$name.log" 2>&1
  echo "    exit $? ($(date +%H:%M:%S)) $(grep -a 'Best val F1' "$LOGDIR/$name.log")"
}

# Seed variance of the baseline config (noise floor)
run base_s43 --seed 43
run base_s44 --seed 44
# Optimisation
run ep8 --epochs 8
run lr3e-5 --lr 3e-5
run lr8e-5 --lr 8e-5
run bs16 --batch-size 16
# Class imbalance
run cw_sqrt_cap10 --class-weights sqrt --weight-cap 10
run cw_sqrt_cap50 --class-weights sqrt --weight-cap 50
# Tokenizer casing
run uncased --model-name distilbert-base-uncased
echo "=== sweep done ($(date +%H:%M:%S))"
