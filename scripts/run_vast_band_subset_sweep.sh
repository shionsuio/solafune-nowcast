#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/solafune
COMMON=(
  --root "$ROOT"
  --folds 0
  --folds-json "$ROOT/outputs/cv_schemes/eval_aligned/folds.json"
  --epochs 6
  --batch-size 8
  --workers 8
  --lr-encoder 2e-5
  --lr-head 1e-4
  --loss-type huber
  --heavy-rain-weight-alpha 0.5
  --heavy-rain-weight-scale 10.0
  --heavy-rain-weight-max 2.0
  --use-two-head
  --rain-bce-weight-0-1 0.10
  --rain-bce-weight-1 0.10
  --rain-bce-weight-5 0.05
  --stats-samples-per-satellite 1500
  --seed 42
  --no-amp
  --align-frames
  --exclude-bad-labels
)

for mode in full14_btd vis13_btd ir12_btd core10_btd; do
  echo "=== START $mode ==="
  python src/run_swin_temporal_full.py \
    "${COMMON[@]}" \
    --band-mode "$mode" \
    --model-subdir "swin_twohead_eval_aligned_${mode}_f0"
done
