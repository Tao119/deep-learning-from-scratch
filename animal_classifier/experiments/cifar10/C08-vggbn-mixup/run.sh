#!/usr/bin/env bash
set -e
BASE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
REPO_DIR="$(cd "$BASE_DIR/.." && pwd)"
OUT_DIR="$BASE_DIR/experiments/cifar10/C08-vggbn-mixup"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

echo "========== C08 VGGWithBN + Mixup =========="
PYTHONPATH=. python3 -u animal_classifier/train_cifar10_mixup.py \
  --mix mixup \
  --alpha 0.4 \
  --epochs 30 \
  --batch 256 \
  --decay_at 20 25 \
  --output_dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/train.log"

echo "C08 DONE"
