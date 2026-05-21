#!/usr/bin/env bash
# CIFAR-10実験チェーン: C01完了後にC02→C03→C04を順番に実行
set -e
BASE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="$BASE_DIR/animal_classifier/experiments/cifar10"
SCRIPT="$BASE_DIR/animal_classifier/train_cifar10.py"

run() {
  local id=$1 model=$2 aug=$3 outdir="$EXP_DIR/$4"
  echo ""
  echo "=========================================="
  echo "Starting $4"
  echo "=========================================="
  cd "$BASE_DIR"
  PYTHONPATH=. python3 -u "$SCRIPT" \
    --model "$model" --epochs 50 --batch 128 \
    --aug "$aug" --decay_at 30 40 \
    --output_dir "$outdir" \
    2>&1 | tee "$outdir/train.log"
  echo "$4 DONE"
}

echo "Waiting for C01 to complete..."
until grep -q "Final.*Top-1" "$EXP_DIR/C01-vgglike-baseline/train.log" 2>/dev/null; do
  sleep 30
done

run C02 vgg_bn mild "C02-vggbn-batchnorm"
run C03 resnet  mild "C03-resnet-skipconn"
run C04 vgg_bn  none "C04-vggbn-noaug"

echo ""
echo "All CIFAR-10 experiments done."
cd "$BASE_DIR"
PYTHONPATH=animal_classifier python3 animal_classifier/experiments/generate_report.py
