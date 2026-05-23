#!/usr/bin/env bash
set -e
BASE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="$BASE_DIR/animal_classifier/experiments/cifar10"

run() {
  local model=$1 aug=$2 outdir="$EXP_DIR/$3"
  mkdir -p "$outdir"
  echo ""
  echo "========== Starting $3 =========="
  cd "$BASE_DIR"
  PYTHONPATH=. python3 -u animal_classifier/train_cifar10.py \
    --model "$model" --epochs 30 --batch 256 \
    --aug "$aug" --decay_at 20 25 \
    --output_dir "$outdir" \
    2>&1 | tee "$outdir/train.log"
  echo "$3 DONE"
}

echo "Waiting for C01..."
until grep -q "Final.*Top-1" "$EXP_DIR/C01-vgglike-baseline/train.log" 2>/dev/null; do
  sleep 30
done

run vgg_bn mild "C02-vggbn-batchnorm"
run resnet  mild "C03-resnet-skipconn"
run vgg_bn  none "C04-vggbn-noaug"

echo "All done."
