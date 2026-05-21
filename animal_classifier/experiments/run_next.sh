#!/usr/bin/env bash
# 実験を順番に実行するスクリプト
# 使い方: bash experiments/run_next.sh <実験ID>

set -e
cd "$(dirname "$0")/.."

case "$1" in
  05)
    echo "=== 05: VGGWithBN Cat vs Dog binary 64px ==="
    PYTHONPATH=. python3 -u animal_classifier/train_quick.py \
      --model vgg_bn --mode cat_dog --size 64 --epochs 25 --aug mild \
      --decay_at 15 20 \
      --output_dir animal_classifier/experiments/05-vggbn-catdog-binary \
      2>&1 | tee animal_classifier/experiments/05-vggbn-catdog-binary/train.log
    ;;
  07)
    echo "=== 07: ResNet Top10 64px mild aug ==="
    PYTHONPATH=. python3 -u animal_classifier/train_quick.py \
      --model resnet --mode top10 --size 64 --epochs 50 --aug mild \
      --output_dir animal_classifier/experiments/07-resnet-top10-64px \
      2>&1 | tee animal_classifier/experiments/07-resnet-top10-64px/train.log
    ;;
  08)
    echo "=== 08: VGGWithBN Top10 64px full aug (比較) ==="
    mkdir -p animal_classifier/experiments/08-vggbn-top10-fullaug
    PYTHONPATH=. python3 -u animal_classifier/train_quick.py \
      --model vgg_bn --mode top10 --size 64 --epochs 50 --aug full \
      --output_dir animal_classifier/experiments/08-vggbn-top10-fullaug \
      2>&1 | tee animal_classifier/experiments/08-vggbn-top10-fullaug/train.log
    ;;
  *)
    echo "Usage: $0 <05|07|08>"
    exit 1
    ;;
esac
