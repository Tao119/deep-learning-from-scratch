#!/usr/bin/env bash
# 実験06完了後に05→07→08を順番に実行する
set -e
BASE="$(cd "$(dirname "$0")/.." && pwd)"
EXP_DIR="$BASE/animal_classifier/experiments"

run_exp() {
  local id=$1 model=$2 mode=$3 size=$4 epochs=$5 aug=$6 decay_at=$7
  local outdir="$EXP_DIR/0${id}-${model//_/-}-${mode}-${size}px"
  # フォルダが引数で渡された場合はそちら優先
  [ -n "$8" ] && outdir="$EXP_DIR/$8"
  mkdir -p "$outdir"
  echo ""
  echo "=========================================="
  echo "Starting experiment $id: $model $mode ${size}px $aug"
  echo "Output: $outdir"
  echo "=========================================="
  cd "$BASE"
  PYTHONPATH=. python3 -u animal_classifier/train_quick.py \
    --model "$model" --mode "$mode" --size "$size" \
    --epochs "$epochs" --aug "$aug" \
    --decay_at $decay_at \
    --output_dir "$outdir" \
    2>&1 | tee "$outdir/train.log"
  echo "Experiment $id DONE"
}

echo "Waiting for experiment 06 to complete..."
until grep -q "Final.*Top-1" "$EXP_DIR/06-vggbn-top10-64px/train.log" 2>/dev/null; do
  sleep 30
done
echo "Experiment 06 DONE. Starting 05..."

run_exp 5 vgg_bn cat_dog 64 25 mild "15 20" "05-vggbn-catdog-binary"
echo "Experiment 05 DONE. Starting 07..."

run_exp 7 resnet top10 64 50 mild "30 40" "07-resnet-top10-64px"
echo "Experiment 07 DONE. Starting 08..."

mkdir -p "$EXP_DIR/08-vggbn-top10-fullaug"
cd "$BASE"
PYTHONPATH=. python3 -u animal_classifier/train_quick.py \
  --model vgg_bn --mode top10 --size 64 --epochs 50 --aug full \
  --decay_at 30 40 \
  --output_dir "$EXP_DIR/08-vggbn-top10-fullaug" \
  2>&1 | tee "$EXP_DIR/08-vggbn-top10-fullaug/train.log"

echo ""
echo "All experiments done. Generating report..."
cd "$BASE"
PYTHONPATH=animal_classifier python3 animal_classifier/experiments/generate_report.py
echo "Report saved to experiments/comparison_all.png"
