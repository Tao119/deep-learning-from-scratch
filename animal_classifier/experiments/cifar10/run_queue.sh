#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
BASE="$(pwd)"

wait_for_mem() {
    while true; do
        free_mb=$(vm_stat | awk '/Pages free/{print int($3*4096/1024/1024)}')
        [ "$free_mb" -gt 500 ] && break
        echo "  [queue] waiting for memory (free: ${free_mb}MB)..."
        sleep 60
    done
}

run_exp() {
    local name="$1"; shift
    echo ""; echo "====== Starting $name ======"; date
    wait_for_mem
    PYTHONPATH="$BASE/.." python3 -u "$@" 2>&1 | tee "experiments/cifar10/$name/train.log"
    echo "====== Done $name ======"; date
}

echo "Queue: C09 → C11 → C12"
echo "Waiting for C08 to finish..."

while pgrep -f "train_cifar10_mixup" > /dev/null; do sleep 30; done
echo "C08 done. Starting queue..."

run_exp C09-knowledge-distillation \
    experiments/cifar10/C09-knowledge-distillation/distill.py \
    --epochs 30 --batch 256

run_exp C11-label-smoothing \
    experiments/cifar10/C11-label-smoothing/train_smooth.py

run_exp C12-lr-schedule \
    experiments/cifar10/C12-lr-schedule/compare_schedules.py

echo "All queued experiments done."
