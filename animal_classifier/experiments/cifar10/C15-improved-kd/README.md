# C15 — Improved Knowledge Distillation / CIFAR-10

## Motivation

C09 (Knowledge Distillation) produced lower-than-expected student accuracy.
The main issues were:

1. Student too small (32/64/128 channels, no BN) → underfitting
2. α=0.7 gave too much weight to CE loss → KD signal underutilised
3. Temperature T=4 not ablated against T=2 or T=8

C15 fixes all three problems.

## Changes from C09

| Aspect | C09 | C15 |
|--------|-----|-----|
| Student capacity | 32/64/128 channels, no BN | 64/128/256 channels + BN in each block |
| Loss balance | α=0.7·CE + 0.3·KD | α=0.3·CE + 0.7·KD (more KD) |
| Temperature | Fixed T=4 | Sweep T=2, 4, 8 |
| Online distillation | No | Yes (teacher warms up 5 epochs together) |
| Epochs | 30 | 30 |
| Batch size | 256 | 128 |

## Architecture

**Teacher**: VGGWithBN (Conv-BN-ReLU×6 + FC×2), loaded from C02 pkl if available, otherwise trained 10 epochs online.

**Student** (ImprovedStudentNet):
```
Conv(64)-BN-ReLU-Pool → Conv(128)-BN-ReLU-Pool → Conv(256)-BN-ReLU-Pool
→ FC(512)-ReLU-Dropout(0.3) → FC(10)
```
~1.4M parameters (vs C09 ~0.7M).

## KD Loss Formula

```
L = α × CrossEntropy(logits, hard_labels)
  + (1-α) × KL(softmax(logits/T), softmax(teacher_logits/T))

α = 0.3  (more weight on soft-label signal)
```

## Temperature Effect

| T | Effect |
|---|--------|
| T=2 | Mild softening — close to one-hot |
| T=4 | Standard — reveals inter-class similarity |
| T=8 | Strong softening — dark knowledge maximised |

## Running

```bash
cd <repo_root>
python animal_classifier/experiments/cifar10/C15-improved-kd/improved_kd.py
```

## Expected Results

With pre-trained C02 teacher (~90.6%):

| Config | Expected Test Acc |
|--------|-------------------|
| Scratch (no KD) | ~72–75% |
| KD T=2 | ~77–80% |
| KD T=4 | ~78–82% |
| KD T=8 | ~76–80% |

Student is ~40% smaller than teacher, so some accuracy gap is expected.
The key metric is that KD variants clearly outperform the scratch baseline.
