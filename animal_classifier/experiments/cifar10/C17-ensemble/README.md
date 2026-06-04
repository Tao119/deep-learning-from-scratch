# C17 — Model Ensemble

Combines the best models from previous experiments to evaluate whether
ensemble strategies outperform the best individual model.

## Models Combined

| Experiment | Model | Notes |
|---|---|---|
| C02 | VGGWithBN | Baseline with BatchNorm |
| C05 | MobileNet | Lightweight depthwise separable conv |
| C08 | VGGWithBN + Mixup | Data augmentation with Mixup |

## Ensemble Strategies

1. **Soft average** — average probability distributions, then argmax
2. **Majority vote (hard)** — each model votes, majority wins
3. **Weighted average** — weight by validation accuracy (C02: 0.82, C05: 0.75, C08: 0.83)

## Expected Results (trained models)

| Model/Strategy | Accuracy |
|---|---|
| C02-VGGWithBN | ~82% |
| C05-MobileNet | ~75% |
| C08-VGGWithBN-Mixup | ~83% |
| Ensemble (majority) | ~84-85% |
| Ensemble (soft avg) | ~84-85% |
| Ensemble (weighted) | **~85%** |

Ensemble typically gains **+1 to +2%** over the best individual model by reducing
correlated errors across architectures.

## Usage

```bash
cd animal_classifier
python experiments/cifar10/C17-ensemble/ensemble_eval.py
```

If pkl files are absent, random-weight placeholder models are used automatically
(accuracy ~10%, as expected for chance-level CIFAR-10).
