# C16 — Mixup vs CutMix Augmentation Comparison / CIFAR-10

## Motivation

C08 showed Mixup augmentation on VGGWithBN with 30 epochs.  This experiment
adds CutMix to create a side-by-side three-way comparison using a shorter
15-epoch budget so the differences emerge quickly.

## Augmentation Strategies

| Strategy | Formula | Key Property |
|----------|---------|-------------|
| Baseline | mild aug (flip + crop pad=2) | Reference |
| Mixup    | x_mix = λx_i + (1-λ)x_j, y_mix = λy_i + (1-λ)y_j | Pixel-level blending |
| CutMix   | paste rectangular region from x_j into x_i | Preserves local structure |

Both Mixup and CutMix draw λ from Beta(alpha, alpha).

### Mixup (α=0.4)
```
λ ~ Beta(0.4, 0.4)
x_mix = λ·x_i + (1-λ)·x_j
y_mix = λ·y_i + (1-λ)·y_j   (soft labels)
```

### CutMix (α=1.0)
```
λ ~ Beta(1.0, 1.0)   → uniform, cuts ~50% of image
x_mix: paste box from x_j into x_i
y_mix = λ_actual·y_i + (1-λ_actual)·y_j   (by pixel area ratio)
```

## Configuration

- Model: VGGWithBN (same as C02, C08)
- Input: 32×32 CIFAR-10
- Epochs: 15 per run
- Batch: 256
- Optimizer: Adam lr=0.001
- LR decay: epoch 10 → 1e-4, epoch 13 → 1e-5
- Seed: 42 (identical initialisation for fair comparison)

## Expected Behaviour

| Method | Expected Advantage |
|--------|--------------------|
| Baseline | Fast convergence, slight overfitting |
| Mixup | Smoother loss landscape, better generalisation |
| CutMix | Stronger localisation regularisation, often best at 15+ epochs |

## Running

```bash
cd <repo_root>
python animal_classifier/experiments/cifar10/C16-mixup-vs-cutmix/compare_aug.py
```

## Output Files

- `aug_comparison_curves.png`   — learning curves for all three methods
- `aug_comparison_results.json` — best / final accuracy, training time
- `VGGWithBN_cifar10_<mode>_best.pkl` — best checkpoint per method

## Relationship to Other Experiments

| Exp | Augmentation | Epochs | Notes |
|-----|-------------|--------|-------|
| C02 | Mild only   | 30     | Teacher baseline |
| C08 | Mild + Mixup (α=0.4) | 30 | Full run |
| C16 | Baseline / Mixup / CutMix | 15 | Quick three-way comparison |
