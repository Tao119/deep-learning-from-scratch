# C08 - VGGWithBN + Mixup / CIFAR-10

## 概要

C02（VGGWithBN + mild augmentation、90.58%）をベースラインとし、Mixup augmentationを追加した場合の精度変化を検証する。

Mixupは2枚の画像と対応するラベルを線形補間する手法。

```
x_mix = λ·x_i + (1-λ)·x_j
y_mix = λ·y_i + (1-λ)·y_j     (y はone-hot)
λ ~ Beta(alpha, alpha)
```

soft labelを学習することで正則化効果が生まれ、過学習を抑制する。

## 構成

- モデル: VGGWithBN（Conv-BN-ReLU×6 + FC×2）
- 入力サイズ: 32×32 RGB
- クラス数: 10
- Augmentation: mild（flip + crop pad=2）→ Mixup（alpha=0.4）
- Optimizer: Adam (lr=0.001)
- LR decay: epoch 20 → 1e-4, epoch 25 → 1e-5
- Epochs: 30 / Batch: 256

## C02との比較

| 実験 | Augmentation | Top-1 |
|------|-------------|-------|
| C02  | mild only   | 90.58% |
| C08  | mild + Mixup | 実行後に記入 |

Mixupによる過学習抑制でC02を上回ることを期待する。精度評価はハードラベル（argmax）で行う。

## 実行方法

```bash
cd <repo_root>
bash animal_classifier/experiments/cifar10/C08-vggbn-mixup/run.sh
```
