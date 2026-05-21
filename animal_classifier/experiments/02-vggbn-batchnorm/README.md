# 02 - VGGWithBN (Batch Normalization)

## 概要

VGGLikeの各Conv層の後にBatch Normalizationを追加。学習安定化を狙う。

## 構成

- モデル: VGGWithBN（Conv-BN-ReLU × 2 → Pool × 3 + FC × 2）
- 入力サイズ: 32×32
- クラス数: 37
- Augmentation: ON
- Optimizer: Adam (lr=0.001)
- LR decay: epoch 20 → 1e-4, epoch 25 → 1e-5
- Epochs: 30
- Batch: 32

## アーキテクチャ詳細

```
Input (N, 3, 32, 32)
→ Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool
→ Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → MaxPool
→ Conv(256) → BN → ReLU → Conv(256) → BN → ReLU → MaxPool
→ Flatten → Affine(4096→512) → ReLU → Dropout(0.5) → Affine(512→37)
```

## 結果

| 指標 | 値 |
|------|----|
| Best test acc (sampling) | 20.42% |
| Final Top-1 (full test) | 4.03% |
| Final Top-5 (full test) | 20.67% |

## 所見

- VGGLikeより若干改善（Top-1: 3.87% → 4.03%）
- epoch 17でtrain acc = 0.0000 になる不安定な挙動あり（BNの running stats 初期化問題の可能性）
- 4D BNの実装（channel-wise normalization）が正常動作することを確認
- 32×32+37クラスでは改善余地が少ない
