# 01 - VGGLike Baseline

## 概要

3ブロックのVGG風CNNによるベースライン。Oxford-IIIT Petデータセット37クラス分類。

## 構成

- モデル: VGGLike（Conv-ReLU-Conv-ReLU-Pool × 3 + FC × 2）
- チャンネル数: 64 → 128 → 256
- 入力サイズ: 32×32
- クラス数: 37
- Augmentation: ON（flip, crop, color jitter, cutout）
- Optimizer: Adam (lr=0.001)
- LR decay: epoch 20 → 1e-4, epoch 25 → 1e-5
- Epochs: 30
- Batch: 32

## アーキテクチャ詳細

```
Input (N, 3, 32, 32)
→ Conv(64,3×3) → ReLU → Conv(64,3×3) → ReLU → MaxPool(2×2)
→ Conv(128,3×3) → ReLU → Conv(128,3×3) → ReLU → MaxPool(2×2)
→ Conv(256,3×3) → ReLU → Conv(256,3×3) → ReLU → MaxPool(2×2)
→ Flatten (256×4×4=4096)
→ Affine(4096→512) → ReLU → Dropout(0.5)
→ Affine(512→37) → Softmax
```

## 結果

| 指標 | 値 |
|------|----|
| Best test acc (sampling) | 18.75% |
| Final Top-1 (full test) | 3.87% |
| Final Top-5 (full test) | 17.68% |

## 所見

- サンプリング評価と全体評価の乖離が大きい（評価の不安定さに起因）
- 32×32解像度では37クラスの細粒度分類に情報量が不足
- Augmentationにより損失は下降するが汎化が追いつかない
