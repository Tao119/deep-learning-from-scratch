# C02 - VGGWithBN / CIFAR-10 / Mild Augmentation

## 構成

- モデル: VGGWithBN（Conv-BN-ReLU×6 + FC×2）
- 入力サイズ: 32×32 RGB
- クラス数: 10
- Augmentation: mild（flip + crop pad=2）
- Optimizer: Adam (lr=0.001)
- LR decay: epoch 20 → 1e-4, epoch 25 → 1e-5
- Epochs: 30 / Batch: 256

## 結果

| 指標 | 値 |
|------|----|
| Final Top-1 (full test) | **90.58%** |
| Final Top-5 (full test) | 99.70% |

## 所見

- C01（VGGLike）より+2.83pt改善
- BatchNormalizationが安定した学習をもたらしBNなしより高精度
- Oxford PetsのC02（4.03%）と比べ22倍以上の精度 → データ量の重要性を強調
- CIFAR-10の一般的ベンチマーク（VGG系90%前後）に到達
