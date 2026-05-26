# C04 - VGGWithBN / CIFAR-10 / No Augmentation

## 構成

- モデル: VGGWithBN
- 入力サイズ: 32×32 RGB
- クラス数: 10
- Augmentation: **OFF**
- Optimizer: Adam (lr=0.001)
- LR decay: epoch 20 → 1e-4, epoch 25 → 1e-5
- Epochs: 30 / Batch: 256

## 結果

| 指標 | 値 |
|------|----|
| Final Top-1 (full test) | 87.37% |
| Final Top-5 (full test) | 99.31% |
| Train acc (終盤) | 100%（完全過学習） |

## C02との比較（augmentationの効果）

| | C02 (mild aug) | C04 (no aug) | 差 |
|--|--|--|--|
| Top-1 | **90.58%** | 87.37% | +3.21pt |
| Top-5 | **99.70%** | 99.31% | +0.39pt |

## 所見

- augmentationなしでも87.4%と高精度（データが5000枚/classあるため）
- ただしmild augより-3.2pt劣る → CIFAR-10ではaugmentationが正則化として機能
- Oxford Pets（100枚/class）ではaugなしが圧勝した逆の結果
- **データ量が多いほどaugmentationは有効**というパターンが確認された
