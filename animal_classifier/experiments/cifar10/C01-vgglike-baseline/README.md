# C01 - VGGLike / CIFAR-10 / Mild Augmentation

## 構成

- モデル: VGGLike（Conv×6 + FC×2）
- 入力サイズ: 32×32 RGB
- クラス数: 10
- Augmentation: mild（flip + crop pad=2）
- Optimizer: Adam (lr=0.001)
- LR decay: epoch 20 → 1e-4, epoch 25 → 1e-5
- Epochs: 30 / Batch: 256

## 結果

| 指標 | 値 |
|------|----|
| Final Top-1 (full test) | **87.75%** |
| Final Top-5 (full test) | 99.50% |
| 1エポック所要時間 | 約496秒 |

## 所見

- Oxford Pets（同アーキテクチャ）のTop-1 3.87% に対し、CIFAR-10では87.75%
- データ量の差（100枚/class vs 5000枚/class）が精度に直結
- epoch 1 時点ですでにtest=51.7%（Oxford Pets 30エポック後より高い）
