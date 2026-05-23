# C03 - ResNet / CIFAR-10 / Mild Augmentation (途中中止)

## 結果（epoch 17/30 時点）

| 指標 | 値 |
|------|----|
| Best test acc (sampling) | ~88.3% (epoch 14) |
| 実行エポック数 | 17/30 |
| 中止理由 | 1エポック約2〜2.7時間、完了まで数日かかるため |

## 学習推移（抜粋）

| Epoch | Train | Test |
|-------|-------|------|
| 10 | 93.1% | 83.8% |
| 14 | 93.1% | **88.3%** |
| 17 | 92.9% | 86.7% |

## 所見

- pure NumPy im2col実装ではResNetの残差接続計算が非常に重い（VGGの約15倍）
- 精度はVGGWithBN(90.6%)に迫る水準（epoch14で88.3%）
- 本格的な比較にはGPU/PyTorchフレームワークが必要

## アーキテクチャ

- Stem Conv(64) + 3 stage(64→128→256) + GlobalAvgPool + FC(10)
- 残差接続あり、1×1 projection shortcut
