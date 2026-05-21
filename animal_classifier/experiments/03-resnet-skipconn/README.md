# 03 - ResNet (Skip Connections)

## 概要

残差接続（skip connection）付きResNet。勾配消失を防ぎ深いネットワークを学習可能にする。

## 構成

- モデル: ResNet（stem + 3 stage × 2 blocks + GlobalAvgPool + FC）
- チャンネル数: 64 → 128 → 256（stride=2でダウンサンプル）
- 入力サイズ: 32×32
- クラス数: 37
- Augmentation: ON
- Optimizer: Adam (lr=0.001)
- Epochs: 30
- Batch: 32

## アーキテクチャ詳細

```
Input (N, 3, 32, 32)
→ Stem: Conv(64,3×3,pad=1) → BN → ReLU
→ Stage1: ResBlock(64,64) × 2
→ Stage2: ResBlock(64,128,stride=2) + ResBlock(128,128)
→ Stage3: ResBlock(128,256,stride=2) + ResBlock(256,256)
→ GlobalAvgPool → Flatten
→ Affine(256→37)

ResidualBlock:
  Conv-BN-ReLU-Conv-BN + skip(1×1 projection if needed) → ReLU
```

## 結果

| 指標 | 値 |
|------|----|
| Best test acc (sampling) | 10.62% |
| Final Top-1 (full test) | 5.07% |
| Final Top-5 (full test) | 18.97% |

## 所見

- 1エポック約2分（VGGの3倍遅い）
- epoch 3〜7でtest acc = 0.000 が続く学習不安定期あり
- 最終的には収束するがVGGWithBNより低い精度
- 純NumPy実装ではim2col起因のメモリ/速度がボトルネック
- 64×64+top10クラスへの変更で改善期待（実験07で検証）
