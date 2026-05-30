# C07 - SE-ResNet / CIFAR-10 / Mild Augmentation

## 概要

ResNetの各残差ブロックにSqueeze-and-Excitation(SE)モジュールを追加。
チャンネルごとの重要度をGlobal Average Poolingで推定し、自動的にスケーリング。

## SE Blockの仕組み

```
Input (N, C, H, W)
  ↓ Global Avg Pool (Squeeze)
(N, C)
  ↓ FC(C→C/16) → ReLU → FC(C/16→C) → Sigmoid (Excitation)
(N, C) ... チャンネル重要度
  ↓ × 入力テンソル (Scale)
Output (N, C, H, W)
```

## 構成

- ResNetと同じ骨格（stem + 6 SEResidualBlocks）
- 各ブロックにSEモジュール（reduction ratio r=16）
- Batch: 128

## 結果

> 実行予定
