# C05 - MobileNet / CIFAR-10 / Mild Augmentation

## 概要

Depthwise Separable Convolution（深さ方向分離畳み込み）を使った軽量アーキテクチャ。
通常の畳み込みを「チャンネルごとのDepthwise Conv」+「1×1のPointwise Conv」に分解することでパラメータ数を大幅削減。

## 構成

- Stem: Conv(32, 3×3)
- DS blocks: 32→64, 64→128(s2), 128→128, 128→256(s2), 256→256, 256→256
- GlobalAvgPool → FC(10)
- Epochs: 30 / Batch: 256 / Aug: mild

## 計算量の比較（理論値）

| 手法 | 9チャンネル畳み込みのコスト |
|------|--------------------------|
| 通常のConv | C_in × C_out × k² |
| Depthwise Separable | C_in × k² + C_in × C_out |
| 削減率（k=3, C_out=128, C_in=64） | 約8倍削減 |

## 結果

> 実行中
