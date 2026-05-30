# C06 - DenseNet-40 / CIFAR-10 / Mild Augmentation

## 概要

各層が以前の全層の出力をチャンネル方向に結合して受け取るDense Connection。
特徴の再利用により少ないパラメータで高精度を達成。

## 構成

- Init Conv: 16 channels
- DenseBlock × 3（各6層、growth rate k=12）
- TransitionLayer（1×1 Conv + AvgPool で圧縮）
- BN-ReLU → GlobalAvgPool → FC(10)
- Batch: 128（メモリ節約のためC05より小さく）

## Dense Connectionの仕組み

```
Layer 0:  x0 (16ch)
Layer 1:  [x0] → BN-ReLU-Conv → x1 (12ch)
Layer 2:  [x0, x1] → BN-ReLU-Conv → x2 (12ch)
Layer 3:  [x0, x1, x2] → BN-ReLU-Conv → x3 (12ch)
Block出力: [x0, x1, x2, x3] = 16+36=52ch
```

## 結果

> 実行予定
