# C14 - EfficientNet-style Compound Scaling / CIFAR-10

## 概要

EfficientNet（Tan & Le, 2019）が提案した「複合スケーリング（Compound Scaling）」の考え方を純粋なNumPyで実装する。幅・深さ・解像度を独立にスケールするのではなく、固定の比率α、βで同時にスケールすることで、パラメータ効率の良いモデルを目指す。

## アイデア：Compound Scaling

ネットワークのスケールアップ方法を形式化すると：

```
幅  : d_model × α^φ
深さ : n_blocks × β^φ
解像度: resolution × γ^φ   (CIFAR-10では32×32固定)
```

本実験では φ=1.0 で：

| パラメータ | 値 |
|-----------|-----|
| α (幅係数) | 1.2 |
| β (深さ係数) | 1.1 |
| φ (スケール乗数) | 1.0 |
| 幅スケール (α^φ) | 1.20 |
| 深さスケール (β^φ) | 1.10 |

## ベースモデルとスケール後の構成

| ステージ | ベースチャンネル | スケール後 | ベース深さ | スケール後深さ |
|---------|----------------|-----------|-----------|-------------|
| Stem | 32 | 38 | — | — |
| Stage1 | 64 | 78 | 1 | 2 |
| Stage2 | 128 | 154 | 2 | 3 |
| Stage3 | 128→256 | 154→308 | 2→2 | 3→3 |

ビルディングブロックはC05-MobileNetと同じDepthwise Separable Convolution（DSConvBlock）を流用。

## アーキテクチャ詳細

```
入力: (N, 3, 32, 32)

Stem: Conv(3→38, 3×3) → BN → ReLU

Stage1: DS(38→78, s=1) × 2

Stage2: DS(78→154, s=2) → DS(154→154, s=1) × 2

Stage3: DS(154→154, s=1) → DS(154→308, s=2) → DS(308→308, s=1) × 2

GlobalAvgPool → FC(308→10)
```

## 学習設定

| 設定 | 値 |
|------|-----|
| エポック数 | 30 |
| バッチサイズ | 128 |
| 学習率 | 1e-3 → 1e-5（コサイン減衰）|
| オプティマイザ | Adam |
| Augmentation | Mild（flip + crop pad=2）|

## C05-MobileNetとの比較

| 項目 | C05-MobileNet | C14-EfficientNet-style |
|------|---------------|----------------------|
| チャンネル | [32,64,128,128,256,256,256] | [38,78,154,154,308] |
| 総DSブロック数 | 6 | 11（深さスケール後）|
| スケーリング戦略 | なし | Compound（幅×1.2, 深さ×1.1）|

## 実行方法

```bash
# リポジトリルートから実行
python animal_classifier/experiments/cifar10/C14-efficientnet-style/efficientnet_cifar10.py
```

## 結果

| 指標 | 値 |
|------|-----|
| Best Test Accuracy | 実行後に記入 |
| Final Test Accuracy | 実行後に記入 |

## ログ形式

```
epoch X/30: loss=Y.YYYY  train=Z.ZZZZ  test=W.WWWW  (Ns)
```

## 参考文献

- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.
