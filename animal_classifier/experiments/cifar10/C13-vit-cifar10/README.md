# C13 - Vision Transformer (ViT) / CIFAR-10

## 概要

「An Image is Worth 16×16 Words」(Dosovitskiy et al., 2021) で提案されたVision Transformerを、プロジェクトのfrom-scratch方針に従い純粋なNumPyで実装する。CNNを一切使わず、画像をパッチ列として扱うことで、Self-AttentionだけでCIFAR-10を分類する。

## アーキテクチャ

```
入力: (N, 3, 32, 32)

1. Patch Embedding
   - patch_size = 4  →  8×8 = 64 パッチ
   - 各パッチ: 4×4×3 = 48 ピクセル
   - Linear(48 → 64=d_model)
   → (N, 64, 64)

2. [CLS] トークン先頭に付加
   → (N, 65, 64)

3. 学習可能な位置埋め込み (65, 64)

4. Transformer Encoder × 4層
   - Pre-LN Multi-Head Self-Attention (4ヘッド, head_dim=16)
   - Pre-LN FFN: Linear(64→256) → GELU → Linear(256→64)
   - 残差接続

5. [CLS] トークン抽出 → Linear(64→10)
```

## 学習設定

| 設定 | 値 |
|------|-----|
| パッチサイズ | 4×4 |
| d_model | 64 |
| ヘッド数 | 4 |
| Transformerレイヤー数 | 4 |
| FFN中間次元 | 256 |
| エポック数 | 30 |
| バッチサイズ | 128 |
| 学習率 | 1e-3 → 1e-5（コサイン減衰）|
| オプティマイザ | Adam |
| Augmentation | Mild（flip + crop pad=2）|

## 実装上のポイント

**CNNを使わないVision Transformer**

CNNは「局所的・階層的」に特徴を抽出するのに対し、ViTは全トークン間のSelf-Attentionにより「大局的・関係的」に特徴を掴む。CIFAR-10のような小データセットでは過学習しやすく、CNNより学習に多くのデータを要するとされる。

**Pre-LayerNorm**

オリジナルのViTはPost-LNだが、勾配消失を防ぐためPre-LN（Attention/FFNの前にLN）を採用。学習が安定しやすい。

**[CLS] トークン**

BERT同様、系列先頭に学習可能なClassトークンを付加し、最終層でそのトークンの出力を分類ヘッドに通す。

## 実行方法

```bash
# リポジトリルートから実行
python animal_classifier/experiments/cifar10/C13-vit-cifar10/vit_cifar10.py
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
