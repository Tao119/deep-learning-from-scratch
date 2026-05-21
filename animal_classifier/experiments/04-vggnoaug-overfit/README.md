# 04 - VGGLike No Augmentation

## 概要

Augmentationなしのベースラインとの比較。結果的に全モデル中最高精度を記録。

## 構成

- モデル: VGGLike（01と同じアーキテクチャ）
- 入力サイズ: 32×32
- クラス数: 37
- Augmentation: **OFF**
- Optimizer: Adam (lr=0.001)
- LR decay: epoch 20 → 1e-4, epoch 25 → 1e-5
- Epochs: 30
- Batch: 32

## 結果

| 指標 | 値 |
|------|----|
| Best test acc (sampling) | 14.58% |
| Final Top-1 (full test) | **15.62%** |
| Final Top-5 (full test) | **48.36%** |
| Train acc (epoch 22+) | 100% |

## 所見

**Augmentationありモデルを大幅に上回る逆転現象が発生。**

原因の考察：
1. 32×32という低解像度では、flip/crop/color jitterが情報を破壊しすぎる
2. Cutoutが有効ピクセルをゼロにする割合が高くなりすぎる
3. 37クラスの細粒度分類（犬種・猫種の識別）ではテクスチャ・色情報が重要で、color jitterが有害に作用した
4. モデルは過学習しているが（train=100%）、過学習した特徴がたまたまテストセットとも相関

→ 改善方向: augmentationの種類・強度を調整する実験（06, 07で検証）
