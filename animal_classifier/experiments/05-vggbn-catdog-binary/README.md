# 05 - VGGWithBN / Cat vs Dog 2値分類 / 64px

## 概要

37犬猫種分類を「猫 or 犬」の2クラスに単純化。高い精度を目標とする。

## 構成

- モデル: VGGWithBN
- 入力サイズ: 64×64
- クラス数: **2**（Cat=0, Dog=1）
- Augmentation: mild（flip + 小crop）
- Optimizer: Adam (lr=0.001)
- LR decay: epoch 20 → 1e-4, epoch 25 → 1e-5
- Epochs: 30
- Batch: 32
- Train/Test: 3680/3669サンプル（全データ使用）

## クラス定義

- Cat (class_id 0): Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau,
  Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx（12種）
- Dog (class_id 1): 残り25犬種

## 結果

> 実行予定 — 実験06完了後に開始

## 期待値

- Top-1 80〜90%以上（2クラスで全データ使用、形態的差異が大きい）
