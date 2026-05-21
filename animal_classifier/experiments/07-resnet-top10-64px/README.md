# 07 - ResNet / Top10クラス / 64px / Mild Augmentation

## 概要

実験03のResNetを改善版設定（top10, 64px）で再試験。ResNetが本来の性能を発揮できるか検証。

## 構成

- モデル: ResNet（stem + 3 stage + GlobalAvgPool）
- 入力サイズ: 64×64
- クラス数: 10
- Augmentation: mild
- Optimizer: Adam (lr=0.001)
- LR decay: epoch 30 → 1e-4, epoch 40 → 1e-5
- Epochs: 50
- Batch: 32

## 結果

> 実行予定 — 実験05完了後に開始

## 期待値

- ResNetは解像度が高いほど残差接続の恩恵が出やすい
- 実験03（Top-1 5.07%）から大幅改善を期待
- 実験06（VGGWithBN）との比較でアーキテクチャの優劣を判定
