# 08 - VGGWithBN / Top10クラス / 64px / Full Augmentation (比較)

## 概要

実験06（mild aug）との直接比較。64px解像度でfull augmentation（flip+crop+color jitter+cutout）が有効かを検証。

## 構成

- モデル: VGGWithBN
- 入力サイズ: 64×64
- クラス数: 10
- Augmentation: **full**（flip + crop + color jitter + cutout）
- Optimizer: Adam (lr=0.001)
- LR decay: epoch 30 → 1e-4, epoch 40 → 1e-5
- Epochs: 50

## 仮説

32×32では color jitter が有害（実験04で判明）だが、64×64では十分な情報量があり color jitter が正則化として機能するかもしれない。

## 結果

> 実行予定
