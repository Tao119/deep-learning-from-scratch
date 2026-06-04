"""
C16 — Mixup vs CutMix Augmentation Comparison / CIFAR-10

Three runs on the same VGGWithBN architecture:
  1. Baseline  — mild augmentation only (flip + crop pad=2)
  2. Mixup     — mild aug → Mixup(alpha=0.4)
  3. CutMix    — mild aug → CutMix(alpha=1.0)

Each run: 15 epochs (quick comparison), batch=256, Adam lr=0.001

Output
------
  aug_comparison_curves.png     — 3 learning curves on one graph (test acc)
  aug_comparison_results.json   — numeric summary

Background
----------
C08 established that Mixup on VGGWithBN is slightly better than mild-only
augmentation.  C16 adds CutMix to the comparison so we can see whether
pasting random rectangular patches from a second image (CutMix) outperforms
linearly interpolating the whole image (Mixup).

Key difference:
  - Mixup blends pixels globally        → smoother decision boundary
  - CutMix pastes a local region        → object-localisation signal preserved

Run
---
  cd <repo_root>
  python animal_classifier/experiments/cifar10/C16-mixup-vs-cutmix/compare_aug.py
"""

import os
import sys
import time
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ───────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../../.."))
sys.path.insert(0, _REPO_ROOT)

from animal_classifier.dataset.cifar10 import load_cifar10, normalize, CLASS_NAMES
from animal_classifier.dataset.augmentation import (
    batch_mild_augment,
    batch_mixup_augment,
    batch_cutmix_augment,
)
from animal_classifier.models.vgg_bn import VGGWithBN
from common.optimizer import Adam

OUTPUT_DIR   = _SCRIPT_DIR
N_CLASSES    = 10
INPUT_SIZE   = 32
INPUT_CHANNELS = 3


# ===========================================================================
# Training helpers
# ===========================================================================

def _build_model():
    """Fresh VGGWithBN for CIFAR-10 (32×32, 10 classes)."""
    return VGGWithBN(INPUT_CHANNELS, INPUT_SIZE, N_CLASSES)


def _train_epoch(model, x_train, t_train, opt, batch_size, aug_mode, alpha):
    """One training epoch.  Returns mean loss."""
    n = len(x_train)
    itr = max(n // batch_size, 1)
    idx = np.random.permutation(n)
    ep_loss = 0.0

    for i in range(itr):
        bi = idx[i * batch_size:(i + 1) * batch_size]
        xb = x_train[bi]
        tb = t_train[bi]

        if aug_mode == "baseline":
            xb = batch_mild_augment(xb, train_flg=True)
            loss = model.loss(xb, tb)
            dout = model.last_layer.backward(1)
        elif aug_mode == "mixup":
            xb, tb_soft = batch_mixup_augment(xb, tb, train_flg=True)
            # With soft (mixed) labels use CE-with-soft-targets manually
            logits = model.predict(xb, train_flg=True)
            from common.activation import softmax
            pred = softmax(logits)
            loss = -np.mean(np.sum(tb_soft * np.log(pred + 1e-7), axis=1))
            model.last_layer.loss = loss
            dpred = (pred - tb_soft) / xb.shape[0]
            dout = dpred
        elif aug_mode == "cutmix":
            xb, tb_soft = batch_cutmix_augment(xb, tb, train_flg=True)
            logits = model.predict(xb, train_flg=True)
            from common.activation import softmax
            pred = softmax(logits)
            loss = -np.mean(np.sum(tb_soft * np.log(pred + 1e-7), axis=1))
            model.last_layer.loss = loss
            dpred = (pred - tb_soft) / xb.shape[0]
            dout = dpred
        else:
            raise ValueError(f"Unknown aug_mode: {aug_mode}")

        # Backprop through model layers
        for layer in reversed(list(model.layers.values())):
            dout = layer.backward(dout)

        grads = model_gradients(model)
        opt.update(model.params, grads)
        ep_loss += loss

    return ep_loss / itr


def model_gradients(model):
    """Collect gradients from VGGWithBN layers."""
    grads = {}
    conv_names  = ["Conv1","Conv2","Conv3","Conv4","Conv5","Conv6","Affine1","Affine2"]
    for i, name in enumerate(conv_names, 1):
        grads[f"W{i}"] = model.layers[name].dW
        grads[f"b{i}"] = model.layers[name].db
    for i, name in enumerate([f"BN{j}" for j in range(1, 7)], 1):
        grads[f"gamma{i}"] = model.layers[name].dgamma
        grads[f"beta{i}"]  = model.layers[name].dbeta
    return grads


def run_experiment(
    x_train, t_train, x_test, t_test,
    aug_mode="baseline",
    mixup_alpha=0.4,
    cutmix_alpha=1.0,
    epochs=15,
    batch_size=256,
    lr=0.001,
    output_dir=".",
    seed=0,
):
    """Train VGGWithBN with the specified augmentation for *epochs* epochs.

    Parameters
    ----------
    aug_mode:
        ``"baseline"``, ``"mixup"``, or ``"cutmix"``.
    mixup_alpha:
        Beta distribution parameter for Mixup.
    cutmix_alpha:
        Beta distribution parameter for CutMix.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Adam learning rate.
    output_dir:
        Directory to save best model pkl.
    seed:
        Random seed for weight initialisation and data shuffling.

    Returns
    -------
    dict:
        ``{
          "train_acc_hist": [...],
          "test_acc_hist":  [...],
          "loss_hist":      [...],
          "best_test_acc":  float,
          "final_test_acc": float,
          "time_s":         float,
        }``
    """
    np.random.seed(seed)
    alpha = mixup_alpha if aug_mode == "mixup" else cutmix_alpha

    model = _build_model()
    opt = Adam(lr=lr)

    # LR schedule (mild): decay at epoch 10 and 13
    lr_schedule = {10: 1e-4, 13: 1e-5}

    train_acc_hist, test_acc_hist, loss_hist = [], [], []
    best_test_acc = 0.0
    t_start = time.time()

    print(f"\n{'='*55}")
    print(f"Augmentation: {aug_mode.upper()}  (alpha={alpha})")
    print(f"  epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"{'='*55}")

    for ep in range(epochs):
        if ep in lr_schedule:
            opt.lr = lr_schedule[ep]
            print(f"  LR → {lr_schedule[ep]:.2e}")

        t0 = time.time()
        ep_loss = _train_epoch(model, x_train, t_train, opt, batch_size,
                               aug_mode=aug_mode, alpha=alpha)
        loss_hist.append(ep_loss)

        # Evaluate on subset for speed
        eval_n = min(1000, len(x_train))
        te_n   = min(500,  len(x_test))
        tr_idx = np.random.choice(len(x_train), eval_n, replace=False)
        te_idx = np.random.choice(len(x_test),  te_n,   replace=False)
        tr_acc = model.accuracy(x_train[tr_idx], t_train[tr_idx], batch_size=128)
        te_acc = model.accuracy(x_test[te_idx],  t_test[te_idx],  batch_size=128)

        train_acc_hist.append(tr_acc)
        test_acc_hist.append(te_acc)

        elapsed = time.time() - t0
        print(
            f"  epoch {ep+1:3d}/{epochs}  loss={ep_loss:.4f}  "
            f"train={tr_acc:.4f}  test={te_acc:.4f}  ({elapsed:.0f}s)"
        )

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            pkl_name = f"VGGWithBN_cifar10_{aug_mode}_best.pkl"
            with open(os.path.join(output_dir, pkl_name), "wb") as fh:
                pickle.dump(model.params, fh)

    return {
        "train_acc_hist": train_acc_hist,
        "test_acc_hist":  test_acc_hist,
        "loss_hist":      loss_hist,
        "best_test_acc":  round(best_test_acc, 4),
        "final_test_acc": round(test_acc_hist[-1], 4) if test_acc_hist else 0.0,
        "time_s":         round(time.time() - t_start, 1),
    }


# ===========================================================================
# Plotting
# ===========================================================================

def plot_comparison(results: dict, output_path: str) -> None:
    """Plot test-accuracy learning curves for all three augmentation modes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"baseline": "tab:blue", "mixup": "tab:orange", "cutmix": "tab:green"}
    labels = {
        "baseline": f"Baseline (mild aug)",
        "mixup":    f"Mixup (α=0.4)",
        "cutmix":   f"CutMix (α=1.0)",
    }

    for mode, metrics in results.items():
        epochs_x = list(range(1, len(metrics["test_acc_hist"]) + 1))
        color = colors.get(mode, "gray")
        label = labels.get(mode, mode)
        axes[0].plot(epochs_x, metrics["test_acc_hist"], label=label,
                     color=color, linewidth=2)
        axes[1].plot(epochs_x, metrics["loss_hist"], label=label,
                     color=color, linewidth=2, linestyle="--")

    axes[0].set_title("Test Accuracy: Baseline vs Mixup vs CutMix")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    axes[1].set_title("Training Loss: Baseline vs Mixup vs CutMix")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {output_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("C16 — Mixup vs CutMix Augmentation Comparison (CIFAR-10)")
    print("=" * 60)

    # Load data
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    x_train, x_test, _, _ = normalize(x_train, x_test)
    print(f"Data: train={len(x_train)}  test={len(x_test)}")

    # Run three experiments
    all_results = {}

    # 1. Baseline
    all_results["baseline"] = run_experiment(
        x_train, t_train, x_test, t_test,
        aug_mode="baseline",
        epochs=15, batch_size=256, lr=0.001,
        output_dir=OUTPUT_DIR, seed=42,
    )

    # 2. Mixup
    all_results["mixup"] = run_experiment(
        x_train, t_train, x_test, t_test,
        aug_mode="mixup", mixup_alpha=0.4,
        epochs=15, batch_size=256, lr=0.001,
        output_dir=OUTPUT_DIR, seed=42,
    )

    # 3. CutMix
    all_results["cutmix"] = run_experiment(
        x_train, t_train, x_test, t_test,
        aug_mode="cutmix", cutmix_alpha=1.0,
        epochs=15, batch_size=256, lr=0.001,
        output_dir=OUTPUT_DIR, seed=42,
    )

    # Plot
    plot_path = os.path.join(OUTPUT_DIR, "aug_comparison_curves.png")
    plot_comparison(all_results, plot_path)

    # Save JSON
    json_summary = {
        mode: {
            "best_test_acc":  metrics["best_test_acc"],
            "final_test_acc": metrics["final_test_acc"],
            "time_s":         metrics["time_s"],
        }
        for mode, metrics in all_results.items()
    }
    json_path = os.path.join(OUTPUT_DIR, "aug_comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Results saved: {json_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    header = f"{'Augmentation':<12}  {'Best Acc':>10}  {'Final Acc':>12}  {'Time (s)':>10}"
    print(header)
    print("-" * len(header))
    for mode, metrics in all_results.items():
        print(
            f"{mode:<12}  {metrics['best_test_acc']:>10.4f}  "
            f"{metrics['final_test_acc']:>12.4f}  {metrics['time_s']:>10.1f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
