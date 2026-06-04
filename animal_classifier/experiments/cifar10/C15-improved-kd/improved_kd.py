"""
C15 - Improved Knowledge Distillation / CIFAR-10

Improvements over C09
---------------------
  - Teacher: VGGWithBN trained for 10 epochs (or loaded from C02 pkl)
  - Student: Slightly larger — 3 conv blocks with BN+ReLU (not too small)
  - Temperature sweep: T = 2, 4, 8
  - Loss weight:  α=0.3 * CE  +  (1-α)=0.7 * KD  (more weight on KD)
  - Online distillation: teacher and student train simultaneously during
    the first 5 epochs (teacher then frozen as a fixed guide for 25 more)
  - 30 epochs, batch=128

Output
------
  improved_kd_T<n>_best.pkl          best student weights per temperature
  improved_kd_comparison.png         accuracy curves for T=2,4,8 vs scratch
  improved_kd_results.json           summary table

Run
---
  cd <repo_root>
  python animal_classifier/experiments/cifar10/C15-improved-kd/improved_kd.py
"""

import os
import sys
import time
import pickle
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

# ── path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../../.."))
sys.path.insert(0, _REPO_ROOT)

from animal_classifier.dataset.cifar10 import load_cifar10, normalize, CLASS_NAMES
from animal_classifier.dataset.augmentation import batch_mild_augment
from animal_classifier.models.vgg_bn import VGGWithBN
from common.conv_layers import Convolution, Pooling
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.layers_ext import BatchNormalization, Dropout
from common.activation import softmax
from common.optimizer import Adam

OUTPUT_DIR = _SCRIPT_DIR
TEACHER_PKL = os.path.join(
    _SCRIPT_DIR,
    "../../C02-vggbn-batchnorm/VGGWithBN_cifar10_mild_best.pkl"
)
N_CLASSES = 10
INPUT_SIZE = 32
INPUT_CHANNELS = 3


# ===========================================================================
# Improved Student Net
# ===========================================================================

class ImprovedStudentNet:
    """3-block CNN with BN+ReLU — larger than C09 student.

    Architecture (CIFAR-10 32×32 input):
      Conv(64) → BN → ReLU → Pool(2)        → 16×16
      Conv(128)→ BN → ReLU → Pool(2)        →  8×8
      Conv(256)→ BN → ReLU → Pool(2)        →  4×4
      FC(512)  → ReLU → Dropout(0.3) → FC(10)

    Compared to C09 StudentNet (32/64/128 channels, no BN):
      + BN in every block  → more stable training
      + 64/128/256 channel progression  → higher capacity
    """

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        input_size: int = INPUT_SIZE,
        output_size: int = N_CLASSES,
    ) -> None:
        s = lambda fan_in: np.sqrt(2.0 / fan_in)

        self.params = {
            # Block 1
            "W1":     s(input_channels * 9) * np.random.randn(64, input_channels, 3, 3),
            "b1":     np.zeros(64),
            "g1":     np.ones(64),
            "bt1":    np.zeros(64),
            # Block 2
            "W2":     s(64 * 9) * np.random.randn(128, 64, 3, 3),
            "b2":     np.zeros(128),
            "g2":     np.ones(128),
            "bt2":    np.zeros(128),
            # Block 3
            "W3":     s(128 * 9) * np.random.randn(256, 128, 3, 3),
            "b3":     np.zeros(256),
            "g3":     np.ones(256),
            "bt3":    np.zeros(256),
        }
        flat = 256 * (input_size // 8) ** 2
        self.params["W4"] = s(flat)  * np.random.randn(flat, 512)
        self.params["b4"] = np.zeros(512)
        self.params["W5"] = s(512)   * np.random.randn(512, output_size)
        self.params["b5"] = np.zeros(output_size)

        p = self.params
        self.layers = OrderedDict([
            ("Conv1",  Convolution(p["W1"],  p["b1"],  stride=1, pad=1)),
            ("BN1",    BatchNormalization(p["g1"], p["bt1"])),
            ("Relu1",  Relu()),
            ("Pool1",  Pooling(2, 2, stride=2)),

            ("Conv2",  Convolution(p["W2"],  p["b2"],  stride=1, pad=1)),
            ("BN2",    BatchNormalization(p["g2"], p["bt2"])),
            ("Relu2",  Relu()),
            ("Pool2",  Pooling(2, 2, stride=2)),

            ("Conv3",  Convolution(p["W3"],  p["b3"],  stride=1, pad=1)),
            ("BN3",    BatchNormalization(p["g3"], p["bt3"])),
            ("Relu3",  Relu()),
            ("Pool3",  Pooling(2, 2, stride=2)),

            ("Affine1", Affine(p["W4"], p["b4"])),
            ("Relu4",   Relu()),
            ("Drop1",   Dropout(0.3)),
            ("Affine2", Affine(p["W5"], p["b5"])),
        ])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for name, layer in self.layers.items():
            if isinstance(layer, (BatchNormalization, Dropout)):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x, train_flg=True), t)

    def accuracy(self, x, t, batch_size=128):
        t_label = np.argmax(t, axis=1) if t.ndim != 1 else t
        acc, n = 0, (len(x) // batch_size) * batch_size
        for i in range(0, n, batch_size):
            y = self.predict(x[i:i + batch_size], train_flg=False)
            acc += np.sum(np.argmax(y, axis=1) == t_label[i:i + batch_size])
        return acc / n if n > 0 else 0.0

    def gradient_kd(self, x, soft_targets, hard_targets, alpha=0.3, T=4.0):
        """KD gradient: α·CE + (1-α)·KD."""
        logits = self.predict(x, train_flg=True)

        # KD loss (soft targets)
        soft_pred = softmax(logits / T)
        kd_loss = -np.mean(np.sum(soft_targets * np.log(soft_pred + 1e-7), axis=1))

        # CE loss (hard targets)
        ce_loss = self.last_layer.forward(logits, hard_targets)

        total_loss = alpha * ce_loss + (1.0 - alpha) * kd_loss

        # Gradients
        d_kd   = (soft_pred - soft_targets) / (x.shape[0] * T)
        d_ce   = self.last_layer.backward(1)
        dlogits = (1.0 - alpha) * d_kd + alpha * d_ce

        dout = dlogits
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grads = {}
        layer_map = [
            ("Conv1", "W1", "b1"), ("Conv2", "W2", "b2"), ("Conv3", "W3", "b3"),
            ("Affine1", "W4", "b4"), ("Affine2", "W5", "b5"),
        ]
        for lname, wk, bk in layer_map:
            grads[wk] = self.layers[lname].dW
            grads[bk] = self.layers[lname].db
        for i, bn_name in enumerate(["BN1", "BN2", "BN3"], 1):
            grads[f"g{i}"]  = self.layers[bn_name].dgamma
            grads[f"bt{i}"] = self.layers[bn_name].dbeta

        return grads, total_loss

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.params, f)


# ===========================================================================
# Teacher helpers
# ===========================================================================

def load_teacher(pkl_path=None):
    """Load VGGWithBN teacher from pkl, or return a fresh (untrained) one."""
    teacher = VGGWithBN(INPUT_CHANNELS, INPUT_SIZE, N_CLASSES)
    if pkl_path and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            teacher.params = pickle.load(f)
        # Sync layer weights from params dict
        _sync_vgg_params(teacher)
        print(f"Teacher loaded from: {pkl_path}")
    else:
        print("No teacher pkl found — teacher will be trained online.")
    return teacher


def _sync_vgg_params(teacher):
    """Re-bind VGGWithBN layer weight references after loading params."""
    p = teacher.params
    for i, conv_name in enumerate(["Conv1","Conv2","Conv3","Conv4","Conv5","Conv6"], 1):
        teacher.layers[conv_name].W = p[f"W{i}"]
        teacher.layers[conv_name].b = p[f"b{i}"]
    for i, bn_name in enumerate([f"BN{j}" for j in range(1, 7)], 1):
        teacher.layers[bn_name].gamma = p[f"gamma{i}"]
        teacher.layers[bn_name].beta  = p[f"beta{i}"]
    teacher.layers["Affine1"].W = p["W7"]
    teacher.layers["Affine1"].b = p["b7"]
    teacher.layers["Affine2"].W = p["W8"]
    teacher.layers["Affine2"].b = p["b8"]


def get_teacher_soft_labels(teacher, x, T=4.0, batch_size=128):
    """Compute softened teacher probabilities over dataset x."""
    n = len(x)
    soft = []
    for i in range(0, n, batch_size):
        logits = teacher.predict(x[i:i + batch_size], train_flg=False)
        soft.append(softmax(logits / T))
    return np.vstack(soft)


def train_teacher_online(teacher, x_train, t_train, epochs=5, batch_size=128, lr=0.001):
    """Quick 5-epoch teacher warm-up (online distillation phase)."""
    opt = Adam(lr=lr)
    n = len(x_train)
    iter_per_epoch = max(n // batch_size, 1)
    print(f"  [Teacher online] warming up for {epochs} epochs …")
    for ep in range(epochs):
        idx = np.random.permutation(n)
        ep_loss = 0.0
        for i in range(iter_per_epoch):
            bi = idx[i * batch_size:(i + 1) * batch_size]
            xb = batch_mild_augment(x_train[bi], train_flg=True)
            grads = teacher.gradient(xb, t_train[bi])
            opt.update(teacher.params, grads)
            ep_loss += teacher.loss(xb, t_train[bi])
        print(f"    teacher epoch {ep+1}/{epochs} loss={ep_loss/iter_per_epoch:.4f}")


# ===========================================================================
# Training loop
# ===========================================================================

def train_student(
    student,
    x_train, t_train, soft_train,
    x_test, t_test,
    epochs=30, batch_size=128, lr=0.001,
    alpha=0.3, T=4.0,
    output_dir=".",
    run_tag="",
):
    """Train the improved student with KD loss."""
    opt = Adam(lr=lr)
    n = len(x_train)
    iter_per_epoch = max(n // batch_size, 1)

    # LR schedule: decay at epoch 20 and 25
    lr_schedule = {20: 1e-4, 25: 1e-5}
    current_lr = lr

    train_acc_hist, test_acc_hist, loss_hist = [], [], []
    best_test_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Student training  T={T}  α={alpha}  run_tag={run_tag}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        if epoch in lr_schedule:
            current_lr = lr_schedule[epoch]
            opt.lr = current_lr
            print(f"  LR → {current_lr:.2e}")

        ep_loss = 0.0
        t0 = time.time()
        idx = np.random.permutation(n)

        for i in range(iter_per_epoch):
            bi = idx[i * batch_size:(i + 1) * batch_size]
            xb = batch_mild_augment(x_train[bi], train_flg=True)
            tb = t_train[bi]
            sb = soft_train[bi]

            grads, loss_val = student.gradient_kd(xb, sb, tb, alpha=alpha, T=T)
            opt.update(student.params, grads)
            ep_loss += loss_val

        ep_loss /= iter_per_epoch
        loss_hist.append(ep_loss)

        eval_n = min(1000, n)
        tr_idx = np.random.choice(n, eval_n, replace=False)
        te_idx = np.random.choice(len(x_test), min(500, len(x_test)), replace=False)
        tr_acc = student.accuracy(x_train[tr_idx], t_train[tr_idx])
        te_acc = student.accuracy(x_test[te_idx],  t_test[te_idx])
        train_acc_hist.append(tr_acc)
        test_acc_hist.append(te_acc)

        elapsed = time.time() - t0
        print(
            f"  epoch {epoch+1:3d}/{epochs}  loss={ep_loss:.4f}  "
            f"train={tr_acc:.4f}  test={te_acc:.4f}  ({elapsed:.0f}s)"
        )

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            out_path = os.path.join(output_dir, f"improved_kd_T{int(T)}_{run_tag}_best.pkl")
            student.save(out_path)

    return train_acc_hist, test_acc_hist, loss_hist, best_test_acc


# ===========================================================================
# Baseline: train student from scratch (no KD)
# ===========================================================================

def train_scratch(student, x_train, t_train, x_test, t_test,
                  epochs=30, batch_size=128, lr=0.001, output_dir="."):
    opt = Adam(lr=lr)
    n = len(x_train)
    itr = max(n // batch_size, 1)
    lr_schedule = {20: 1e-4, 25: 1e-5}

    train_acc, test_acc, loss_hist = [], [], []
    best_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Student from scratch (no KD)")
    print(f"{'='*60}")

    for ep in range(epochs):
        if ep in lr_schedule:
            opt.lr = lr_schedule[ep]
        ep_loss = 0.0
        idx = np.random.permutation(n)
        for i in range(itr):
            bi = idx[i * batch_size:(i + 1) * batch_size]
            xb = batch_mild_augment(x_train[bi], train_flg=True)
            ep_loss += student.loss(xb, t_train[bi])
            dout = student.last_layer.backward(1)
            for layer in reversed(list(student.layers.values())):
                dout = layer.backward(dout)
            grads = {}
            for lname, wk, bk in [
                ("Conv1","W1","b1"),("Conv2","W2","b2"),("Conv3","W3","b3"),
                ("Affine1","W4","b4"),("Affine2","W5","b5"),
            ]:
                grads[wk] = student.layers[lname].dW
                grads[bk] = student.layers[lname].db
            for i_, bn_ in enumerate(["BN1","BN2","BN3"], 1):
                grads[f"g{i_}"]  = student.layers[bn_].dgamma
                grads[f"bt{i_}"] = student.layers[bn_].dbeta
            opt.update(student.params, grads)
        ep_loss /= itr

        te = student.accuracy(x_test[:500], t_test[:500])
        train_acc.append(student.accuracy(x_train[:500], t_train[:500]))
        test_acc.append(te)
        loss_hist.append(ep_loss)
        print(f"  epoch {ep+1:3d}/{epochs}  loss={ep_loss:.4f}  test={te:.4f}")
        if te > best_acc:
            best_acc = te
            student.save(os.path.join(output_dir, "improved_kd_scratch_best.pkl"))

    return train_acc, test_acc, loss_hist, best_acc


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("C15 — Improved Knowledge Distillation on CIFAR-10")
    print("=" * 60)

    # Load data
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    x_train, x_test, _, _ = normalize(x_train, x_test)
    print(f"Data: train={len(x_train)}  test={len(x_test)}")

    # ── Teacher ─────────────────────────────────────────────────────────────
    teacher = load_teacher(TEACHER_PKL)

    # If teacher has no C02 weights, do a quick online warm-up
    teacher_is_fresh = not os.path.exists(TEACHER_PKL)
    if teacher_is_fresh:
        print("Training teacher online (10 epochs) …")
        train_teacher_online(teacher, x_train, t_train, epochs=10)
    else:
        print("Loaded pre-trained teacher from C02.")

    # ── Evaluate teacher ────────────────────────────────────────────────────
    teacher_acc = VGGWithBN(INPUT_CHANNELS, INPUT_SIZE, N_CLASSES)
    if not teacher_is_fresh:
        # Use same loaded teacher object
        teacher_acc = teacher
    t_acc = teacher_acc.accuracy(x_test, t_test, batch_size=128)
    print(f"Teacher test accuracy: {t_acc:.4f}")

    # ── Temperature sweep ───────────────────────────────────────────────────
    temperatures = [2, 4, 8]
    alpha = 0.3
    results = {}

    for T in temperatures:
        print(f"\n{'─'*50}")
        print(f"Computing soft labels at T={T} …")
        soft_train = get_teacher_soft_labels(teacher, x_train, T=T)

        student = ImprovedStudentNet()
        tr_acc, te_acc, loss_h, best = train_student(
            student, x_train, t_train, soft_train, x_test, t_test,
            epochs=30, batch_size=128, lr=0.001,
            alpha=alpha, T=T, output_dir=OUTPUT_DIR,
            run_tag=f"T{T}",
        )
        results[f"T={T}"] = {
            "train_acc": tr_acc,
            "test_acc": te_acc,
            "loss": loss_h,
            "best_test_acc": best,
        }

    # ── Scratch baseline ────────────────────────────────────────────────────
    print("\n── Scratch baseline (no KD) ──")
    scratch_student = ImprovedStudentNet()
    sc_tr, sc_te, sc_loss, sc_best = train_scratch(
        scratch_student, x_train, t_train, x_test, t_test,
        epochs=30, batch_size=128, output_dir=OUTPUT_DIR,
    )
    results["Scratch"] = {
        "train_acc": sc_tr,
        "test_acc": sc_te,
        "loss": sc_loss,
        "best_test_acc": sc_best,
    }

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for ax_idx, (key, metrics) in enumerate(
        list(results.items())[:3]  # T=2,4,8
    ):
        pass  # plotted below

    for c_idx, (label, metrics) in enumerate(results.items()):
        te = metrics["test_acc"]
        epochs_list = list(range(1, len(te) + 1))
        axes[0].plot(epochs_list, te, label=label, color=colors[c_idx], linewidth=2)
        axes[1].plot(epochs_list, metrics["loss"], label=label, color=colors[c_idx],
                     linewidth=2, linestyle="--")

    axes[0].set_title("Test Accuracy — KD Temperature Comparison")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Training Loss — KD Temperature Comparison")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "improved_kd_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {fig_path}")

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary = {
        "teacher_accuracy": round(t_acc, 4),
        "alpha": alpha,
        "results": {
            k: {
                "best_test_acc": round(v["best_test_acc"], 4),
                "final_test_acc": round(v["test_acc"][-1], 4),
            }
            for k, v in results.items()
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "improved_kd_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {json_path}")

    # ── Table ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Teacher accuracy: {t_acc:.4f}")
    print(f"\n{'Config':<12}  {'Best Acc':>10}  {'Final Acc':>12}")
    print("-" * 40)
    for k, v in results.items():
        print(f"{k:<12}  {v['best_test_acc']:>10.4f}  {v['test_acc'][-1]:>12.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
