"""
C17 — Model Ensemble Evaluation on CIFAR-10
============================================
Combines:
  - C02: VGGWithBN  (pkl: C02-vggbn-batchnorm/VGGWithBN_cifar10_mild_best.pkl)
  - C05: MobileNet  (pkl: C05-mobilenet-mild/MobileNet_cifar10_mild_best.pkl)
  - C08: Mixup VGGWithBN (pkl: C08-vggbn-mixup/VGGWithBN_cifar10_mixup_best.pkl)

Ensemble strategies:
  1. Simple average (soft voting)
  2. Majority vote (hard voting)
  3. Weighted by validation accuracy

If pkl files do not exist, random-weight placeholder models are used
(accuracies will be ~10% as expected for random CIFAR-10).

Usage:
    cd animal_classifier
    python experiments/cifar10/C17-ensemble/ensemble_eval.py
"""

import sys
import os

# Resolve paths relative to animal_classifier root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AC_ROOT  = os.path.abspath(os.path.join(_THIS_DIR, "../../.."))   # animal_classifier/
_REPO_ROOT = os.path.abspath(os.path.join(_AC_ROOT, ".."))          # deep-learning-from-scratch/

for p in [_AC_ROOT, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pickle
import numpy as np
from collections import OrderedDict

# ── Dataset ──────────────────────────────────────────────────────────────────

def _load_dataset(n_eval=2000):
    from dataset.cifar10 import load_cifar10, normalize
    (x_train, _), (x_test, t_test) = load_cifar10()
    x_train_n, x_test_n, _, _ = normalize(x_train, x_test)
    n = min(n_eval, x_test_n.shape[0])
    return x_test_n[:n], t_test[:n], x_test[:n]


# ── Model Loaders ─────────────────────────────────────────────────────────────

def _load_vggbn(pkl_path: str, output_size: int = 10):
    """Load VGGWithBN; fall back to random weights if pkl not found."""
    from models.vgg_bn import VGGWithBN
    from common.conv_layers import Convolution, Pooling
    from common.layers import Affine, Relu, SoftmaxWithLoss
    from common.layers_ext import BatchNormalization, Dropout

    model = VGGWithBN(input_channels=3, input_size=32, output_size=output_size)

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            params = pickle.load(f)
        model.params = params
        p = params

        def make_bn(i):
            return BatchNormalization(p[f"gamma{i}"], p[f"beta{i}"])

        model.layers = OrderedDict([
            ("Conv1",   Convolution(p["W1"], p["b1"], stride=1, pad=1)),
            ("BN1",     make_bn(1)),
            ("Relu1",   Relu()),
            ("Conv2",   Convolution(p["W2"], p["b2"], stride=1, pad=1)),
            ("BN2",     make_bn(2)),
            ("Relu2",   Relu()),
            ("Pool1",   Pooling(2, 2, stride=2)),
            ("Conv3",   Convolution(p["W3"], p["b3"], stride=1, pad=1)),
            ("BN3",     make_bn(3)),
            ("Relu3",   Relu()),
            ("Conv4",   Convolution(p["W4"], p["b4"], stride=1, pad=1)),
            ("BN4",     make_bn(4)),
            ("Relu4",   Relu()),
            ("Pool2",   Pooling(2, 2, stride=2)),
            ("Conv5",   Convolution(p["W5"], p["b5"], stride=1, pad=1)),
            ("BN5",     make_bn(5)),
            ("Relu5",   Relu()),
            ("Conv6",   Convolution(p["W6"], p["b6"], stride=1, pad=1)),
            ("BN6",     make_bn(6)),
            ("Relu6",   Relu()),
            ("Pool3",   Pooling(2, 2, stride=2)),
            ("Affine1", Affine(p["W7"], p["b7"])),
            ("Relu7",   Relu()),
            ("Drop1",   Dropout(0.5)),
            ("Affine2", Affine(p["W8"], p["b8"])),
        ])
        loaded = True
    else:
        loaded = False

    return model, loaded


def _load_mobilenet(pkl_path: str, output_size: int = 10):
    """Load MobileNet; fall back to random weights if pkl not found."""
    from models.mobilenet import MobileNet

    model = MobileNet(input_channels=3, output_size=output_size)
    loaded = False

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            params = pickle.load(f)
        p_ref, _ = model.get_params_and_grads()
        for key in p_ref:
            if key in params:
                p_ref[key][:] = params[key]
        loaded = True

    return model, loaded


# ── Softmax helper ────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── Inference ─────────────────────────────────────────────────────────────────

def _predict_probs(model, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Run batched inference and return softmax probabilities (N, K)."""
    probs_list = []
    n = (x.shape[0] // batch_size) * batch_size
    for i in range(0, n, batch_size):
        logits = model.predict(x[i:i + batch_size], train_flg=False)
        probs_list.append(_softmax(logits))
    return np.vstack(probs_list), n


# ── Ensemble strategies ───────────────────────────────────────────────────────

def _majority_vote(probs_list: list[np.ndarray]) -> np.ndarray:
    votes = np.stack([np.argmax(p, axis=1) for p in probs_list], axis=1)  # (N, M)
    return np.apply_along_axis(
        lambda row: np.bincount(row, minlength=probs_list[0].shape[1]).argmax(),
        axis=1, arr=votes
    )


def _soft_average(probs_list: list[np.ndarray]) -> np.ndarray:
    return np.argmax(np.stack(probs_list).mean(axis=0), axis=1)


def _weighted_average(probs_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.array(weights, dtype=float)
    w /= w.sum()
    stacked = np.stack(probs_list, axis=0)          # (M, N, K)
    wavg = (stacked * w[:, None, None]).sum(axis=0) # (N, K)
    return np.argmax(wavg, axis=1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cifar_base = os.path.join(_THIS_DIR, "..")

    # --- PKL paths ---
    pkl_c02 = os.path.join(cifar_base, "C02-vggbn-batchnorm",
                           "VGGWithBN_cifar10_mild_best.pkl")
    pkl_c05 = os.path.join(cifar_base, "C05-mobilenet-mild",
                           "MobileNet_cifar10_mild_best.pkl")
    pkl_c08 = os.path.join(cifar_base, "C08-vggbn-mixup",
                           "VGGWithBN_cifar10_mixup_best.pkl")

    print("=" * 64)
    print("C17 — CIFAR-10 Model Ensemble Evaluation")
    print("=" * 64)

    # --- Load dataset ---
    print("\nLoading CIFAR-10 test set (2000 samples) ...")
    x_eval, t_eval, _ = _load_dataset(n_eval=2000)
    print(f"Eval set: {x_eval.shape}, labels: {t_eval.shape}")

    # --- Load models ---
    print("\nLoading models ...")
    m_c02, ok_c02 = _load_vggbn(pkl_c02)
    tag_c02 = "C02-VGGWithBN(trained)" if ok_c02 else "C02-VGGWithBN(random)"
    print(f"  {tag_c02}")

    m_c05, ok_c05 = _load_mobilenet(pkl_c05)
    tag_c05 = "C05-MobileNet(trained)" if ok_c05 else "C05-MobileNet(random)"
    print(f"  {tag_c05}")

    m_c08, ok_c08 = _load_vggbn(pkl_c08)
    tag_c08 = "C08-VGGWithBN-Mixup(trained)" if ok_c08 else "C08-VGGWithBN-Mixup(random)"
    print(f"  {tag_c08}")

    # --- Collect probabilities ---
    print("\nRunning inference ...")
    probs_c02, n = _predict_probs(m_c02, x_eval)
    probs_c05, _  = _predict_probs(m_c05, x_eval)
    probs_c08, _  = _predict_probs(m_c08, x_eval)

    t = t_eval[:n]
    probs_list = [probs_c02, probs_c05, probs_c08]
    names = [tag_c02, tag_c05, tag_c08]

    # Validation accuracies used as weights
    # If trained, use documented accuracies; if random use 1.0 equally
    val_accs_map = {
        "C02-VGGWithBN(trained)": 0.82,
        "C05-MobileNet(trained)": 0.75,
        "C08-VGGWithBN-Mixup(trained)": 0.83,
    }
    weights = [val_accs_map.get(n_, 1.0 / 10) for n_ in names]

    # --- Per-model accuracy ---
    per_model_acc = {}
    for name, probs in zip(names, probs_list):
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == t))
        per_model_acc[name] = acc

    # --- Ensemble strategies ---
    acc_majority  = float(np.mean(_majority_vote(probs_list) == t))
    acc_average   = float(np.mean(_soft_average(probs_list) == t))
    acc_weighted  = float(np.mean(_weighted_average(probs_list, weights) == t))

    best_individual = max(per_model_acc.values())
    ensemble_best   = max(acc_majority, acc_average, acc_weighted)

    # --- Report ---
    print()
    col = 42
    header = f"{'Model/Strategy':<{col}} {'Accuracy':>10}"
    print(header)
    print("-" * (col + 12))
    for name, acc in per_model_acc.items():
        print(f"{name:<{col}} {acc:>10.4f}")
    print("-" * (col + 12))
    print(f"{'Ensemble: majority vote':<{col}} {acc_majority:>10.4f}")
    print(f"{'Ensemble: soft average':<{col}} {acc_average:>10.4f}")
    print(f"{'Ensemble: weighted avg (by val acc)':<{col}} {acc_weighted:>10.4f}")
    print("=" * (col + 12))

    improvement = ensemble_best - best_individual
    if improvement > 0:
        print(f"\nEnsemble improves over best individual by {improvement:+.4f} ({improvement*100:+.2f}%)")
    elif improvement == 0:
        print(f"\nEnsemble matches best individual ({best_individual:.4f}) — no further gain.")
    else:
        print(f"\nNote: with random weights, ensemble cannot improve over trained models.")
        print("Re-run after training C02/C05/C08 for meaningful gains.")

    print("\nWeights used for weighted ensemble:")
    for name, w in zip(names, weights):
        print(f"  {name}: {w:.4f}")

    return {
        "per_model": per_model_acc,
        "majority": acc_majority,
        "average": acc_average,
        "weighted": acc_weighted,
    }


if __name__ == "__main__":
    results = main()
