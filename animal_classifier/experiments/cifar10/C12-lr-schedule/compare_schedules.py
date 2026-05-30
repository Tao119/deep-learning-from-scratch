"""
C12: LR Schedule Comparison on VGGWithBN / CIFAR-10

Three schedules, each trained for 15 epochs:
  1. StepLR       — decay by 0.1 at epoch 10
  2. CosineAnnealing — lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*t/T))
  3. WarmupCosine — linear warmup for 3 epochs, then cosine decay

Saves lr_schedules.png with 4 subplots:
  - LR curves for all schedules
  - Training loss curves
  - Train accuracy curves
  - Test accuracy curves
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from animal_classifier.dataset.cifar10 import load_cifar10, normalize
from animal_classifier.dataset.augmentation import batch_augment
from animal_classifier.models.vgg_bn import VGGWithBN
from common.optimizer import Adam


EPOCHS = 15
BATCH_SIZE = 128
LR_MAX = 0.001
LR_MIN = 1e-5
WARMUP_EPOCHS = 3


def step_lr(epoch, lr_max=LR_MAX, decay_epoch=10, factor=0.1):
    return lr_max * (factor if epoch >= decay_epoch else 1.0)


def cosine_lr(epoch, T=EPOCHS, lr_max=LR_MAX, lr_min=LR_MIN):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T))


def warmup_cosine_lr(epoch, T=EPOCHS, warmup=WARMUP_EPOCHS, lr_max=LR_MAX, lr_min=LR_MIN):
    if epoch < warmup:
        return lr_max * (epoch + 1) / warmup
    t = epoch - warmup
    T_cos = T - warmup
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T_cos))


SCHEDULES = {
    "StepLR":        step_lr,
    "CosineAnnealing": cosine_lr,
    "WarmupCosine":  warmup_cosine_lr,
}


def _accuracy(model, x, t, batch_size=128):
    acc, n = 0, (len(x) // batch_size) * batch_size
    for i in range(0, n, batch_size):
        y = model.predict(x[i:i+batch_size], train_flg=False)
        acc += np.sum(np.argmax(y, axis=1) == t[i:i+batch_size])
    return acc / n if n > 0 else 0.0


def train_with_schedule(x_train, t_train, x_test, t_test, lr_fn, schedule_name):
    model = VGGWithBN(input_channels=3, input_size=32, output_size=10)
    optimizer = Adam(lr=lr_fn(0))

    train_size = len(x_train)
    iter_per_epoch = max(train_size // BATCH_SIZE, 1)

    lr_hist, loss_hist, train_acc_hist, test_acc_hist = [], [], [], []

    print(f"\n{'='*55}")
    print(f"  Schedule: {schedule_name}")
    print(f"{'='*55}")

    for epoch in range(EPOCHS):
        lr = lr_fn(epoch)
        optimizer.lr = lr
        lr_hist.append(lr)

        epoch_loss = 0.0
        t0 = time.time()
        idx = np.random.permutation(train_size)

        for i in range(iter_per_epoch):
            bi = idx[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            x_b = batch_augment(x_train[bi], train_flg=True)
            t_b = t_train[bi]
            grads = model.gradient(x_b, t_b)
            optimizer.update(model.params, grads)
            epoch_loss += model.last_layer.loss

        epoch_loss /= iter_per_epoch
        loss_hist.append(epoch_loss)

        eval_n = 500
        tr_idx = np.random.choice(len(x_train), eval_n, replace=False)
        te_idx = np.random.choice(len(x_test),  eval_n, replace=False)
        tr_acc = _accuracy(model, x_train[tr_idx], t_train[tr_idx])
        te_acc = _accuracy(model, x_test[te_idx],  t_test[te_idx])
        train_acc_hist.append(tr_acc)
        test_acc_hist.append(te_acc)

        elapsed = time.time() - t0
        print(f"  epoch {epoch+1:3d}/{EPOCHS}: lr={lr:.2e}  loss={epoch_loss:.4f}  "
              f"train={tr_acc:.4f}  test={te_acc:.4f}  ({elapsed:.0f}s)")

    return lr_hist, loss_hist, train_acc_hist, test_acc_hist


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading CIFAR-10 ...")
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    x_train, x_test, _, _ = normalize(x_train, x_test)
    print(f"train: {x_train.shape}  test: {x_test.shape}")

    results = {}
    for name, fn in SCHEDULES.items():
        results[name] = train_with_schedule(x_train, t_train, x_test, t_test, fn, name)

    epochs = range(1, EPOCHS + 1)
    colors = {"StepLR": "steelblue", "CosineAnnealing": "darkorange", "WarmupCosine": "forestgreen"}
    styles = {"StepLR": "-", "CosineAnnealing": "--", "WarmupCosine": ":"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for name, (lr_h, loss_h, tr_h, te_h) in results.items():
        c, ls = colors[name], styles[name]
        axes[0, 0].plot(epochs, lr_h,   label=name, color=c, linestyle=ls)
        axes[0, 1].plot(epochs, loss_h, label=name, color=c, linestyle=ls)
        axes[1, 0].plot(epochs, tr_h,   label=name, color=c, linestyle=ls)
        axes[1, 1].plot(epochs, te_h,   label=name, color=c, linestyle=ls)

    axes[0, 0].set_title("Learning Rate Curves")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("lr")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()

    axes[0, 1].set_title("Training Loss")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("loss")
    axes[0, 1].legend()

    axes[1, 0].set_title("Train Accuracy")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 0].legend()

    axes[1, 1].set_title("Test Accuracy")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("accuracy")
    axes[1, 1].legend()

    plt.suptitle("C12: LR Schedule Comparison — VGGWithBN / CIFAR-10 (15 epochs)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "lr_schedules.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved {out_path}")

    print("\nFinal test accuracy by schedule:")
    for name, (_, _, _, te_h) in results.items():
        print(f"  {name:<22}: {te_h[-1]:.4f}  (best={max(te_h):.4f})")


if __name__ == "__main__":
    main()
