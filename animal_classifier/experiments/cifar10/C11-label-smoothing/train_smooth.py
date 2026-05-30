"""
C11: Label Smoothing vs Standard Cross-Entropy on VGGWithBN / CIFAR-10

Compares:
  - Standard CE (one-hot targets, as in C02)
  - Label smoothing with epsilon=0.1
    y_smooth = (1-eps)*one_hot + eps/K

Loss for label smoothing: -sum(y_smooth * log(p))
  = (1-eps)*CE + eps * (-sum(log(p)/K))
  = standard KL divergence between y_smooth and p
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

from animal_classifier.dataset.cifar10 import load_cifar10, normalize, CLASS_NAMES
from animal_classifier.dataset.augmentation import batch_augment
from animal_classifier.models.vgg_bn import VGGWithBN
from common.optimizer import Adam
from common.activation import softmax


NUM_CLASSES = 10
EPOCHS = 30
BATCH_SIZE = 128
LR = 0.001
LR_DECAY_AT = {20: 1e-4, 25: 1e-5}
EPSILON = 0.1


def label_smooth(t_int, n_classes, eps):
    """
    Convert integer labels to smoothed soft labels.

    y_smooth = (1 - eps) * one_hot + eps / K
    """
    one_hot = np.zeros((len(t_int), n_classes), dtype=np.float32)
    one_hot[np.arange(len(t_int)), t_int] = 1.0
    return (1.0 - eps) * one_hot + eps / n_classes


def smooth_loss(logits, t_int, n_classes, eps):
    """
    KL-divergence-based label-smoothing loss.

    L = -sum_k( y_smooth_k * log(p_k) )
    """
    p = softmax(logits)
    p = np.clip(p, 1e-7, 1.0)
    y_s = label_smooth(t_int, n_classes, eps)
    return -np.mean(np.sum(y_s * np.log(p), axis=1))


def smooth_backward(logits, t_int, n_classes, eps):
    """
    Gradient of smooth_loss w.r.t. logits.

    d_L/d_logit_j = p_j - y_smooth_j
    averaged over batch.
    """
    p = softmax(logits)
    y_s = label_smooth(t_int, n_classes, eps)
    return (p - y_s) / len(t_int)


def _accuracy(model, x, t, batch_size=128):
    acc, n = 0, (len(x) // batch_size) * batch_size
    for i in range(0, n, batch_size):
        y = model.predict(x[i:i+batch_size], train_flg=False)
        acc += np.sum(np.argmax(y, axis=1) == t[i:i+batch_size])
    return acc / n if n > 0 else 0.0


def train_one_config(x_train, t_train, x_test, t_test, use_smoothing, label):
    model = VGGWithBN(input_channels=3, input_size=32, output_size=NUM_CLASSES)
    optimizer = Adam(lr=LR)
    train_size = len(x_train)
    iter_per_epoch = max(train_size // BATCH_SIZE, 1)

    loss_hist, train_acc_hist, test_acc_hist = [], [], []
    current_lr = LR

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    for epoch in range(EPOCHS):
        if epoch in LR_DECAY_AT:
            current_lr = LR_DECAY_AT[epoch]
            optimizer.lr = current_lr
            print(f"  LR -> {current_lr:.2e}")

        epoch_loss = 0.0
        t0 = time.time()
        idx = np.random.permutation(train_size)

        for i in range(iter_per_epoch):
            bi = idx[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            x_b = batch_augment(x_train[bi], train_flg=True)
            t_b = t_train[bi]

            if use_smoothing:
                # Forward through all layers up to the last linear layer
                logits = model.predict(x_b, train_flg=True)
                loss_val = smooth_loss(logits, t_b, NUM_CLASSES, EPSILON)

                # Inject custom gradient into last layer's backward
                dlogits = smooth_backward(logits, t_b, NUM_CLASSES, EPSILON)
                # Store loss so train loop can read model.last_layer.loss
                model.last_layer.loss = loss_val
                model.last_layer.y = softmax(logits)
                model.last_layer.t = t_b

                # Run backward from dlogits (skip SoftmaxWithLoss.backward)
                dout = dlogits
                for layer in reversed(list(model.layers.values())):
                    dout = layer.backward(dout)

                grads = {}
                conv_names = ["Conv1","Conv2","Conv3","Conv4","Conv5","Conv6","Affine1","Affine2"]
                for j, name in enumerate(conv_names, 1):
                    grads[f"W{j}"] = model.layers[name].dW
                    grads[f"b{j}"] = model.layers[name].db
                for j, name in enumerate([f"BN{k}" for k in range(1, 7)], 1):
                    grads[f"gamma{j}"] = model.layers[name].dgamma
                    grads[f"beta{j}"]  = model.layers[name].dbeta
                optimizer.update(model.params, grads)
            else:
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
        print(f"  epoch {epoch+1:3d}/{EPOCHS}: loss={epoch_loss:.4f}  "
              f"train={tr_acc:.4f}  test={te_acc:.4f}  ({elapsed:.0f}s)")

    return loss_hist, train_acc_hist, test_acc_hist


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading CIFAR-10 ...")
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    x_train, x_test, _, _ = normalize(x_train, x_test)
    print(f"train: {x_train.shape}  test: {x_test.shape}")

    ce_hist    = train_one_config(x_train, t_train, x_test, t_test,
                                  use_smoothing=False, label="Standard CE (baseline)")
    smooth_hist = train_one_config(x_train, t_train, x_test, t_test,
                                   use_smoothing=True,  label=f"Label Smoothing (eps={EPSILON})")

    ce_loss,    ce_train,    ce_test    = ce_hist
    sm_loss,    sm_train,    sm_test    = smooth_hist

    epochs = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, ce_loss,  label="Standard CE")
    axes[0].plot(epochs, sm_loss,  label=f"Label Smooth (eps={EPSILON})", linestyle="--")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    axes[1].plot(epochs, ce_train,  label="CE train")
    axes[1].plot(epochs, sm_train,  label="Smooth train", linestyle="--")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Train Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, ce_test,  label="CE test")
    axes[2].plot(epochs, sm_test,  label="Smooth test", linestyle="--")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("accuracy")
    axes[2].set_title("Test Accuracy")
    axes[2].legend()

    plt.suptitle(f"C11: Label Smoothing (eps={EPSILON}) vs Standard CE — VGGWithBN/CIFAR-10",
                 fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "label_smoothing_comparison.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved {out_path}")

    print("\nFinal test accuracy:")
    print(f"  Standard CE       : {ce_test[-1]:.4f}")
    print(f"  Label Smoothing   : {sm_test[-1]:.4f}")
    print(f"  Best CE test      : {max(ce_test):.4f}")
    print(f"  Best Smooth test  : {max(sm_test):.4f}")


if __name__ == "__main__":
    main()
