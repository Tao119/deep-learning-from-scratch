import sys
sys.path.append("..")
import os
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset.cifar10 import load_cifar10, normalize, CLASS_NAMES
from dataset.augmentation import batch_mild_augment, mixup, cutmix
from models.vgg_bn import VGGWithBN
from common.optimizer import Adam
from utils.visualization import plot_confusion_matrix, plot_sample_predictions, plot_top5_accuracy


def train_with_mixup(model, model_name, x_train, t_train, x_test, t_test,
                     epochs=30, batch_size=256, lr=0.001,
                     lr_decay_at=(20, 25), lr_decay_factor=0.1,
                     mix_fn=mixup, mix_alpha=0.4,
                     output_dir="."):

    optimizer = Adam(lr=lr)
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)
    num_classes = 10

    train_acc_hist, test_acc_hist, loss_hist = [], [], []
    best_test_acc = 0.0
    current_lr = lr

    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"  train={train_size}, batch={batch_size}, epochs={epochs}, lr={lr}")
    print(f"  mix_fn={mix_fn.__name__}, alpha={mix_alpha}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        if epoch in lr_decay_at:
            current_lr *= lr_decay_factor
            optimizer.lr = current_lr
            print(f"  LR decayed to {current_lr:.2e}")

        epoch_loss = 0.0
        t0 = time.time()
        idx = np.random.permutation(train_size)

        for i in range(iter_per_epoch):
            batch_idx = idx[i * batch_size:(i + 1) * batch_size]
            x_batch = x_train[batch_idx]
            t_batch = t_train[batch_idx]

            x_batch = batch_mild_augment(x_batch, train_flg=True)
            x_batch, t_mix = mix_fn(x_batch, t_batch, alpha=mix_alpha)

            grads = model.gradient(x_batch, t_mix)
            optimizer.update(model.params, grads)
            epoch_loss += model.last_layer.loss

        epoch_loss /= iter_per_epoch
        loss_hist.append(epoch_loss)

        eval_n = min(500, len(x_train))
        tr_idx = np.random.choice(len(x_train), eval_n, replace=False)
        te_idx = np.random.choice(len(x_test), eval_n, replace=False)
        train_acc = model.accuracy(x_train[tr_idx], t_train[tr_idx])
        test_acc = model.accuracy(x_test[te_idx], t_test[te_idx])
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

        elapsed = time.time() - t0
        print(f"epoch {epoch+1:3d}/{epochs}: loss={epoch_loss:.4f}  "
              f"train={train_acc:.4f}  test={test_acc:.4f}  ({elapsed:.0f}s)")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model.save(os.path.join(output_dir, f"{model_name}_best.pkl"))

    _plot_history(train_acc_hist, test_acc_hist, loss_hist, model_name, output_dir)
    print(f"\nBest test acc: {best_test_acc:.4f}")
    return train_acc_hist, test_acc_hist, loss_hist


def _plot_history(train_acc, test_acc, loss, name, output_dir):
    epochs = range(1, len(train_acc) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_acc, label="train")
    ax1.plot(epochs, test_acc, linestyle="--", label="test")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.set_title(f"{name} - Accuracy")
    ax1.legend()
    ax2.plot(epochs, loss, color="red")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_title(f"{name} - Loss")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{name}_history.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", choices=["mixup", "cutmix"], default="mixup")
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay_at", type=int, nargs=2, default=[20, 25])
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    mix_fn = mixup if args.mix == "mixup" else cutmix
    OUTPUT_DIR = args.output_dir if args.output_dir else f"results_cifar10_vgg_bn_{args.mix}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading CIFAR-10 ...")
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    x_train, x_test, mean, std = normalize(x_train, x_test)
    print(f"train: {x_train.shape}  test: {x_test.shape}")

    model = VGGWithBN(3, 32, 10)
    name = f"VGGWithBN_cifar10_{args.mix}"

    train_with_mixup(
        model, name, x_train, t_train, x_test, t_test,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        lr_decay_at=tuple(args.decay_at), lr_decay_factor=0.1,
        mix_fn=mix_fn, mix_alpha=args.alpha,
        output_dir=OUTPUT_DIR,
    )

    n = (len(x_test) // args.batch) * args.batch
    all_probs = np.vstack([
        model.predict(x_test[i:i + args.batch], train_flg=False)
        for i in range(0, n, args.batch)
    ])
    y_pred = np.argmax(all_probs, axis=1)
    y_true = t_test[:n]

    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES,
        os.path.join(OUTPUT_DIR, f"{name}_confusion.png"), title=name)
    plot_sample_predictions(x_test[:n], y_true, y_pred, CLASS_NAMES,
        os.path.join(OUTPUT_DIR, f"{name}_samples.png"),
        denorm_mean=mean, denorm_std=std)
    top1, top5 = plot_top5_accuracy(model, x_test, t_test, CLASS_NAMES,
        os.path.join(OUTPUT_DIR, f"{name}_per_class.png"))
    print(f"\nFinal  Top-1: {top1:.4f}  Top-5: {top5:.4f}")
    model.save(os.path.join(OUTPUT_DIR, f"{name}.pkl"))
