import sys
sys.path.append("../..")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, class_names, output_path, title="Confusion Matrix"):
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_sample_predictions(x, y_true, y_pred, class_names, output_path,
                            n_rows=4, n_cols=6, denorm_mean=None, denorm_std=None):
    wrong_idx = np.where(y_true != y_pred)[0]
    right_idx = np.where(y_true == y_pred)[0]
    n = n_rows * n_cols
    indices = np.concatenate([
        right_idx[:n // 2] if len(right_idx) >= n // 2 else right_idx,
        wrong_idx[:n // 2] if len(wrong_idx) >= n // 2 else wrong_idx,
    ])[:n]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.5))
    for i, ax in enumerate(axes.flat):
        if i >= len(indices):
            ax.axis("off")
            continue
        idx = indices[i]
        img = x[idx].transpose(1, 2, 0)
        if denorm_mean is not None:
            img = img * denorm_std.squeeze() + denorm_mean.squeeze()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        correct = y_true[idx] == y_pred[idx]
        color = "green" if correct else "red"
        ax.set_title(
            f"T:{class_names[y_true[idx]][:8]}\nP:{class_names[y_pred[idx]][:8]}",
            color=color, fontsize=7
        )
        ax.axis("off")
    plt.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved {output_path}")


def plot_top5_accuracy(model, x_test, t_test, class_names, output_path, batch_size=32):
    all_probs = []
    n = (x_test.shape[0] // batch_size) * batch_size
    for i in range(0, n, batch_size):
        y = model.predict(x_test[i:i+batch_size], train_flg=False)
        all_probs.append(y)
    all_probs = np.vstack(all_probs)
    t = t_test[:n]

    top1 = np.mean(np.argmax(all_probs, axis=1) == t)
    top5_correct = 0
    for i, prob in enumerate(all_probs):
        top5 = np.argsort(prob)[::-1][:5]
        if t[i] in top5:
            top5_correct += 1
    top5 = top5_correct / n

    per_class_acc = []
    for c in range(len(class_names)):
        mask = t == c
        if mask.sum() == 0:
            per_class_acc.append(0)
            continue
        pred = np.argmax(all_probs[mask], axis=1)
        per_class_acc.append(np.mean(pred == c))

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(class_names)), per_class_acc, color="steelblue")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-class Accuracy  (Top-1: {top1:.3f}, Top-5: {top5:.3f})")
    ax.axhline(top1, color="red", linestyle="--", label=f"Top-1 avg={top1:.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved {output_path}")
    return top1, top5


def plot_model_comparison(results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, (train_acc, test_acc, loss) in results.items():
        epochs = range(1, len(train_acc) + 1)
        axes[0].plot(epochs, test_acc, label=name)
        axes[1].plot(epochs, loss, label=name)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Model Comparison - Test Accuracy")
    axes[0].legend()
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Model Comparison - Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved {output_path}")
