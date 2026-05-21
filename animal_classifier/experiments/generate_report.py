import sys
sys.path.append("../..")
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

EXPERIMENTS = [
    ("01-vgglike-baseline",    "VGGLike\n37cls 32px\nnone→full", 37, 32, "full"),
    ("02-vggbn-batchnorm",     "VGGWithBN\n37cls 32px\nfull",    37, 32, "full"),
    ("03-resnet-skipconn",     "ResNet\n37cls 32px\nfull",       37, 32, "full"),
    ("04-vggnoaug-overfit",    "VGGLike\n37cls 32px\nnone",      37, 32, "none"),
    ("05-vggbn-catdog-binary", "VGGWithBN\n2cls 64px\nmild",      2, 64, "mild"),
    ("06-vggbn-top10-64px",    "VGGWithBN\n10cls 64px\nmild",   10, 64, "mild"),
    ("07-resnet-top10-64px",   "ResNet\n10cls 64px\nmild",      10, 64, "mild"),
    ("08-vggbn-top10-fullaug", "VGGWithBN\n10cls 64px\nfull",   10, 64, "full"),
]


def parse_log(log_path):
    if not os.path.exists(log_path):
        return None, None, None
    top1, top5, best = None, None, None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Final\s+Top-1:\s+([\d.]+)[,\s]+Top-5:\s+([\d.]+)", line)
            if m:
                top1, top5 = float(m.group(1)), float(m.group(2))
            m = re.search(r"Best test acc:\s+([\d.]+)", line)
            if m:
                best = float(m.group(1))
    return top1, top5, best


def parse_history(log_path):
    if not os.path.exists(log_path):
        return [], []
    train_acc, test_acc = [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"epoch\s+\d+/\d+:.*train=([\d.]+)\s+test=([\d.]+)", line)
            if m:
                train_acc.append(float(m.group(1)))
                test_acc.append(float(m.group(2)))
    return train_acc, test_acc


BASE = os.path.dirname(os.path.abspath(__file__))

results = []
for folder, label, n_cls, size, aug in EXPERIMENTS:
    log = os.path.join(BASE, folder, "train.log")
    top1, top5, best = parse_log(log)
    train_hist, test_hist = parse_history(log)
    results.append({
        "folder": folder, "label": label,
        "n_cls": n_cls, "size": size, "aug": aug,
        "top1": top1, "top5": top5, "best": best,
        "train_hist": train_hist, "test_hist": test_hist,
    })

done = [r for r in results if r["top1"] is not None]
if not done:
    print("完了した実験が見つかりません")
    sys.exit(0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Oxford-IIIT Pet Classification Experiments", fontsize=14, fontweight='bold')

labels = [r["label"] for r in done]
top1_vals = [r["top1"] * 100 for r in done]
top5_vals = [r["top5"] * 100 for r in done]

x = np.arange(len(done))
w = 0.35
ax = axes[0]
bars1 = ax.bar(x - w/2, top1_vals, w, label="Top-1", color="steelblue")
bars2 = ax.bar(x + w/2, top5_vals, w, label="Top-5", color="coral")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Top-1 / Top-5 Accuracy (full test set)")
ax.legend()
ax.set_ylim(0, 100)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=7)

ax2 = axes[1]
for r in results:
    if r["test_hist"]:
        epochs = range(1, len(r["test_hist"]) + 1)
        ax2.plot(epochs, [v*100 for v in r["test_hist"]], label=r["label"].replace('\n', ' '))
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Test Accuracy (%)")
ax2.set_title("Learning Curve (test)")
ax2.legend(fontsize=6)

ax3 = axes[2]
for r in results:
    if r["train_hist"]:
        epochs = range(1, len(r["train_hist"]) + 1)
        ax3.plot(epochs, [v*100 for v in r["train_hist"]], label=r["label"].replace('\n', ' '))
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Train Accuracy (%)")
ax3.set_title("Learning Curve (train)")
ax3.legend(fontsize=6)

plt.tight_layout()
out = os.path.join(BASE, "comparison_all.png")
plt.savefig(out, dpi=120)
plt.close()
print(f"Saved: {out}")

print("\n=== Experiment Summary ===")
print(f"{'Experiment':<30} {'Top-1':>8} {'Top-5':>8} {'BestSamp':>10}")
print("-" * 60)
for r in results:
    t1 = f"{r['top1']*100:.2f}%" if r['top1'] else "—"
    t5 = f"{r['top5']*100:.2f}%" if r['top5'] else "—"
    bs = f"{r['best']*100:.2f}%" if r['best'] else "—"
    print(f"{r['folder']:<30} {t1:>8} {t5:>8} {bs:>10}")
