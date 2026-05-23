import sys, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS = [
    ("C01-vgglike-baseline",  "VGGLike\nmild aug",    "mild"),
    ("C02-vggbn-batchnorm",   "VGGWithBN\nmild aug",  "mild"),
    ("C03-resnet-skipconn",   "ResNet\nmild aug\n(17ep)", "mild"),
    ("C04-vggbn-noaug",       "VGGWithBN\nno aug",    "none"),
]

BASE = os.path.dirname(os.path.abspath(__file__))


def parse_log(path):
    top1, top5, best = None, None, None
    if not os.path.exists(path):
        return top1, top5, best
    train_hist, test_hist = [], []
    with open(path) as f:
        for line in f:
            m = re.search(r"Final\s+Top-1:\s+([\d.]+)[,\s]+Top-5:\s+([\d.]+)", line)
            if m:
                top1, top5 = float(m.group(1)), float(m.group(2))
            m = re.search(r"Best test acc:\s+([\d.]+)", line)
            if m:
                best = float(m.group(1))
            m = re.search(r"epoch\s+\d+/\d+:.*train=([\d.]+)\s+test=([\d.]+)", line)
            if m:
                train_hist.append(float(m.group(1)))
                test_hist.append(float(m.group(2)))
    return top1, top5, best, train_hist, test_hist


results = []
for folder, label, aug in EXPERIMENTS:
    log = os.path.join(BASE, folder, "train.log")
    parsed = parse_log(log)
    if len(parsed) == 5:
        top1, top5, best, tr, te = parsed
    else:
        top1, top5, best, tr, te = None, None, None, [], []
    results.append({"folder": folder, "label": label, "aug": aug,
                    "top1": top1, "top5": top5, "best": best,
                    "train_hist": tr, "test_hist": te})

done = [r for r in results if r["top1"] is not None]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("CIFAR-10 Classification Experiments (pure NumPy CNN)", fontsize=13, fontweight='bold')

if done:
    labels = [r["label"] for r in done]
    top1_vals = [r["top1"] * 100 for r in done]
    top5_vals = [r["top5"] * 100 for r in done]
    x = np.arange(len(done))
    w = 0.35
    ax = axes[0]
    b1 = ax.bar(x - w/2, top1_vals, w, label="Top-1", color="steelblue")
    ax.bar(x + w/2, top5_vals, w, label="Top-5", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Top-1 / Top-5 Accuracy")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5, label='90%')
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')

colors = ['steelblue', 'coral', 'green', 'purple']
for i, r in enumerate(results):
    if r["test_hist"]:
        ep = range(1, len(r["test_hist"]) + 1)
        axes[1].plot(ep, [v*100 for v in r["test_hist"]],
                     label=r["label"].replace('\n', ' '), color=colors[i % len(colors)])
        axes[2].plot(ep, [v*100 for v in r["train_hist"]],
                     label=r["label"].replace('\n', ' '), color=colors[i % len(colors)])

axes[1].set_title("Learning Curve (test acc)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test Accuracy (%)")
axes[1].legend(fontsize=7)
axes[1].set_ylim(0, 100)

axes[2].set_title("Learning Curve (train acc)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Train Accuracy (%)")
axes[2].legend(fontsize=7)
axes[2].set_ylim(0, 100)

plt.tight_layout()
out = os.path.join(BASE, "comparison_cifar10.png")
plt.savefig(out, dpi=120)
plt.close()
print(f"Saved: {out}")

print("\n=== CIFAR-10 Results ===")
print(f"{'Experiment':<25} {'Top-1':>8} {'Top-5':>8}")
print("-" * 45)
for r in results:
    t1 = f"{r['top1']*100:.2f}%" if r['top1'] else "running..."
    t5 = f"{r['top5']*100:.2f}%" if r['top5'] else "—"
    print(f"{r['folder']:<25} {t1:>8} {t5:>8}")
