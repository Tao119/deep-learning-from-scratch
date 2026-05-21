import sys
sys.path.append("..")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ch07.simple_conv_net import SimpleConvNet
from dataset.mnist import load_mnist
from common.optimizer import Adam


def normalize_for_vis(filters):
    f_min, f_max = filters.min(), filters.max()
    return (filters - f_min) / (f_max - f_min + 1e-8)


def plot_filters(filters, title, filename):
    n = filters.shape[0]
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(filters[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"saved {filename}")


network = SimpleConvNet(
    input_dim=(1, 28, 28),
    conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
    hidden_size=100,
    output_size=10,
)

filters_before = normalize_for_vis(network.params["W1"].copy())
plot_filters(filters_before, "Filters before training", "filters_before.png")

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
x_train = x_train[:3000]
t_train = t_train[:3000]

optimizer = Adam(lr=0.001)
batch_size = 100
for epoch in range(10):
    for i in range(x_train.shape[0] // batch_size):
        x_batch = x_train[i * batch_size : (i + 1) * batch_size]
        t_batch = t_train[i * batch_size : (i + 1) * batch_size]
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)
    print(f"epoch {epoch + 1}/10 done")

filters_after = normalize_for_vis(network.params["W1"].copy())
plot_filters(filters_after, "Filters after training", "filters_after.png")
