import sys
sys.path.append("..")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from ch05.two_layer_net_backprop import TwoLayerNet
from common.optimizer import SGD, Momentum, AdaGrad, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

optimizers = {
    "SGD": SGD(lr=0.01),
    "Momentum": Momentum(lr=0.01),
    "AdaGrad": AdaGrad(lr=0.01),
    "Adam": Adam(lr=0.001),
}

train_loss = {name: [] for name in optimizers}

iters = 2000
batch_size = 100
train_size = x_train.shape[0]

for name, optimizer in optimizers.items():
    network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    for i in range(iters):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)
        train_loss[name].append(network.loss(x_batch, t_batch))
    print(f"{name}: final_loss={train_loss[name][-1]:.4f}")

for name, losses in train_loss.items():
    plt.plot(losses, label=name)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 3)
plt.legend()
plt.title("Optimizer Comparison")
plt.savefig("optimizer_compare.png")
plt.close()
print("saved optimizer_compare.png")
