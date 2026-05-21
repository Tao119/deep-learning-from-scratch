import sys
sys.path.append("..")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from ch07.simple_conv_net import SimpleConvNet
from common.optimizer import Adam

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

x_train = x_train[:5000]
t_train = t_train[:5000]
x_test = x_test[:1000]
t_test = t_test[:1000]

network = SimpleConvNet(
    input_dim=(1, 28, 28),
    conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
    hidden_size=100,
    output_size=10,
    weight_init_std=0.01,
)

optimizer = Adam(lr=0.001)

epochs = 20
batch_size = 100
train_size = x_train.shape[0]
iter_per_epoch = max(train_size // batch_size, 1)

train_acc_list = []
test_acc_list = []

for epoch in range(epochs):
    for _ in range(iter_per_epoch):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)

    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f"epoch {epoch + 1:2d}: train={train_acc:.4f}, test={test_acc:.4f}")

plt.plot(range(1, epochs + 1), train_acc_list, label="train")
plt.plot(range(1, epochs + 1), test_acc_list, linestyle="--", label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("conv_accuracy.png")
plt.close()
print("saved conv_accuracy.png")
