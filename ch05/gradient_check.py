import sys
sys.path.append("..")
import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_net_backprop import TwoLayerNet

(x_train, t_train), _ = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in ("W1", "b1", "W2", "b2"):
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(f"{key}: {diff:.8f}")
