import sys
sys.path.append("..")
import numpy as np
from collections import OrderedDict
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.gradient import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size),
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * np.random.randn(hidden_size, output_size),
            "b2": np.zeros(output_size),
        }
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x), t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y_label = np.argmax(y, axis=1)
        t_label = np.argmax(t, axis=1) if t.ndim != 1 else t
        return np.sum(y_label == t_label) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_fn = lambda W: self.loss(x, t)
        return {
            "W1": numerical_gradient(loss_fn, self.params["W1"]),
            "b1": numerical_gradient(loss_fn, self.params["b1"]),
            "W2": numerical_gradient(loss_fn, self.params["W2"]),
            "b2": numerical_gradient(loss_fn, self.params["b2"]),
        }

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        self.params["W1"][:] = self.layers["Affine1"].W
        self.params["b1"][:] = self.layers["Affine1"].b
        self.params["W2"][:] = self.layers["Affine2"].W
        self.params["b2"][:] = self.layers["Affine2"].b

        return {
            "W1": self.layers["Affine1"].dW,
            "b1": self.layers["Affine1"].db,
            "W2": self.layers["Affine2"].dW,
            "b2": self.layers["Affine2"].db,
        }
