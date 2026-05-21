import sys
sys.path.append("..")
import numpy as np
from collections import OrderedDict
from common.conv_layers import Convolution, Pooling
from common.layers import Affine, Relu, SoftmaxWithLoss


class SimpleConvNet:
    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01,
    ):
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) // filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) ** 2)

        self.params = {
            "W1": weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size),
            "b1": np.zeros(filter_num),
            "W2": weight_init_std * np.random.randn(pool_output_size, hidden_size),
            "b2": np.zeros(hidden_size),
            "W3": weight_init_std * np.random.randn(hidden_size, output_size),
            "b3": np.zeros(output_size),
        }

        p = self.params
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(p["W1"], p["b1"], filter_stride, filter_pad)
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(p["W2"], p["b2"])
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(p["W3"], p["b3"])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x), t)

    def accuracy(self, x, t, batch_size=100):
        t_label = np.argmax(t, axis=1) if t.ndim != 1 else t
        acc = 0.0
        for i in range(x.shape[0] // batch_size):
            tx = x[i * batch_size : (i + 1) * batch_size]
            tt = t_label[i * batch_size : (i + 1) * batch_size]
            y = self.predict(tx)
            acc += np.sum(np.argmax(y, axis=1) == tt)
        return acc / x.shape[0]

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grads = {
            "W1": self.layers["Conv1"].dW,
            "b1": self.layers["Conv1"].db,
            "W2": self.layers["Affine1"].dW,
            "b2": self.layers["Affine1"].db,
            "W3": self.layers["Affine2"].dW,
            "b3": self.layers["Affine2"].db,
        }
        return grads
