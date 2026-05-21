import sys
sys.path.append("../..")
import numpy as np
from collections import OrderedDict
from common.conv_layers import Convolution, Pooling
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.layers_ext import Dropout


class VGGLike:
    """
    Block1: Conv(64)-ReLU-Conv(64)-ReLU-Pool
    Block2: Conv(128)-ReLU-Conv(128)-ReLU-Pool
    Block3: Conv(256)-ReLU-Conv(256)-ReLU-Pool
    FC: Affine(512)-ReLU-Dropout-Affine(n_class)
    Input: (N, 3, 64, 64) → 8x8x256 → flatten 16384
    """

    def __init__(self, input_channels=3, input_size=64, output_size=37):
        self.output_size = output_size
        flat_size = 256 * (input_size // 8) * (input_size // 8)

        scale = lambda fan_in: np.sqrt(2.0 / fan_in)

        self.params = {
            "W1": scale(input_channels * 9) * np.random.randn(64, input_channels, 3, 3),
            "b1": np.zeros(64),
            "W2": scale(64 * 9) * np.random.randn(64, 64, 3, 3),
            "b2": np.zeros(64),
            "W3": scale(64 * 9) * np.random.randn(128, 64, 3, 3),
            "b3": np.zeros(128),
            "W4": scale(128 * 9) * np.random.randn(128, 128, 3, 3),
            "b4": np.zeros(128),
            "W5": scale(128 * 9) * np.random.randn(256, 128, 3, 3),
            "b5": np.zeros(256),
            "W6": scale(256 * 9) * np.random.randn(256, 256, 3, 3),
            "b6": np.zeros(256),
            "W7": scale(flat_size) * np.random.randn(flat_size, 512),
            "b7": np.zeros(512),
            "W8": scale(512) * np.random.randn(512, output_size),
            "b8": np.zeros(output_size),
        }
        p = self.params
        self.layers = OrderedDict([
            ("Conv1",   Convolution(p["W1"], p["b1"], stride=1, pad=1)),
            ("Relu1",   Relu()),
            ("Conv2",   Convolution(p["W2"], p["b2"], stride=1, pad=1)),
            ("Relu2",   Relu()),
            ("Pool1",   Pooling(2, 2, stride=2)),
            ("Conv3",   Convolution(p["W3"], p["b3"], stride=1, pad=1)),
            ("Relu3",   Relu()),
            ("Conv4",   Convolution(p["W4"], p["b4"], stride=1, pad=1)),
            ("Relu4",   Relu()),
            ("Pool2",   Pooling(2, 2, stride=2)),
            ("Conv5",   Convolution(p["W5"], p["b5"], stride=1, pad=1)),
            ("Relu5",   Relu()),
            ("Conv6",   Convolution(p["W6"], p["b6"], stride=1, pad=1)),
            ("Relu6",   Relu()),
            ("Pool3",   Pooling(2, 2, stride=2)),
            ("Affine1", Affine(p["W7"], p["b7"])),
            ("Relu7",   Relu()),
            ("Drop1",   Dropout(0.5)),
            ("Affine2", Affine(p["W8"], p["b8"])),
        ])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for name, layer in self.layers.items():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x, train_flg=True), t)

    def accuracy(self, x, t, batch_size=32):
        t_label = np.argmax(t, axis=1) if t.ndim != 1 else t
        acc = 0
        n = (x.shape[0] // batch_size) * batch_size
        for i in range(0, n, batch_size):
            y = self.predict(x[i:i+batch_size], train_flg=False)
            acc += np.sum(np.argmax(y, axis=1) == t_label[i:i+batch_size])
        return acc / n if n > 0 else 0.0

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        conv_names = ["Conv1","Conv2","Conv3","Conv4","Conv5","Conv6","Affine1","Affine2"]
        grads = {}
        for i, name in enumerate(conv_names):
            grads[f"W{i+1}"] = self.layers[name].dW
            grads[f"b{i+1}"] = self.layers[name].db
        return grads

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            self.params = pickle.load(f)
        conv_names = ["Conv1","Conv2","Conv3","Conv4","Conv5","Conv6","Affine1","Affine2"]
        for i, name in enumerate(conv_names):
            self.layers[name].W = self.params[f"W{i+1}"]
            self.layers[name].b = self.params[f"b{i+1}"]
