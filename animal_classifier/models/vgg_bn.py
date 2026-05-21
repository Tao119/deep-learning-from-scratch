import sys
sys.path.append("../..")
import numpy as np
from collections import OrderedDict
from common.conv_layers import Convolution, Pooling
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.layers_ext import BatchNormalization, Dropout


class VGGWithBN:
    """
    VGGLikeと同じ構成だが各Conv後にBatch Normalizationを追加。
    """

    def __init__(self, input_channels=3, input_size=64, output_size=37):
        flat_size = 256 * (input_size // 8) * (input_size // 8)
        scale = lambda fan_in: np.sqrt(2.0 / fan_in)

        channel_sizes = [
            (input_channels, 64), (64, 64),
            (64, 128), (128, 128),
            (128, 256), (256, 256),
        ]

        self.params = {}
        for i, (cin, cout) in enumerate(channel_sizes, 1):
            self.params[f"W{i}"] = scale(cin * 9) * np.random.randn(cout, cin, 3, 3)
            self.params[f"b{i}"] = np.zeros(cout)
            self.params[f"gamma{i}"] = np.ones(cout)
            self.params[f"beta{i}"] = np.zeros(cout)

        self.params["W7"] = scale(flat_size) * np.random.randn(flat_size, 512)
        self.params["b7"] = np.zeros(512)
        self.params["W8"] = scale(512) * np.random.randn(512, output_size)
        self.params["b8"] = np.zeros(output_size)

        p = self.params

        def make_bn(i):
            return BatchNormalization(p[f"gamma{i}"], p[f"beta{i}"])

        self.layers = OrderedDict([
            ("Conv1",   Convolution(p["W1"], p["b1"], stride=1, pad=1)),
            ("BN1",     make_bn(1)),
            ("Relu1",   Relu()),
            ("Conv2",   Convolution(p["W2"], p["b2"], stride=1, pad=1)),
            ("BN2",     make_bn(2)),
            ("Relu2",   Relu()),
            ("Pool1",   Pooling(2, 2, stride=2)),
            ("Conv3",   Convolution(p["W3"], p["b3"], stride=1, pad=1)),
            ("BN3",     make_bn(3)),
            ("Relu3",   Relu()),
            ("Conv4",   Convolution(p["W4"], p["b4"], stride=1, pad=1)),
            ("BN4",     make_bn(4)),
            ("Relu4",   Relu()),
            ("Pool2",   Pooling(2, 2, stride=2)),
            ("Conv5",   Convolution(p["W5"], p["b5"], stride=1, pad=1)),
            ("BN5",     make_bn(5)),
            ("Relu5",   Relu()),
            ("Conv6",   Convolution(p["W6"], p["b6"], stride=1, pad=1)),
            ("BN6",     make_bn(6)),
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
            if isinstance(layer, (BatchNormalization, Dropout)):
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

        grads = {}
        conv_names = ["Conv1","Conv2","Conv3","Conv4","Conv5","Conv6","Affine1","Affine2"]
        for i, name in enumerate(conv_names, 1):
            grads[f"W{i}"] = self.layers[name].dW
            grads[f"b{i}"] = self.layers[name].db
        for i, name in enumerate([f"BN{j}" for j in range(1,7)], 1):
            grads[f"gamma{i}"] = self.layers[name].dgamma
            grads[f"beta{i}"] = self.layers[name].dbeta
        return grads

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.params, f)
