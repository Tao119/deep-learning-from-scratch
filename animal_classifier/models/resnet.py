import sys
sys.path.append("../..")
import numpy as np
from collections import OrderedDict
from common.conv_layers import Convolution, Pooling
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.layers_ext import BatchNormalization, Dropout
from common.im2col import im2col, col2im


class ResidualBlock:
    """
    Conv(3x3)-BN-ReLU-Conv(3x3)-BN + skip → ReLU
    チャネル数が変わる場合は1x1 Conv でショートカット射影。
    """

    def __init__(self, in_ch, out_ch, stride=1):
        scale = np.sqrt(2.0 / (in_ch * 9))
        self.conv1 = Convolution(
            scale * np.random.randn(out_ch, in_ch, 3, 3),
            np.zeros(out_ch), stride=stride, pad=1
        )
        self.bn1 = BatchNormalization(np.ones(out_ch), np.zeros(out_ch))
        self.relu1 = Relu()
        self.conv2 = Convolution(
            np.sqrt(2.0 / (out_ch * 9)) * np.random.randn(out_ch, out_ch, 3, 3),
            np.zeros(out_ch), stride=1, pad=1
        )
        self.bn2 = BatchNormalization(np.ones(out_ch), np.zeros(out_ch))
        self.relu2 = Relu()

        self.use_projection = (in_ch != out_ch or stride != 1)
        if self.use_projection:
            self.proj = Convolution(
                np.sqrt(2.0 / in_ch) * np.random.randn(out_ch, in_ch, 1, 1),
                np.zeros(out_ch), stride=stride, pad=0
            )
            self.proj_bn = BatchNormalization(np.ones(out_ch), np.zeros(out_ch))

        self._x = None
        self._residual = None

    def forward(self, x, train_flg=False):
        self._x = x
        residual = x
        if self.use_projection:
            residual = self.proj_bn.forward(self.proj.forward(x), train_flg)
        self._residual = residual

        out = self.conv1.forward(x)
        out = self.bn1.forward(out, train_flg)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, train_flg)
        out += residual
        return self.relu2.forward(out)

    def backward(self, dout):
        dout = self.relu2.backward(dout)
        dresidual = dout.copy()
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        if self.use_projection:
            dresidual = self.proj_bn.backward(dresidual)
            dresidual = self.proj.backward(dresidual)
        return dout + dresidual

    def params_and_grads(self):
        layers = [self.conv1, self.bn1, self.conv2, self.bn2]
        if self.use_projection:
            layers += [self.proj, self.proj_bn]
        return layers


class ResNet:
    """
    stem: Conv(64, 3x3, pad=1)-BN-ReLU
    stage1: 2x ResidualBlock(64→64)
    stage2: 2x ResidualBlock(64→128, stride=2)
    stage3: 2x ResidualBlock(128→256, stride=2)
    GlobalAvgPool → Affine(37)
    Input: (N, 3, 64, 64)
    """

    def __init__(self, input_channels=3, output_size=37):
        self.stem_conv = Convolution(
            np.sqrt(2.0 / (input_channels * 9)) * np.random.randn(64, input_channels, 3, 3),
            np.zeros(64), stride=1, pad=1
        )
        self.stem_bn = BatchNormalization(np.ones(64), np.zeros(64))
        self.stem_relu = Relu()

        self.blocks = [
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        ]

        self.fc = Affine(
            np.sqrt(2.0 / 256) * np.random.randn(256, output_size),
            np.zeros(output_size)
        )
        self.last_layer = SoftmaxWithLoss()
        self._train_flg = False

    def predict(self, x, train_flg=False):
        self._train_flg = train_flg
        x = self.stem_conv.forward(x)
        x = self.stem_bn.forward(x, train_flg)
        x = self.stem_relu.forward(x)
        for block in self.blocks:
            x = block.forward(x, train_flg)
        x = x.mean(axis=(2, 3))
        return self.fc.forward(x)

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
        dout = self.fc.backward(dout)
        dout = dout[:, :, np.newaxis, np.newaxis] * np.ones((1, 1, *self.blocks[-1]._x.shape[2:]))
        dout = dout / (self.blocks[-1]._x.shape[2] * self.blocks[-1]._x.shape[3])

        for block in reversed(self.blocks):
            dout = block.backward(dout)
        dout = self.stem_relu.backward(dout)
        dout = self.stem_bn.backward(dout)
        self.stem_conv.backward(dout)
        return {}

    def get_params_and_grads(self):
        params = {}
        grads = {}
        idx = 0

        def register(layer, key_w="W", key_b="b"):
            nonlocal idx
            k_w = f"p{idx}_{key_w}"
            k_b = f"p{idx}_{key_b}"
            params[k_w] = layer.W if hasattr(layer, "W") else layer.gamma
            params[k_b] = layer.b if hasattr(layer, "b") else layer.beta
            grads[k_w] = layer.dW if hasattr(layer, "dW") else layer.dgamma
            grads[k_b] = layer.db if hasattr(layer, "db") else layer.dbeta

        for layer in [self.stem_conv, self.stem_bn]:
            register(layer)
            idx += 1
        for block in self.blocks:
            for layer in block.params_and_grads():
                register(layer)
                idx += 1
        register(self.fc)
        return params, grads

    def update(self, optimizer):
        params, grads = self.get_params_and_grads()
        optimizer.update(params, grads)

    def save(self, path):
        import pickle
        params, _ = self.get_params_and_grads()
        with open(path, "wb") as f:
            pickle.dump(params, f)
