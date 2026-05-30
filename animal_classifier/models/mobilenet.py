import sys
sys.path.append("../..")
import numpy as np
from common.conv_layers import Convolution
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.layers_ext import BatchNormalization
from common.im2col import im2col, col2im


class DepthwiseConvolution:
    """Depthwise conv: one filter per input channel"""

    def __init__(self, channels, kernel_size=3, stride=1, pad=1):
        k = kernel_size
        self.W = np.sqrt(2.0/(channels*k*k)) * np.random.randn(channels, 1, k, k)
        self.b = np.zeros(channels)
        self.stride = stride
        self.pad = pad
        self.dW = None
        self.db = None

    def forward(self, x):
        N, C, H, W = x.shape
        _, _, FH, FW = self.W.shape
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2*self.pad - FW) // self.stride + 1
        out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
        self._cols = []
        for c in range(C):
            x_c = x[:, c:c+1, :, :]
            col = im2col(x_c, FH, FW, self.stride, self.pad)
            self._cols.append(col)
            w_c = self.W[c].reshape(1, -1).T
            out_c = (col @ w_c + self.b[c]).reshape(N, out_h, out_w)
            out[:, c, :, :] = out_c
        self._x = x
        self._out_h = out_h
        self._out_w = out_w
        return out

    def backward(self, dout):
        N, C, H, W = self._x.shape
        _, _, FH, FW = self.W.shape
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros(C)
        dx = np.zeros_like(self._x)
        for c in range(C):
            dout_c = dout[:, c, :, :].reshape(-1, 1)
            col = self._cols[c]
            self.db[c] = dout_c.sum()
            self.dW[c] = (col.T @ dout_c).reshape(1, FH, FW)
            w_c = self.W[c].reshape(1, -1).T
            dcol = dout_c @ w_c.T
            dx[:, c:c+1, :, :] = col2im(dcol, self._x[:, c:c+1].shape, FH, FW, self.stride, self.pad)
        return dx


class DSConvBlock:
    """Depthwise Separable Conv block: DW-BN-ReLU-PW-BN-ReLU"""

    def __init__(self, in_ch, out_ch, stride=1):
        self.dw = DepthwiseConvolution(in_ch, stride=stride)
        self.dw_bn = BatchNormalization(np.ones(in_ch), np.zeros(in_ch))
        self.dw_relu = Relu()
        self.pw = Convolution(
            np.sqrt(2.0/in_ch)*np.random.randn(out_ch, in_ch, 1, 1),
            np.zeros(out_ch), stride=1, pad=0)
        self.pw_bn = BatchNormalization(np.ones(out_ch), np.zeros(out_ch))
        self.pw_relu = Relu()

    def forward(self, x, train_flg=False):
        out = self.dw.forward(x)
        out = self.dw_bn.forward(out, train_flg)
        out = self.dw_relu.forward(out)
        out = self.pw.forward(out)
        out = self.pw_bn.forward(out, train_flg)
        return self.pw_relu.forward(out)

    def backward(self, dout):
        dout = self.pw_relu.backward(dout)
        dout = self.pw_bn.backward(dout)
        dout = self.pw.backward(dout)
        dout = self.dw_relu.backward(dout)
        dout = self.dw_bn.backward(dout)
        return self.dw.backward(dout)

    def params_and_grads(self):
        return [self.dw, self.dw_bn, self.pw, self.pw_bn]


class MobileNet:
    """
    Simplified MobileNet for CIFAR-10 (32×32):
      Conv(32) → DS(32→64) → DS(64→128,s2) → DS(128→128) → DS(128→256,s2)
      → DS(256→256) × 2 → GlobalAvgPool → FC(10)
    """

    def __init__(self, input_channels=3, output_size=10):
        self.stem = Convolution(
            np.sqrt(2.0/(input_channels*9))*np.random.randn(32, input_channels, 3, 3),
            np.zeros(32), stride=1, pad=1)
        self.stem_bn = BatchNormalization(np.ones(32), np.zeros(32))
        self.stem_relu = Relu()

        self.blocks = [
            DSConvBlock(32,  64,  stride=1),
            DSConvBlock(64,  128, stride=2),
            DSConvBlock(128, 128, stride=1),
            DSConvBlock(128, 256, stride=2),
            DSConvBlock(256, 256, stride=1),
            DSConvBlock(256, 256, stride=1),
        ]

        self.fc = Affine(
            np.sqrt(2.0/256)*np.random.randn(256, output_size),
            np.zeros(output_size))
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        x = self.stem_relu.forward(self.stem_bn.forward(self.stem.forward(x), train_flg))
        for block in self.blocks:
            x = block.forward(x, train_flg)
        return self.fc.forward(x.mean(axis=(2, 3)))

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x, train_flg=True), t)

    def accuracy(self, x, t, batch_size=32):
        t_label = np.argmax(t, axis=1) if t.ndim != 1 else t
        acc, n = 0, (x.shape[0]//batch_size)*batch_size
        for i in range(0, n, batch_size):
            y = self.predict(x[i:i+batch_size], train_flg=False)
            acc += np.sum(np.argmax(y, axis=1) == t_label[i:i+batch_size])
        return acc/n if n > 0 else 0.0

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        dout = self.fc.backward(dout)
        H, W = self.blocks[-1].pw.x.shape[2:]
        dout = dout[:, :, None, None] * np.ones((1, 1, H, W)) / (H * W)
        for block in reversed(self.blocks):
            dout = block.backward(dout)
        dout = self.stem_relu.backward(dout)
        dout = self.stem_bn.backward(dout)
        self.stem.backward(dout)
        return {}

    def get_params_and_grads(self):
        params, grads = {}, {}
        idx = 0

        def reg_conv(layer):
            nonlocal idx
            kw, kb = f"p{idx}_W", f"p{idx}_b"
            params[kw] = layer.W
            params[kb] = layer.b
            grads[kw] = layer.dW
            grads[kb] = layer.db
            idx += 1

        def reg_bn(layer):
            nonlocal idx
            kw, kb = f"p{idx}_g", f"p{idx}_bt"
            params[kw] = layer.gamma
            params[kb] = layer.beta
            grads[kw] = layer.dgamma
            grads[kb] = layer.dbeta
            idx += 1

        reg_conv(self.stem)
        reg_bn(self.stem_bn)
        for block in self.blocks:
            reg_conv(block.dw)
            reg_bn(block.dw_bn)
            reg_conv(block.pw)
            reg_bn(block.pw_bn)
        reg_conv(self.fc)
        return params, grads

    def update(self, optimizer):
        params, grads = self.get_params_and_grads()
        optimizer.update(params, grads)

    def save(self, path):
        import pickle
        params, _ = self.get_params_and_grads()
        try:
            with open(path, "wb") as f:
                pickle.dump(params, f)
        except OSError as e:
            print(f"  [warn] save failed: {e}")
