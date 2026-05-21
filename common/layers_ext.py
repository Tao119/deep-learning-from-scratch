import numpy as np


class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None
        self.running_mean = running_mean
        self.running_var = running_var
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim == 4:
            N, C, H, W = x.shape
            x_2d = x.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            x_2d = x

        if self.running_mean is None:
            D = x_2d.shape[1]
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x_2d.mean(axis=0)
            xc = x_2d - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 1e-7)
            xn = xc / std
            self.batch_size = x_2d.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x_2d - self.running_mean
            xn = xc / np.sqrt(self.running_var + 1e-7)

        out = self.gamma * xn + self.beta
        if self.input_shape.__len__() == 4:
            N, C, H, W = self.input_shape
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        if dout.ndim == 4:
            N, C, H, W = dout.shape
            dout_2d = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            dout_2d = dout

        dxn = self.gamma * dout_2d
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std ** 2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        self.dgamma = np.sum(self.xn * dout_2d, axis=0)
        self.dbeta = np.sum(dout_2d, axis=0)

        if len(self.input_shape) == 4:
            N, C, H, W = self.input_shape
            dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return dx


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
