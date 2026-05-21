import numpy as np


def random_flip_horizontal(x):
    if np.random.rand() > 0.5:
        return x[:, :, ::-1].copy()
    return x


def random_crop(x, pad=4):
    C, H, W = x.shape
    padded = np.pad(x, [(0, 0), (pad, pad), (pad, pad)], mode="reflect")
    top = np.random.randint(0, 2 * pad)
    left = np.random.randint(0, 2 * pad)
    return padded[:, top:top + H, left:left + W]


def color_jitter(x, brightness=0.2, contrast=0.2):
    b = 1.0 + np.random.uniform(-brightness, brightness)
    c = 1.0 + np.random.uniform(-contrast, contrast)
    x = x * b
    mean = x.mean(axis=(1, 2), keepdims=True)
    x = (x - mean) * c + mean
    return np.clip(x, 0, 1)


def cutout(x, size=16):
    C, H, W = x.shape
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(0, cx - size // 2)
    x2 = min(W, cx + size // 2)
    y1 = max(0, cy - size // 2)
    y2 = min(H, cy + size // 2)
    x = x.copy()
    x[:, y1:y2, x1:x2] = 0
    return x


def augment(x, train_flg=True, use_cutout=False):
    if not train_flg:
        return x
    x = random_flip_horizontal(x)
    x = random_crop(x, pad=4)
    x = color_jitter(x, brightness=0.2, contrast=0.2)
    if use_cutout:
        x = cutout(x, size=16)
    return x


def batch_augment(x_batch, train_flg=True, use_cutout=False):
    return np.array([augment(x, train_flg, use_cutout) for x in x_batch])


def mild_augment(x, train_flg=True):
    if not train_flg:
        return x
    x = random_flip_horizontal(x)
    x = random_crop(x, pad=2)
    return x


def batch_mild_augment(x_batch, train_flg=True):
    return np.array([mild_augment(x, train_flg) for x in x_batch])
