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


def mixup(x_batch, t_batch, alpha=0.4):
    n, num_classes = x_batch.shape[0], t_batch.max() + 1 if t_batch.ndim == 1 else t_batch.shape[1]
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(n)

    if t_batch.ndim == 1:
        t_onehot = np.eye(num_classes, dtype=np.float32)[t_batch]
    else:
        t_onehot = t_batch.astype(np.float32)

    x_mix = lam * x_batch + (1.0 - lam) * x_batch[idx]
    t_mix = lam * t_onehot + (1.0 - lam) * t_onehot[idx]
    return x_mix, t_mix


def cutmix(x_batch, t_batch, alpha=1.0):
    n = x_batch.shape[0]
    _, C, H, W = x_batch.shape
    num_classes = t_batch.max() + 1 if t_batch.ndim == 1 else t_batch.shape[1]

    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(n)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x_mix = x_batch.copy()
    x_mix[:, :, y1:y2, x1:x2] = x_batch[idx, :, y1:y2, x1:x2]

    lam_actual = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)

    if t_batch.ndim == 1:
        t_onehot = np.eye(num_classes, dtype=np.float32)[t_batch]
    else:
        t_onehot = t_batch.astype(np.float32)

    t_mix = lam_actual * t_onehot + (1.0 - lam_actual) * t_onehot[idx]
    return x_mix, t_mix


def batch_mixup_augment(x_batch, t_batch, train_flg=True):
    x_aug = batch_mild_augment(x_batch, train_flg=train_flg)
    return mixup(x_aug, t_batch)


def batch_cutmix_augment(x_batch, t_batch, train_flg=True):
    x_aug = batch_mild_augment(x_batch, train_flg=train_flg)
    return cutmix(x_aug, t_batch)
