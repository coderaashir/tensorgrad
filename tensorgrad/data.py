"""Dataset download/parse helpers and a minimal DataLoader.

Files are cached under ``./data`` (override with the TENSORGRAD_DATA env var).
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

DATA_DIR = os.environ.get("TENSORGRAD_DATA", os.path.join(os.getcwd(), "data"))

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
# Per-pixel mean/std of the MNIST training set (after scaling to [0, 1]).
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def download(url, path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {url}")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def _read_idx(path):
    """Parse the IDX format MNIST ships in: magic, dims, then raw uint8s."""
    with gzip.open(path, "rb") as f:
        _zero, _dtype, ndim = struct.unpack(">HBB", f.read(4))
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(normalize=True, flatten=False):
    """Return (X_train, y_train, X_test, y_test).

    X is float32 of shape (N, 1, 28, 28), or (N, 784) if ``flatten``.
    y is int64 of shape (N,).
    """
    root = os.path.join(DATA_DIR, "mnist")
    arrays = {}
    for key, fname in MNIST_FILES.items():
        arrays[key] = _read_idx(download(MNIST_URL + fname, os.path.join(root, fname)))

    def prep(images):
        x = images.astype(np.float32) / 255.0
        if normalize:
            x = (x - MNIST_MEAN) / MNIST_STD
        return x.reshape(len(x), -1) if flatten else x[:, None, :, :]

    return (
        prep(arrays["train_images"]),
        arrays["train_labels"].astype(np.int64),
        prep(arrays["test_images"]),
        arrays["test_labels"].astype(np.int64),
    )


class DataLoader:
    """Iterate over (X, y) NumPy arrays in mini-batches.

    Yields plain NumPy arrays; wrap them in Tensors in the training loop.
    """

    def __init__(self, X, y, batch_size=64, shuffle=True, drop_last=False, seed=None):
        assert len(X) == len(y)
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n = len(self.X)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.X)
        idx = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            batch = idx[start : start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield self.X[batch], self.y[batch]


# ---------------------------------------------------------------- CIFAR-10
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def load_cifar10(normalize=True):
    """Return (X_train, y_train, X_test, y_test); X is float32 (N, 3, 32, 32).

    Uses the binary release (not the pickled one), where each record is
    1 label byte followed by 3072 pixel bytes (1024 R, 1024 G, 1024 B).
    """
    import tarfile

    root = os.path.join(DATA_DIR, "cifar10")
    archive = download(CIFAR10_URL, os.path.join(root, "cifar-10-binary.tar.gz"))

    def read(names):
        xs, ys = [], []
        with tarfile.open(archive, "r:gz") as tar:
            for name in names:
                raw = tar.extractfile(f"cifar-10-batches-bin/{name}").read()
                records = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3073)
                ys.append(records[:, 0].astype(np.int64))
                xs.append(records[:, 1:].reshape(-1, 3, 32, 32))
        x = np.concatenate(xs).astype(np.float32) / 255.0
        if normalize:
            x = (x - CIFAR10_MEAN[None, :, None, None]) / CIFAR10_STD[None, :, None, None]
        return x, np.concatenate(ys)

    X_train, y_train = read([f"data_batch_{i}.bin" for i in range(1, 6)])
    X_test, y_test = read(["test_batch.bin"])
    return X_train, y_train, X_test, y_test


def random_crop_flip(x, padding=4, rng=None):
    """Standard CIFAR augmentation on a batch: pad + random crop, random h-flip."""
    rng = rng or np.random.default_rng()
    N, C, H, W = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="reflect")
    out = np.empty_like(x)
    dy = rng.integers(0, 2 * padding + 1, N)
    dx = rng.integers(0, 2 * padding + 1, N)
    flip = rng.random(N) < 0.5
    for i in range(N):
        crop = padded[i, :, dy[i] : dy[i] + H, dx[i] : dx[i] + W]
        out[i] = crop[:, :, ::-1] if flip[i] else crop
    return out
