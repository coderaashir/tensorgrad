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
