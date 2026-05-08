"""Stateless functions on Tensors: activations and losses."""

import numpy as np

from .tensor import Tensor


def relu(x):
    return x.relu()


def tanh(x):
    return x.tanh()


def sigmoid(x):
    return x.sigmoid()


def log_softmax(x, axis=-1):
    """log(softmax(x)), computed stably as one fused op.

    Forward:  y = x - max(x) - log(sum(exp(x - max(x))))
    Backward: dx = g - softmax(x) * sum(g)

    Building this out of exp/sum/log ops would also work, but the fused
    version is faster and never overflows.
    """
    z = x.data - x.data.max(axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=axis, keepdims=True))
    out = z - logsumexp
    softmax = np.exp(out)

    def backward(g):
        return (g - softmax * g.sum(axis=axis, keepdims=True),)

    return Tensor._make(out, (x,), "log_softmax", backward)


def softmax(x, axis=-1):
    return log_softmax(x, axis=axis).exp()


def cross_entropy(logits, targets):
    """Mean negative log-likelihood of integer class ``targets``.

    logits: (N, C) Tensor of unnormalised scores
    targets: (N,) array/Tensor of class indices in [0, C)
    """
    if isinstance(targets, Tensor):
        targets = targets.data
    targets = np.asarray(targets).astype(np.int64)
    logp = log_softmax(logits, axis=-1)
    picked = logp[np.arange(len(targets)), targets]
    return -picked.mean()


def mse_loss(pred, target):
    target = target if isinstance(target, Tensor) else Tensor(target)
    return ((pred - target) ** 2).mean()


def accuracy(logits, targets):
    """Fraction of rows whose argmax matches the target (no gradient)."""
    if isinstance(targets, Tensor):
        targets = targets.data
    preds = logits.data.argmax(axis=-1)
    return float((preds == np.asarray(targets)).mean())


# ---------------------------------------------------------------- im2col
def _out_size(size, k, stride, padding):
    return (size + 2 * padding - k) // stride + 1


def im2col(x, kh, kw, stride=1, padding=0):
    """Unfold image patches into rows so convolution becomes one matmul.

    x: (N, C, H, W) array  ->  (N * Ho * Wo, C * kh * kw) array

    Each row holds every input value one output pixel "sees". Uses a strided
    view, so the only copy is the final reshape.
    """
    N, C, H, W = x.shape
    Ho, Wo = _out_size(H, kh, stride, padding), _out_size(W, kw, stride, padding)
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    # windows: (N, C, H', W', kh, kw) for every stride-1 position
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride][:, :, :Ho, :Wo]
    # -> (N, Ho, Wo, C, kh, kw) so each output pixel's patch is contiguous
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * kh * kw)
    return cols, Ho, Wo


def col2im(cols, x_shape, kh, kw, stride=1, padding=0):
    """The adjoint of im2col: scatter-add patch rows back into an image.

    Overlapping patches touched the same input pixel several times, so their
    gradients add up. We loop over the kh*kw kernel offsets (small) rather
    than over output pixels (large).
    """
    N, C, H, W = x_shape
    Ho, Wo = _out_size(H, kh, stride, padding), _out_size(W, kw, stride, padding)
    cols = cols.reshape(N, Ho, Wo, C, kh, kw)
    out = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            out[:, :, i : i + stride * Ho : stride, j : j + stride * Wo : stride] += cols[
                :, :, :, :, i, j
            ].transpose(0, 3, 1, 2)
    if padding:
        out = out[:, :, padding:-padding, padding:-padding]
    return out
