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
    """log(softmax(x)), computed stably as one fused op."""
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
    """Mean negative log-likelihood of integer class ``targets``."""
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


def _out_size(size, k, stride, padding):
    return (size + 2 * padding - k) // stride + 1


def im2col(x, kh, kw, stride=1, padding=0):
    """Unfold image patches into rows so convolution becomes one matmul."""
    N, C, H, W = x.shape
    Ho, Wo = _out_size(H, kh, stride, padding), _out_size(W, kw, stride, padding)
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride][:, :, :Ho, :Wo]
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * kh * kw)
    return cols, Ho, Wo


def col2im(cols, x_shape, kh, kw, stride=1, padding=0):
    """The adjoint of im2col: scatter-add patch rows back into an image."""
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


def conv2d(x, weight, bias=None, stride=1, padding=0):
    """2-D convolution (really cross-correlation, like every DL library)."""
    F_, C, kh, kw = weight.shape
    N = x.shape[0]
    cols, Ho, Wo = im2col(x.data, kh, kw, stride, padding)
    w_flat = weight.data.reshape(F_, -1)
    out = cols @ w_flat.T
    if bias is not None:
        out = out + bias.data
    out = out.reshape(N, Ho, Wo, F_).transpose(0, 3, 1, 2)

    x_shape = x.shape

    def backward(g):
        G = g.transpose(0, 2, 3, 1).reshape(-1, F_)
        dw = (G.T @ cols).reshape(weight.shape) if weight.requires_grad else None
        dx = col2im(G @ w_flat, x_shape, kh, kw, stride, padding) if x.requires_grad else None
        if bias is None:
            return dx, dw
        return dx, dw, G.sum(axis=0)

    parents = (x, weight) if bias is None else (x, weight, bias)
    return Tensor._make(np.ascontiguousarray(out), parents, "conv2d", backward)


def max_pool2d(x, kernel_size=2, stride=None):
    """Max pooling over k x k windows. x: (N, C, H, W) -> (N, C, Ho, Wo)."""
    k = kernel_size
    s = stride or k
    N, C, H, W = x.shape
    Ho, Wo = (H - k) // s + 1, (W - k) // s + 1
    windows = np.lib.stride_tricks.sliding_window_view(x.data, (k, k), axis=(2, 3))
    windows = windows[:, :, ::s, ::s][:, :, :Ho, :Wo].reshape(N, C, Ho, Wo, k * k)
    argmax = windows.argmax(axis=-1)
    out = np.take_along_axis(windows, argmax[..., None], axis=-1)[..., 0]

    shape, dtype = x.shape, x.dtype

    def backward(g):
        dx = np.zeros(shape, dtype=dtype)
        for i in range(k):
            for j in range(k):
                mask = argmax == i * k + j
                dx[:, :, i : i + s * Ho : s, j : j + s * Wo : s] += g * mask
        return (dx,)

    return Tensor._make(out, (x,), "max_pool2d", backward)
