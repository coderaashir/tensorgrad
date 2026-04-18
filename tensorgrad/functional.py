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
