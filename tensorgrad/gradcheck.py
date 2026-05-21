"""Numerical gradient checking."""

import numpy as np

from .tensor import Tensor


def numerical_grad(f, inputs, eps=1e-6):
    """Central-difference estimate of d(sum(f(*inputs)))/d(input) for each input."""
    grads = []
    for x in inputs:
        g = np.zeros_like(x.data)
        it = np.nditer(x.data, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            orig = x.data[i]
            x.data[i] = orig + eps
            plus = f(*inputs).data.sum()
            x.data[i] = orig - eps
            minus = f(*inputs).data.sum()
            x.data[i] = orig
            g[i] = (plus - minus) / (2 * eps)
        grads.append(g)
    return grads


def gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4, verbose=False):
    """Check ``f``'s backward against finite differences."""
    inputs = [x if isinstance(x, Tensor) else Tensor(x) for x in inputs]
    for x in inputs:
        if x.dtype != np.float64:
            raise TypeError("gradcheck needs float64 inputs")
        x.requires_grad = True
        x.grad = None

    out = f(*inputs)
    out.backward(np.ones_like(out.data))
    analytic = [x.grad if x.grad is not None else np.zeros_like(x.data) for x in inputs]
    numeric = numerical_grad(f, inputs, eps)

    for k, (a, n) in enumerate(zip(analytic, numeric)):
        if verbose:
            print(f"input {k}: max abs diff {np.abs(a - n).max():.2e}")
        if not np.allclose(a, n, atol=atol, rtol=rtol):
            worst = np.abs(a - n).max()
            raise AssertionError(
                f"gradient mismatch for input {k} (max abs diff {worst:.3e})\n"
                f"analytic:\n{a}\nnumeric:\n{n}"
            )
    return True
