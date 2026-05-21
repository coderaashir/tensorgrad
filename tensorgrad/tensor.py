"""An N-dimensional Tensor with reverse-mode autograd, backed by NumPy."""

import contextlib

import numpy as np

_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    """Context manager that stops ops from recording the graph."""
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


def is_grad_enabled():
    return _grad_enabled


def _to_array(data, dtype=None):
    if isinstance(data, Tensor):
        data = data.data
    arr = np.asarray(data, dtype=dtype)
    if dtype is None and not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    return arr


def _unbroadcast(grad, shape):
    """Sum grad down to shape, undoing broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, dtype=None):
        self.data = _to_array(data, dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = None
        self._prev = ()
        self._op = ""

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    def __len__(self):
        return len(self.data)

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def detach(self):
        """Same data, cut off from the graph."""
        return Tensor(self.data)

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        grad = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{grad})"

    @staticmethod
    def _make(data, parents, op, backward):
        """Create an output tensor and, if needed, hook it into the graph."""
        out = Tensor(data)
        out.requires_grad = _grad_enabled and any(p.requires_grad for p in parents)
        if out.requires_grad:
            out._prev = parents
            out._op = op
            out._backward = backward
        return out

    def _accumulate(self, g):
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.array(g, dtype=self.data.dtype, copy=True)
        else:
            self.grad = self.grad + g

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("backward() on a tensor that doesn't require grad")
        if grad is None:
            if self.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)

        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._prev:
                if id(parent) not in visited:
                    stack.append((parent, False))

        self._accumulate(_to_array(grad, self.data.dtype))
        for node in reversed(topo):
            if node._backward is None or node.grad is None:
                continue
            grads = node._backward(node.grad)
            for parent, g in zip(node._prev, grads):
                if g is not None:
                    parent._accumulate(g)

    def __add__(self, other):
        other = _ensure(other)

        sa, sb = self.shape, other.shape

        def backward(g):
            return _unbroadcast(g, sa), _unbroadcast(g, sb)

        return Tensor._make(self.data + other.data, (self, other), "+", backward)

    def __mul__(self, other):
        other = _ensure(other)
        a, b = self.data, other.data

        def backward(g):
            return _unbroadcast(g * b, a.shape), _unbroadcast(g * a, b.shape)

        return Tensor._make(a * b, (self, other), "*", backward)

    def __neg__(self):
        return Tensor._make(-self.data, (self,), "neg", lambda g: (-g,))

    def __sub__(self, other):
        other = _ensure(other)
        sa, sb = self.shape, other.shape

        def backward(g):
            return _unbroadcast(g, sa), _unbroadcast(-g, sb)

        return Tensor._make(self.data - other.data, (self, other), "-", backward)

    def __truediv__(self, other):
        other = _ensure(other)
        a, b = self.data, other.data

        def backward(g):
            return _unbroadcast(g / b, a.shape), _unbroadcast(-g * a / (b * b), b.shape)

        return Tensor._make(a / b, (self, other), "/", backward)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return _ensure(other) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return _ensure(other) / self

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only constant exponents"
        x = self.data

        def backward(g):
            return (g * exponent * x ** (exponent - 1),)

        return Tensor._make(x**exponent, (self,), f"**{exponent}", backward)

    def exp(self):
        e = np.exp(self.data)
        return Tensor._make(e, (self,), "exp", lambda g: (g * e,))

    def log(self):
        x = self.data
        return Tensor._make(np.log(x), (self,), "log", lambda g: (g / x,))

    def sqrt(self):
        return self**0.5

    def tanh(self):
        t = np.tanh(self.data)
        return Tensor._make(t, (self,), "tanh", lambda g: (g * (1 - t * t),))

    def sigmoid(self):
        # overflow-safe form of 1 / (1 + exp(-x))
        s = 0.5 * (1 + np.tanh(0.5 * self.data))
        return Tensor._make(s, (self,), "sigmoid", lambda g: (g * s * (1 - s),))

    def relu(self):
        mask = self.data > 0
        return Tensor._make(self.data * mask, (self,), "relu", lambda g: (g * mask,))

    def sum(self, axis=None, keepdims=False):
        shape = self.shape

        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, shape),)

        out = self.data.sum(axis=axis, keepdims=keepdims)
        return Tensor._make(out, (self,), "sum", backward)

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            n = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / n

    def max(self, axis=None, keepdims=False):
        x = self.data
        out = x.max(axis=axis, keepdims=keepdims)

        def backward(g):
            full = out if (keepdims or axis is None) else np.expand_dims(out, axis)
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            mask = (x == full).astype(x.dtype)
            mask = mask / mask.sum(axis=axis, keepdims=True)
            return (mask * g,)

        return Tensor._make(out, (self,), "max", backward)

    def __matmul__(self, other):
        other = _ensure(other)
        a, b = self.data, other.data
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("matmul needs inputs with at least 2 dims")

        def backward(g):
            ga = g @ np.swapaxes(b, -1, -2)
            gb = np.swapaxes(a, -1, -2) @ g
            return _unbroadcast(ga, a.shape), _unbroadcast(gb, b.shape)

        return Tensor._make(a @ b, (self, other), "@", backward)

    def __rmatmul__(self, other):
        return _ensure(other) @ self

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        old = self.shape
        return Tensor._make(
            self.data.reshape(shape), (self,), "reshape", lambda g: (g.reshape(old),)
        )

    def flatten(self, start_dim=1):
        return self.reshape(self.shape[:start_dim] + (-1,))

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        axes = axes or tuple(reversed(range(self.ndim)))
        inverse = tuple(np.argsort(axes))
        return Tensor._make(
            self.data.transpose(axes), (self,), "transpose", lambda g: (g.transpose(inverse),)
        )

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            idx = idx.data.astype(np.int64)
        elif isinstance(idx, tuple):
            idx = tuple(i.data.astype(np.int64) if isinstance(i, Tensor) else i for i in idx)
        shape, dtype = self.shape, self.dtype

        def backward(g):
            full = np.zeros(shape, dtype=dtype)
            # add.at so repeated indices accumulate
            np.add.at(full, idx, g)
            return (full,)

        return Tensor._make(self.data[idx], (self,), "getitem", backward)

    @staticmethod
    def concat(tensors, axis=0):
        tensors = [_ensure(t) for t in tensors]
        sizes = [t.shape[axis] for t in tensors]
        splits = np.cumsum(sizes)[:-1]

        def backward(g):
            return tuple(np.split(g, splits, axis=axis))

        data = np.concatenate([t.data for t in tensors], axis=axis)
        return Tensor._make(data, tuple(tensors), "concat", backward)


def _ensure(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def tensor(data, requires_grad=False, dtype=None):
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(*shape, requires_grad=False, dtype=np.float32):
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def ones(*shape, requires_grad=False, dtype=np.float32):
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)


def randn(*shape, requires_grad=False, dtype=np.float32):
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)
