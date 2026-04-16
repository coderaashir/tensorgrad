"""An N-dimensional Tensor with reverse-mode autograd, backed by NumPy.

Same idea as the scalar ``Value`` engine, but each node holds a whole array.
That changes two things:

* Every op's backward has to work on arrays (e.g. matmul's gradient is another
  matmul, not a product of two numbers).
* NumPy broadcasting means an input can be smaller than the output. Its
  gradient must be summed back down to the input's shape.
"""

import numpy as np


def _to_array(data, dtype=None):
    if isinstance(data, Tensor):
        data = data.data
    arr = np.asarray(data, dtype=dtype)
    if dtype is None and not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    return arr


def _unbroadcast(grad, shape):
    """Sum ``grad`` down to ``shape``, undoing NumPy broadcasting.

    Broadcasting can (1) add leading dimensions and (2) stretch size-1
    dimensions. If an input was stretched to produce the output, each of its
    elements was used several times, so its gradient is the sum over those uses.
    """
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

    # ------------------------------------------------------------------ basics
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

    def __repr__(self):
        grad = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{grad})"

    # ------------------------------------------------------------ graph helpers
    @staticmethod
    def _make(data, parents, op, backward):
        """Create an output tensor and, if needed, hook it into the graph.

        ``backward`` is a function ``g -> tuple of grads`` (one per parent)
        where ``g`` is the gradient flowing into the output.
        """
        out = Tensor(data)
        out.requires_grad = any(p.requires_grad for p in parents)
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

        # Iterative topological sort (a recursive one hits Python's recursion
        # limit on long graphs, e.g. unrolled training loops).
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

    # --------------------------------------------------------- elementwise ops
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
            # d(a/b)/da = 1/b,  d(a/b)/db = -a/b^2
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

    # ------------------------------------------------------------- unary ops
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
        # sigmoid(x) = (1 + tanh(x/2)) / 2, which never overflows (1/(1+e^-x)
        # does for very negative x).
        s = 0.5 * (1 + np.tanh(0.5 * self.data))
        return Tensor._make(s, (self,), "sigmoid", lambda g: (g * s * (1 - s),))

    def relu(self):
        mask = self.data > 0
        return Tensor._make(self.data * mask, (self,), "relu", lambda g: (g * mask,))

    # ------------------------------------------------------------ reductions
    def sum(self, axis=None, keepdims=False):
        shape = self.shape

        def backward(g):
            # Every input element contributed once to its output element, so
            # the gradient is just copied (broadcast) back out.
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
            # Gradient goes to the max element(s); ties share it equally.
            mask = (x == full).astype(x.dtype)
            mask = mask / mask.sum(axis=axis, keepdims=True)
            return (mask * g,)

        return Tensor._make(out, (self,), "max", backward)

    # ---------------------------------------------------------------- matmul
    def __matmul__(self, other):
        other = _ensure(other)
        a, b = self.data, other.data
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("matmul needs inputs with at least 2 dims")

        def backward(g):
            # For C = A @ B:  dA = dC @ B^T,  dB = A^T @ dC.
            # With batch dims, the batch part may have been broadcast, so we
            # unbroadcast like any other op.
            ga = g @ np.swapaxes(b, -1, -2)
            gb = np.swapaxes(a, -1, -2) @ g
            return _unbroadcast(ga, a.shape), _unbroadcast(gb, b.shape)

        return Tensor._make(a @ b, (self, other), "@", backward)

    def __rmatmul__(self, other):
        return _ensure(other) @ self


def _ensure(x):
    return x if isinstance(x, Tensor) else Tensor(x)
