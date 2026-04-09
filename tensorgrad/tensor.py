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

        def backward(g):
            return g, g

        return Tensor._make(self.data + other.data, (self, other), "+", backward)

    def __mul__(self, other):
        other = _ensure(other)
        a, b = self.data, other.data

        def backward(g):
            return g * b, g * a

        return Tensor._make(a * b, (self, other), "*", backward)


def _ensure(x):
    return x if isinstance(x, Tensor) else Tensor(x)
