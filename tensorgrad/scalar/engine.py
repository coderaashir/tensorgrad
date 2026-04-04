"""A scalar-valued autograd engine.

Each ``Value`` wraps a single Python float. Doing math on Values builds a graph
(a DAG) where every node remembers its parents and how to send gradient back to
them. Calling ``backward()`` on the output walks that graph in reverse.
"""


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        # Called during the backward pass. It reads self.grad and adds the
        # right amount into each parent's .grad (chain rule).
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # for debugging / graph visualisation

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Backpropagate from this node through the whole graph.

        We need every node's gradient to be complete before it pushes gradient
        to its parents, so we visit nodes in reverse topological order.
        """
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0  # d(self)/d(self)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
