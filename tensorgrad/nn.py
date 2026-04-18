"""Neural-network modules, PyTorch-style.

A ``Module`` is any object with a ``forward``. It finds its ``Parameter``s by
walking its own attributes (including sub-modules and lists of sub-modules),
so layers don't have to register anything by hand.
"""

from collections import OrderedDict

import numpy as np

from .tensor import Tensor


class Parameter(Tensor):
    """A Tensor that a Module should train. Always requires grad."""

    def __init__(self, data):
        super().__init__(np.asarray(data, dtype=np.float32), requires_grad=True)

    def __repr__(self):
        return f"Parameter(shape={self.shape})"


class Module:
    training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ---------------------------------------------------------- traversal
    def _children(self):
        for name, value in vars(self).items():
            if isinstance(value, (Parameter, Module)):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, (Parameter, Module)):
                        yield f"{name}.{i}", item

    def named_parameters(self, prefix=""):
        for name, value in self._children():
            full = f"{prefix}{name}"
            if isinstance(value, Parameter):
                yield full, value
            else:
                yield from value.named_parameters(prefix=full + ".")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def modules(self):
        yield self
        for _, value in self._children():
            if isinstance(value, Module):
                yield from value.modules()

    def num_parameters(self):
        return sum(p.size for p in self.parameters())

    # -------------------------------------------------------------- state
    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self, mode=True):
        for m in self.modules():
            m.training = mode
        return self

    def eval(self):
        return self.train(False)

    def __repr__(self):
        lines = [f"{type(self).__name__}("]
        for name, value in self._children():
            if isinstance(value, Module):
                child = repr(value).replace("\n", "\n  ")
                lines.append(f"  ({name}): {child}")
        lines.append(")")
        return "\n".join(lines) if len(lines) > 2 else f"{type(self).__name__}()"
