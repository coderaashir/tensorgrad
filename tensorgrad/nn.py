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


# ------------------------------------------------------------------ layers
def kaiming_normal(fan_in, shape):
    """He initialisation: keeps activation variance roughly constant through
    ReLU layers, so deep nets don't start with vanishing/exploding signals."""
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)


class Linear(Module):
    """y = x @ W + b, with W of shape (in_features, out_features)."""

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(kaiming_normal(in_features, (in_features, out_features)))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out

    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features}, bias={self.bias is not None})"


# ------------------------------------------------------------- activations
class ReLU(Module):
    def forward(self, x):
        return x.relu()

    def __repr__(self):
        return "ReLU()"


class Tanh(Module):
    def forward(self, x):
        return x.tanh()

    def __repr__(self):
        return "Tanh()"


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"


class Flatten(Module):
    """(N, d1, d2, ...) -> (N, d1*d2*...)"""

    def forward(self, x):
        return x.flatten(1)

    def __repr__(self):
        return "Flatten()"


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        inner = "\n".join(
            f"  ({i}): " + repr(layer).replace("\n", "\n  ") for i, layer in enumerate(self.layers)
        )
        return f"Sequential(\n{inner}\n)"
