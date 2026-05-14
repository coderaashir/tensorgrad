"""Neural-network modules, PyTorch-style.

A ``Module`` is any object with a ``forward``. It finds its ``Parameter``s by
walking its own attributes (including sub-modules and lists of sub-modules),
so layers don't have to register anything by hand.
"""

from collections import OrderedDict

import numpy as np

from . import functional as F
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

    def state_dict(self):
        return OrderedDict((name, p.data.copy()) for name, p in self.named_parameters())

    def load_state_dict(self, state):
        params = dict(self.named_parameters())
        missing = set(params) - set(state)
        unexpected = set(state) - set(params)
        if missing or unexpected:
            raise KeyError(f"missing: {sorted(missing)}, unexpected: {sorted(unexpected)}")
        for name, value in state.items():
            if params[name].shape != value.shape:
                raise ValueError(f"{name}: expected {params[name].shape}, got {value.shape}")
            params[name].data[...] = value

    def save(self, path):
        """Save parameters to a .npz file."""
        np.savez(path, **self.state_dict())

    def load(self, path):
        with np.load(path) as f:
            self.load_state_dict({k: f[k] for k in f.files})
        return self

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


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size * kernel_size
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = Parameter(kaiming_normal(fan_in, shape))
        self.bias = Parameter(np.zeros(out_channels)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return (
            f"Conv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding})"
        )


class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride)

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


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


class Dropout(Module):
    """Zero each activation with probability p during training.

    Survivors are scaled by 1/(1-p) ("inverted dropout") so the expected value
    is unchanged and eval mode can just pass inputs through.
    """

    def __init__(self, p=0.5):
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        return x * Tensor(mask)

    def __repr__(self):
        return f"Dropout(p={self.p})"


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
