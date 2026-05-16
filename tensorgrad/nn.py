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
    # Names of non-trainable arrays (e.g. BatchNorm running stats) that
    # should still be saved in checkpoints.
    _buffers = ()

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

    def _named_state(self, prefix=""):
        """(name, owner, attr) for every parameter and buffer, recursively."""
        for name, value in self._children():
            full = f"{prefix}{name}"
            if isinstance(value, Parameter):
                yield full, value, None
            else:
                yield from value._named_state(prefix=full + ".")
        for buf in self._buffers:
            yield f"{prefix}{buf}", self, buf

    def state_dict(self):
        state = OrderedDict()
        for name, owner, attr in self._named_state():
            arr = owner.data if attr is None else getattr(owner, attr)
            state[name] = np.array(arr, copy=True)
        return state

    def load_state_dict(self, state):
        entries = {name: (owner, attr) for name, owner, attr in self._named_state()}
        missing = set(entries) - set(state)
        unexpected = set(state) - set(entries)
        if missing or unexpected:
            raise KeyError(f"missing: {sorted(missing)}, unexpected: {sorted(unexpected)}")
        for name, value in state.items():
            owner, attr = entries[name]
            current = owner.data if attr is None else getattr(owner, attr)
            if current.shape != value.shape:
                raise ValueError(f"{name}: expected {current.shape}, got {value.shape}")
            if attr is None:
                owner.data[...] = value
            else:
                setattr(owner, attr, np.array(value, dtype=current.dtype))

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


class BatchNorm2d(Module):
    """Normalise each channel over (N, H, W), then scale and shift.

    Built entirely from existing Tensor ops, so autograd derives the (fairly
    involved) backward pass for us. Keeps running averages for eval mode.
    """

    _buffers = ("running_mean", "running_var")

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(np.ones(num_features))
        self.bias = Parameter(np.zeros(num_features))
        self.running_mean = np.zeros((1, num_features, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, num_features, 1, 1), dtype=np.float32)

    def forward(self, x):
        axes = (0, 2, 3)
        if self.training:
            mean = x.mean(axis=axes, keepdims=True)
            centered = x - mean
            var = (centered * centered).mean(axis=axes, keepdims=True)
            m = self.momentum
            n = x.size / x.shape[1]
            self.running_mean = (1 - m) * self.running_mean + m * mean.data
            # running var uses the unbiased estimate, like PyTorch
            self.running_var = (1 - m) * self.running_var + m * var.data * n / max(n - 1, 1)
            x_hat = centered / (var + self.eps).sqrt()
        else:
            x_hat = (x - Tensor(self.running_mean)) / np.sqrt(self.running_var + self.eps)
        C = self.num_features
        return x_hat * self.weight.reshape(1, C, 1, 1) + self.bias.reshape(1, C, 1, 1)

    def __repr__(self):
        return f"BatchNorm2d({self.num_features})"


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
