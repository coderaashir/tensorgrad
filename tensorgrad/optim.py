"""Optimisers. Each one updates ``p.data`` in place using ``p.grad``."""

import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight decay.

        v = momentum * v + (grad + weight_decay * p)
        p = p - lr * v
    """

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self.momentum:
                v *= self.momentum
                v += g
                g = v
            p.data -= self.lr * g


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015), optionally with decoupled weight decay (AdamW).

        m = b1 * m + (1 - b1) * g          # running mean of gradients
        v = b2 * v + (1 - b2) * g^2        # running mean of squared gradients
        m_hat, v_hat = bias-corrected m, v (they start at 0)
        p = p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        c1 = 1 - self.b1**self.t
        c2 = 1 - self.b2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            update = (m / c1) / (np.sqrt(v / c2) + self.eps)
            if self.weight_decay:
                update = update + self.weight_decay * p.data
            p.data -= self.lr * update
