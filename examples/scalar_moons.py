"""Train the scalar MLP on a 2-D "two moons" toy dataset.

This is deliberately small: every number is its own graph node, so the scalar
engine is slow (~30s for this run). The point is to see that backprop works end to end before we
build the tensor version.

    python examples/scalar_moons.py
"""

import math
import random

from tensorgrad.scalar.nn import MLP


def make_moons(n=100, noise=0.1, seed=0):
    rng = random.Random(seed)
    X, y = [], []
    for i in range(n):
        t = math.pi * rng.random()
        if i % 2 == 0:
            x0, x1, label = math.cos(t), math.sin(t), 1
        else:
            x0, x1, label = 1 - math.cos(t), 0.5 - math.sin(t), -1
        X.append([x0 + rng.gauss(0, noise), x1 + rng.gauss(0, noise)])
        y.append(label)
    return X, y


def main():
    random.seed(1337)
    X, y = make_moons(100)
    model = MLP(2, [16, 16, 1])
    print(model)
    print("parameters:", len(model.parameters()))

    for step in range(60):
        scores = [model(x) for x in X]
        # SVM "max-margin" loss + L2 regularisation
        losses = [(1 + -yi * si).relu() for yi, si in zip(y, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))
        reg_loss = 1e-4 * sum(p * p for p in model.parameters())
        loss = data_loss + reg_loss
        acc = sum((yi > 0) == (si.data > 0) for yi, si in zip(y, scores)) / len(y)

        model.zero_grad()
        loss.backward()

        lr = 1.0 - 0.9 * step / 60
        for p in model.parameters():
            p.data -= lr * p.grad

        if step % 5 == 0 or step == 59:
            print(f"step {step:3d}  loss {loss.data:.4f}  accuracy {acc * 100:.1f}%")


if __name__ == "__main__":
    main()
