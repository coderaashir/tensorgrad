"""Train a small CNN on MNIST with tensorgrad.

    conv(1->16, 3x3) -> relu -> maxpool 2   28x28 -> 14x14
    conv(16->32, 3x3) -> relu -> maxpool 2  14x14 -> 7x7
    flatten -> linear(1568->128) -> relu -> dropout -> linear(128->10)

    python examples/train_mnist_cnn.py --epochs 3
"""

import argparse

import numpy as np

from tensorgrad import nn, optim
from tensorgrad.data import load_mnist
from tensorgrad.trainer import fit


def build_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 10),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset", type=int, default=0, help="use only N training images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/mnist_cnn.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    X_train, y_train, X_test, y_test = load_mnist()
    if args.subset:
        X_train, y_train = X_train[: args.subset], y_train[: args.subset]

    model = build_model()
    print(model)
    print(f"parameters: {model.num_parameters():,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    fit(
        model,
        optimizer,
        (X_train, y_train, X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        history_path=args.out,
    )


if __name__ == "__main__":
    main()
