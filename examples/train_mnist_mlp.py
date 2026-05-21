"""Train a 2-hidden-layer MLP on MNIST with tensorgrad."""

import argparse

import numpy as np

from tensorgrad import nn, optim
from tensorgrad.data import load_mnist
from tensorgrad.trainer import fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--subset", type=int, default=0, help="use only N training images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="", help="path to save weights (.npz)")
    parser.add_argument("--out", default="results/mnist_mlp.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    X_train, y_train, X_test, y_test = load_mnist(flatten=True)
    if args.subset:
        X_train, y_train = X_train[: args.subset], y_train[: args.subset]

    model = nn.Sequential(
        nn.Linear(784, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, args.hidden // 2),
        nn.ReLU(),
        nn.Linear(args.hidden // 2, 10),
    )
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
    if args.save:
        model.save(args.save)
        print(f"saved weights to {args.save}")


if __name__ == "__main__":
    main()
