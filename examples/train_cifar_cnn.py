"""Train a small VGG-style CNN on CIFAR-10 with tensorgrad.

    [conv-bn-relu x2, maxpool] at widths w, 2w, 4w, then a linear classifier.

    python examples/train_cifar_cnn.py --epochs 10 --width 32

This is CPU-only NumPy, so it's slow compared with PyTorch on a GPU; use
--width 16 or --subset for a quicker run.
"""

import argparse

import numpy as np

from tensorgrad import nn, optim
from tensorgrad.data import load_cifar10, random_crop_flip
from tensorgrad.trainer import cosine_schedule, fit


def conv_block(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()]


def build_model(width=32):
    w = width
    return nn.Sequential(
        *conv_block(3, w), *conv_block(w, w), nn.MaxPool2d(2),            # 32 -> 16
        *conv_block(w, 2 * w), *conv_block(2 * w, 2 * w), nn.MaxPool2d(2),  # 16 -> 8
        *conv_block(2 * w, 4 * w), nn.MaxPool2d(2),                       # 8 -> 4
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(4 * w * 4 * 4, 10),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--subset", type=int, default=0, help="use only N training images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="", help="path to save weights (.npz)")
    parser.add_argument("--out", default="results/cifar_cnn.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    X_train, y_train, X_test, y_test = load_cifar10()
    if args.subset:
        X_train, y_train = X_train[: args.subset], y_train[: args.subset]

    model = build_model(args.width)
    print(model)
    print(f"parameters: {model.num_parameters():,}")

    rng = np.random.default_rng(args.seed)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    fit(
        model,
        optimizer,
        (X_train, y_train, X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        history_path=args.out,
        lr_schedule=cosine_schedule(args.lr, args.epochs, warmup=1, min_lr=args.lr * 0.01),
        augment=None if args.no_augment else (lambda xb: random_crop_flip(xb, rng=rng)),
        log_every=50,
    )
    if args.save:
        model.save(args.save)
        print(f"saved weights to {args.save}")


if __name__ == "__main__":
    main()
