"""Plot training curves from the JSON histories written by the example scripts.

    python examples/plot_results.py results/mnist_mlp.json results/mnist_cnn.json
    python examples/plot_results.py results/*.json --out assets/curves.png
"""

import argparse
import json
import os

import numpy as np


def smooth(values, k=50):
    if len(values) < k:
        return np.asarray(values)
    kernel = np.ones(k) / k
    return np.convolve(values, kernel, mode="valid")


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("histories", nargs="+")
    parser.add_argument("--out", default="assets/curves.png")
    args = parser.parse_args()

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for path in args.histories:
        with open(path) as f:
            h = json.load(f)
        name = os.path.splitext(os.path.basename(path))[0]
        steps_per_epoch = len(h["step_loss"]) / max(1, len(h["epochs"]))
        s = smooth(h["step_loss"])
        x = (np.arange(len(s)) + 50) / steps_per_epoch
        ax_loss.plot(x, s, label=f"{name} (train, smoothed)")
        epochs = [e["epoch"] for e in h["epochs"]]
        ax_acc.plot(epochs, [100 * e["test_acc"] for e in h["epochs"]], marker="o", label=name)

    ax_loss.set(xlabel="epoch", ylabel="cross-entropy", title="Training loss", yscale="log")
    ax_acc.set(xlabel="epoch", ylabel="accuracy (%)", title="Test accuracy")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
