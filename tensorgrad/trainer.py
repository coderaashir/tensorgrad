"""A plain training/evaluation loop shared by the example scripts."""

import json
import os
import time

import numpy as np

from . import functional as F
from .data import DataLoader
from .tensor import Tensor, no_grad


def evaluate(model, X, y, batch_size=500):
    model.eval()
    total_loss, correct = 0.0, 0
    with no_grad():
        for xb, yb in DataLoader(X, y, batch_size=batch_size, shuffle=False):
            logits = model(Tensor(xb))
            total_loss += F.cross_entropy(logits, yb).item() * len(yb)
            correct += int((logits.data.argmax(axis=1) == yb).sum())
    model.train()
    return total_loss / len(y), correct / len(y)


def fit(
    model,
    optimizer,
    data,
    epochs,
    batch_size=64,
    log_every=100,
    lr_schedule=None,
    seed=0,
    history_path=None,
    augment=None,
):
    """Train ``model`` with cross-entropy."""
    X_train, y_train, X_test, y_test = data
    loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    history = {"step_loss": [], "epochs": []}
    step = 0

    for epoch in range(1, epochs + 1):
        if lr_schedule is not None:
            optimizer.lr = lr_schedule(epoch)
        model.train()
        t0 = time.time()
        running, running_acc = [], []
        for xb, yb in loader:
            if augment is not None:
                xb = augment(xb)
            logits = model(Tensor(xb))
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running.append(loss.item())
            running_acc.append(float((logits.data.argmax(axis=1) == yb).mean()))
            history["step_loss"].append(round(loss.item(), 5))
            step += 1
            if log_every and step % log_every == 0:
                print(
                    f"epoch {epoch} step {step:5d}  "
                    f"loss {np.mean(running[-log_every:]):.4f}  "
                    f"acc {np.mean(running_acc[-log_every:]) * 100:.2f}%"
                )

        test_loss, test_acc = evaluate(model, X_test, y_test)
        record = {
            "epoch": epoch,
            "lr": optimizer.lr,
            "train_loss": float(np.mean(running)),
            "train_acc": float(np.mean(running_acc)),
            "test_loss": test_loss,
            "test_acc": test_acc,
            "seconds": round(time.time() - t0, 1),
        }
        history["epochs"].append(record)
        print(
            f"== epoch {epoch}: train loss {record['train_loss']:.4f} "
            f"acc {record['train_acc'] * 100:.2f}% | test loss {test_loss:.4f} "
            f"acc {test_acc * 100:.2f}% | {record['seconds']}s"
        )
        if history_path:
            os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
            with open(history_path, "w") as f:
                json.dump(history, f)

    return history


def cosine_schedule(base_lr, epochs, warmup=0, min_lr=0.0):
    """Linear warmup for ``warmup`` epochs, then cosine decay to ``min_lr``."""
    import math

    def lr(epoch):
        if epoch <= warmup:
            return base_lr * epoch / warmup
        t = (epoch - warmup - 1) / max(1, epochs - warmup - 1)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))

    return lr
