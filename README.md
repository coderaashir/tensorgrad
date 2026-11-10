# tensorgrad

My own mini PyTorch, built from scratch on NumPy

Started with autograd on single numbers (`tensorgrad/scalar`), then moved to
full arrays, then built layers and optimizers on top and trained some models

## Results

| Model | Data | Test acc | Time/epoch |
|---|---|---|---|
| MLP | MNIST | 97.9% | 0.4s |
| CNN | MNIST | 99.0% | 24s |
| CNN + BatchNorm | CIFAR-10 | 83.8% | 8 min |

Logs in `results/`, plots in `assets/`.

## What's where

- `scalar/` – autograd on single numbers (the prototype)
- `tensor.py` – the real engine: arrays, broadcasting, `backward()`
- `functional.py` – softmax, cross-entropy, conv2d, max pool
- `nn.py` – Linear, Conv2d, BatchNorm, Dropout, Sequential
- `optim.py` – SGD, Adam
- `data.py`, `trainer.py` – datasets and the training loop
- `gradcheck.py` – checks gradients against finite differences

## Things worth remembering

- **backward()** sorts the graph, then walks it in reverse; each op passes
  gradient to its inputs (the chain rule).
- **Broadcasting:** if an input got stretched, its gradient has to be summed
  back down to its original shape (`_unbroadcast`).
- **Conv = matmul:** `im2col` turns patches into rows. In the backward pass,
  `col2im` *adds* gradients back because the windows overlap.
- **BatchNorm** is built from normal ops, so autograd does its backward for free.

## Run

```bash
pip install -e ".[dev]"
python examples/train_mnist_cnn.py --epochs 5
python examples/train_cifar_cnn.py --epochs 10   # slow, try --width 16
python examples/plot_results.py results/*.json
```
