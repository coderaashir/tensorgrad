# tensorgrad

A from-scratch autograd engine in Python + NumPy.

It starts as a scalar engine (one number per node, micrograd-style), grows into an
N-dimensional `Tensor` with broadcasting, and ends up training an MLP and a CNN on
MNIST and CIFAR-10.

Work in progress.
