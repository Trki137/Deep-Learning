import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    start = 0
    end = 10
    weigth = 0.3
    bias = 0.7
    epochs = 100

    X = torch.arange(start, end, step=0.1)
    Y_true = weigth * X + bias

    optimizer = optim.SGD([a, b], lr=0.001)

    losses = np.zeros(epochs)
    my_grad_a = np.zeros(epochs)
    my_grad_b = np.zeros(epochs)
    grad_a = np.zeros(epochs)
    grad_b = np.zeros(epochs)

    for i in range(epochs):
        Y_pred = a * X + b

        diff = (Y_true - Y_pred)

        loss = torch.sum(diff ** 2) / len(X)
        losses[i] = loss
        loss.backward()

        grad_a[i] = a.grad.item()
        grad_b[i] = b.grad.item()
        my_grad_a[i] = (2 / len(X)) * torch.sum((X * a + b - Y_true) * X)
        my_grad_b[i] = (2 / len(X)) * torch.sum(b + X * a - Y_true)

        optimizer.step()
        optimizer.zero_grad()

        print(f'step: {i}, loss:{loss}, Y_:{Y_pred}, a:{a}, b {b}')
    plt.subplot(2, 1, 1)
    plot_grad(np.arange(0, 100, 1), grad_a, "Grad a")
    plt.subplot(2, 1, 2)
    plot_grad(np.arange(0, 100, 1), grad_b, "Grad b")
    plt.show()

    plt.subplot(2, 1, 1)
    plot_grad(np.arange(0, 100, 1), my_grad_a, "My grad a")
    plt.subplot(2, 1, 2)
    plot_grad(np.arange(0, 100, 1), my_grad_b, "My grad b")
    plt.show()


def plot_grad(X: np.ndarray, Y: np.ndarray, y_label: str):
    plt.plot(X, Y, label=y_label)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.tight_layout()


if __name__ == '__main__':
    main()
