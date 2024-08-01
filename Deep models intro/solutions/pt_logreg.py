import torch
from torch import nn, Tensor
import numpy as np
from numpy import ndarray
import data
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class PTLogreg(nn.Module):
    def __init__(self, D: int, C: int, param_lambda: float):
        """
           :param D: dimensions of each datapoint
           :param C: number of classes
           :param param_lambda: regularization factor
        """
        super().__init__()
        self.W: Tensor = nn.Parameter(torch.randn(size=(C, D)), requires_grad=True)
        self.b: Tensor = nn.Parameter(torch.zeros(size=(1, C)), requires_grad=True)
        self.param_lambda: float = param_lambda

    def forward(self, X: Tensor) -> Tensor:
        """
        Advance pass through the model

        :param X: data
        :return: probabilities of a class for each row in X
        """
        scores: Tensor = torch.mm(X, torch.transpose(self.W, 0, 1)) + self.b
        return torch.softmax(scores, dim=1)

    def get_loss(self, X: Tensor, Yoh_: Tensor) -> Tensor:
        """
        Calculates the loss for data X with current parameters W, b and lambda

        :param X: data
        :param Yoh_: One hot encoded Y data points
        :return: loss for data X with current parameters
        """
        probs: Tensor = self.forward(X)
        indices: Tensor = Yoh_.nonzero()[:, 1]
        max_probs: Tensor = probs.gather(1, indices.unsqueeze(1))
        return -torch.mean(torch.log(max_probs)) + torch.norm(self.W) * self.param_lambda


def train(
        model: PTLogreg,
        X: ndarray,
        Y_true_oh: ndarray,
        param_niter: int,
        print_loss: bool,
        param_delta: float = 0.05,
        device: str = 'cpu'

) -> None:
    """
    :param model: model to train
    :param X: model inputs [NxD], type: ndarray
    :param Y_true_oh: ground truth [NxC], type: ndarray
    :param param_niter: number of training iterations
    :param print_loss: print loss data
    :param param_delta: learning rate
    :param device: where will computation occur
    """
    X: Tensor = torch.from_numpy(X).to(device).type(torch.float)
    Y_true_oh: Tensor = torch.from_numpy(Y_true_oh).to(device).type(torch.float)
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

    model.train()
    for i in range(param_niter):
        loss = model.get_loss(X, Y_true_oh)
        if print_loss and i % (param_niter / 10) == 0:
            print(f"Iteration: {i}, Loss: {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model: PTLogreg, X: ndarray, device: str = 'cpu') -> ndarray:
    """Arguments:
       :param device: device to make opperations
       :param model: type: PTLogreg
       :param X: actual datapoints [NxD], type: np.ndarray
       Returns: predicted class probability [NxC], type: np.ndarray
    """
    X = torch.from_numpy(X).to(device=device).type(torch.float)

    model.eval()
    with torch.inference_mode():
        Y_pred = model(X)

    Y_pred = Y_pred.detach()
    return Y_pred.numpy()


def display_confusion_matrix(Y_pred: ndarray, Y_true: ndarray) -> None:
    """
    Displays confusion matrix based on predicted labels Y_pred and true labels Y_true
    :param Y_pred: True labels
    :param Y_true: Predicted labels
    """
    classes: ndarray = np.unique(Y_true)
    cm = confusion_matrix(Y_true, Y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="YlGnBu")
    plt.show()


def log_reg(
        n_classes: int = 2,
        n_samples: int = 100,
        print_loss: bool = True,
        param_niter: int = 10000,
        param_delta: float = 0.01,
        param_lambda: float = 0.1
) -> None:
    """
    Trains model, evaluates model and prints graph surface and statistical data
    :param n_classes: Number of classes in our data
    :param n_samples: Number of samples per class
    :param print_loss: Print loss while training model
    :param param_niter: Number of iterations in training (epochs)
    :param param_delta: Learning rate parameter
    :param param_lambda: Regularization parameter
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y_true = data.sample_gauss_2d(n_classes, n_samples)
    Y_true_oh = data.class_to_onehot(Y_true)

    log_regression_model = PTLogreg(X.shape[1], Y_true_oh.shape[1], param_lambda=param_lambda).to(device)
    train(
        model=log_regression_model,
        X=X,
        Y_true_oh=Y_true_oh,
        param_niter=param_niter,
        print_loss=print_loss,
        param_delta=param_delta
    )

    probs: ndarray = evaluate(log_regression_model, X)
    Y_pred = np.argmax(probs, axis=1)

    statistics(Y_pred, Y_true)
    visualize_graph_surface(log_regression_model, device, X, Y_pred, Y_true)


def statistics(Y_pred: ndarray, Y_true: ndarray) -> None:
    """
    Prints Accuracy, precision and recall and plots confusion matrix
    :param Y_pred: Predicted labels
    :param Y_true: True labels
    """
    accuracy, p_r, cm = data.eval_perf_multi(Y_pred, Y_true)

    print(f"Accuracy: {accuracy}")

    for i, (precision, recall) in enumerate(p_r):
        print(f"For class {i}, Precision: {precision}, Recall: {recall}")

    display_confusion_matrix(Y_pred=Y_pred, Y_true=Y_true)


def visualize_graph_surface(
        model: PTLogreg,
        device: str,
        X: ndarray,
        Y_pred: ndarray,
        Y_true: ndarray
) -> None:
    """
    Visualizes graph surface for model and data

    :param model: model, type: PTLogreg
    :param device: device on which we do operations
    :param X: data
    :param Y_pred: predicted values
    :param Y_true: true values
    """

    def graph_surface_helper(X: ndarray) -> ndarray:
        probs: ndarray = evaluate(model, X, device)
        return np.argmax(probs, axis=1)

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(graph_surface_helper, rect, offset=0)
    data.graph_data(X, Y_true, Y_pred, special=[])
    plt.show()


def test_hiperparameters() -> None:
    """
    Test different parameters for regularization
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y_true = data.sample_gauss_2d(2, 100)
    Y_true_oh = data.class_to_onehot(Y_true)
    lambdas = np.arange(0, 10, 2, dtype=float)

    precisions = np.zeros(len(lambdas))
    accuracies = np.zeros(len(lambdas))
    recalls = np.zeros(len(lambdas))

    for i, p_lambda in enumerate(lambdas):
        log_regression_model = PTLogreg(X.shape[1], Y_true_oh.shape[1], param_lambda=p_lambda).to(device)
        train(model=log_regression_model, X=X, Y_true_oh=Y_true_oh, param_niter=1000, param_delta=0.01,
              print_loss=False)
        probs = evaluate(log_regression_model, X)

        Y_pred = np.argmax(probs, axis=1)
        accuracy, precision, recall = data.eval_perf_binary(Y_pred, Y_true)
        precisions[i] = precision
        recalls[i] = recall
        accuracies[i] = accuracy
        visualize_graph_surface(log_regression_model, device, X, Y_pred, Y_true)

    plt.subplot(3, 1, 1)
    plt.plot(lambdas, accuracies, label="Accuracy")
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")

    plt.subplot(3, 1, 2)
    plt.plot(lambdas, precisions, label="Precision")
    plt.xlabel("Lambda")
    plt.ylabel("Precision")

    plt.subplot(3, 1, 3)
    plt.plot(lambdas, recalls, label="Recall")
    plt.xlabel("Lambda")
    plt.ylabel("Recall")

    plt.show()


if __name__ == "__main__":
    np.random.seed(100)

    log_reg()
    log_reg(n_classes=3, n_samples=300)
    log_reg(n_classes=5, n_samples=100, print_loss=False)
    test_hiperparameters()
