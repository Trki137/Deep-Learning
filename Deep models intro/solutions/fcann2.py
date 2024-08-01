from typing import Tuple
import numpy as np
from numpy import ndarray
from data import sample_gmm_2d, graph_data, graph_surface
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def loss(Y_true: ndarray, probs: ndarray) -> float:
    """
    Calculates the loss
    :param Y_true: True labels
    :param probs: calculated probabilities
    :return: loss
    """
    correct_probs = probs[np.arange(len(Y_true)), Y_true]
    return -np.mean(np.log(correct_probs))



class fcann():
    def __init__(
            self,
            X: ndarray,
            Y_true: ndarray,
            hidden_layers: int = 5
    ):
        num_of_classes = len(np.unique(Y_true))
        self.X = X
        self.Y_true = Y_true
        self.W1 = np.random.randn(hidden_layers, X.shape[1])
        self.b1 = np.zeros(shape=(1, hidden_layers))
        self.W2 = np.random.randn(num_of_classes, hidden_layers)
        self.b2 = np.zeros(shape=(1, num_of_classes))

    def forward(self, X: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Forwards data from input part of graph to output
        :param X: Data
        :return: Returns probabilities, h1 -> result after ReLUM and s1 is result of first layer before ReLU
        """
        s1: ndarray = np.dot(X, np.transpose(self.W1)) + self.b1
        h1: ndarray = np.maximum(0, s1)
        s2: ndarray = np.dot(h1, self.W2.transpose()) + self.b2

        exp_scores: ndarray = np.exp(s2)
        sum_exp: ndarray = np.sum(exp_scores, axis=1, keepdims=True)

        return exp_scores / sum_exp, h1, s1

    def fcann2_train(
            self,
            num_of_iter: int = 100000,
            lr: float = 0.05,
            param_lambda: float = 0.001) -> None:
        """
        Trains parameters w and b for fcann class
        :param num_of_iter: broj iteracija treniranja
        :param lr: learning rate - stopa učenja
        :param param_lambda: koeficijent regularizacije
        :return: None
        """
        X, Y_true = self.X, self.Y_true
        N:int = X.shape[0]
        num_of_classes: int = len(np.unique(Y_true))
        Y_one_hot_encoded:ndarray = to_categorical(Y_true, num_classes=num_of_classes)

        for i in range(num_of_iter):
            Y_probs, h1, s1 = self.forward(X)

            it_loss = loss(Y_true, Y_probs) + np.linalg.norm(self.W1) * param_lambda + np.linalg.norm(self.W2) * param_lambda
            if i % (num_of_iter / 10) == 0:
                print(f"Iteration: {i}, Loss: {it_loss}")

            Gs2 = Y_probs - Y_one_hot_encoded  # N x C
            grad_W2 = np.dot(np.transpose(Gs2), h1) / N  + param_lambda * self.W2 # C x H
            grad_b2 = np.sum(np.transpose(Gs2), keepdims=True, axis=1) / N
            Gh1 = np.dot(Gs2, self.W2)  # N x H - jer je Gs2 = N x C, a W2 = C * H

            Gs1 = Gh1
            Gs1[s1 < 0] = 0

            grad_W1 = np.dot(np.transpose(Gs1), X) / N + param_lambda * self.W1
            grad_b1 = np.sum(np.transpose(Gs1), keepdims=True, axis=1) / N

            grad_b1 = grad_b1.reshape(self.b1.shape)
            grad_b2 = grad_b2.reshape(self.b2.shape)

            self.W2 += -lr * grad_W2
            self.b2 += -lr * grad_b2
            self.W1 += -lr * grad_W1
            self.b1 += -lr * grad_b1

    def fcann_classify(self, X) -> ndarray:
        """
        Classify data X using trained parameters w and b

        :param X: data to classify
        :return: predictions for data
        """
        probs, h1, s1 = self.forward(X)
        return np.argmax(probs, axis=1)


def main():
    np.random.seed(100)
    X, Y = sample_gmm_2d(nsamples=10, nclasses=2, ncomponents=6)
    X = (X - np.mean(X)) / np.std(X)
    fcann1 = fcann(X, Y)
    fcann1.fcann2_train(num_of_iter=100000)
    y_pred = fcann1.fcann_classify(X)

    classes = np.unique(Y)
    cm = confusion_matrix(Y, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.show()

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(rect=rect, function=fcann1.fcann_classify)
    graph_data(X, Y, y_pred)
    plt.show()


if __name__ == "__main__":
    main()
