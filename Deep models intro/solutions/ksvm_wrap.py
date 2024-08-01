from typing import Tuple

import data
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class KSVMWrap:
    def __init__(
            self,
            X: ndarray,
            Y_true: ndarray,
            param_svm_c: float = 1,
            param_svm_gama: str = 'auto'
    ):
        self.clf = svm.SVC(
            probability=True,
            kernel='rbf',
            decision_function_shape='ovo',
            C=param_svm_c,
            gamma=param_svm_gama
        )
        self.clf.fit(X, Y_true)

    def predict(self, X: ndarray) -> ndarray:
        """
        :param X: data
        :return: predicted classes
        """
        return self.clf.predict(X)

    def get_scores(self, X: ndarray, Y_true: ndarray) -> Tuple[float, Tuple[float, float], ConfusionMatrixDisplay]:
        Y_pred = self.predict(X)
        return data.eval_perf_multi(Y_pred, Y_true)

    def support(self) -> ndarray:
        """
        :return: support vectors indices
        """
        return self.clf.support_


def display_confusion_matrix(Y_pred: ndarray, Y_true: ndarray) -> None:
    """
    :param Y_pred: Predicted labels
    :param Y_true: True labels
    """
    classes: ndarray = np.unique(Y_true)
    cm = confusion_matrix(Y_true, Y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="YlGnBu")
    plt.show()


def statistics(
        model: KSVMWrap,
        X: ndarray,
        Y_true: ndarray,
        Y_pred: ndarray
) -> None:
    """
    Prints Accuracy, precision and recall and plots confusion matrix
    :param model: model, type: KSVMWrap
    :param X: data
    :param Y_pred: Predicted labels
    :param Y_true: True labels
    """
    accuracy, p_r, cm = model.get_scores(X, Y_true)

    print(f"Accuracy: {accuracy}")
    i = 0
    for precision, recall in p_r:
        print(f"For class {i}, Precision: {precision}, Recall: {recall}")
        i += 1
    display_confusion_matrix(Y_pred, Y_true)


def train_and_visualize(X: ndarray, Y_true: ndarray) -> None:
    svm_model_3 = KSVMWrap(X, Y_true)
    Y_pred = svm_model_3.predict(X)

    print(f"Broj potpornih vektora: {len(svm_model_3.support())}")

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(function=svm_model_3.predict, rect=rect)
    data.graph_data(X, Y_true, Y_pred, svm_model_3.support())
    plt.show()

    statistics(svm_model_3, X, Y_true, Y_pred)


if __name__ == '__main__':
    task2 = False
    task3 = True
    task4 = False

    if task2:
        X, Y_true = data.sample_gmm_2d(4, 2, 20)
        train_and_visualize(X, Y_true)

    if task3:
        X, Y_true = data.sample_gmm_2d(12, 2, 200)
        train_and_visualize(X, Y_true)

    if task4:
        X, Y_true = data.sample_gmm_2d(6, 2, 100)
        train_and_visualize(X, Y_true)
