import torch
from torch import nn, Tensor
import numpy as np
from numpy import ndarray
from torch.nn import ParameterList

import data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class PTDeep(nn.Module):

    def __init__(
            self,
            dimensions: list,
            activation_function,
            param_lambda: float = 0.0001
    ):
        super().__init__()

        if len(dimensions) < 2:
            raise ValueError("Dimensions must be at least size of 2!")

        self.param_lambda: float = param_lambda
        self.activation_function = activation_function
        self.weights: ParameterList = nn.ParameterList(
            [nn.Parameter(torch.randn(size=(dimensions[dim], dimensions[dim + 1])), requires_grad=True) for dim in
             range(len(dimensions) - 1)]
        )

        self.biases: ParameterList = nn.ParameterList(
            [nn.Parameter(torch.zeros(size=(1, dimensions[dim + 1])), requires_grad=True) for dim in
             range(len(dimensions) - 1)]
        )

    def count_params(self):
        total_num_of_params = 0
        for params in self.weights.state_dict().values():
            total_num_of_params += params.shape[0] * params.shape[1]

        for params in self.biases.state_dict().values():
            total_num_of_params += params.shape[1]

        return total_num_of_params

    def forward(self, X: Tensor) -> Tensor:
        """
        Advance pass through the model

        :param X: Data
        :return: class probabilities for each data
        """
        for i in range(len(self.weights) - 1):
            X: Tensor = torch.mm(X, self.weights[i]) + self.biases[i]
            X: Tensor = self.activation_function(X)

        scores: Tensor = torch.mm(X, self.weights[-1]) + self.biases[-1]
        return torch.softmax(scores, dim=1)

    def get_loss(self, X: Tensor, Y_true_oh: Tensor) -> Tensor:
        """
        Calculates the loss for data X with current parameters W, b and lambda

        :param X: data
        :param Y_true_oh: One hot encoded Y data points
        :return: loss for data X with current parameters
        """
        probs: Tensor = self.forward(X) + 1e-13
        indices: Tensor = Y_true_oh.nonzero()[:, 1]
        max_probs: Tensor = probs.gather(1, indices.unsqueeze(1))
        loss: Tensor = -torch.mean(torch.log(max_probs))

        for W in self.weights:
            loss += torch.norm(W) * self.param_lambda

        return loss


def train(
        deep_model: PTDeep,
        X: ndarray,
        Y_true_oh: ndarray,
        param_niter: int,
        print_loss: bool,
        param_delta: float = 0.05,
        device: str = 'cpu'
) -> None:
    """
    :param deep_model:
    :param X: model inputs [NxD], type: np.ndarray
    :param Y_true_oh: ground truth [NxC], type: np.ndarray
    :param param_niter: number of training iterations
    :param print_loss: print loss in training loop
    :param param_delta: learning rate
    :param device: device where calculations are done
    """
    X: Tensor = torch.from_numpy(X).to(device).type(torch.float)
    Y_true_oh: Tensor = torch.from_numpy(Y_true_oh).to(device).type(torch.float)
    optimizer = torch.optim.SGD(deep_model.parameters(), lr=param_delta)

    deep_model.train()
    for i in range(param_niter):
        loss = deep_model.get_loss(X, Y_true_oh)
        if print_loss and i % 1000 == 0:
            print(f"Iteration: {i}, Loss: {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(deep_model: PTDeep, X: np.ndarray, device='cpu') -> np.ndarray:
    """
    Returns:

    :param deep_model: type: PTLogreg
    :param X: actual datapoints [NxD], type: np.ndarray
    :param device: device where calculations are done
    :return: predicted class probabilities [NxC], type: np.ndarray
    """
    X: Tensor = torch.from_numpy(X).to(device=device).type(torch.float)

    deep_model.eval()
    with torch.inference_mode():
        Y_pred = deep_model(X)

    Y_pred = Y_pred.detach()
    return Y_pred.numpy()


def train_and_visualize_ptdeep(
        dim: list,
        activation_function,
        X: ndarray,
        Y_true: ndarray,
        print_loss: bool = True,
        param_lambda: float = 0.1,
        param_delta: float = 0.05,
        param_niter: int = 10000,
        visualize_surface: bool = False
) -> None:
    """

    :param dim: model dimensions, input, output and hidden layers
    :param activation_function: activation function in hidden layers
    :param X: Data
    :param Y_true: True labels
    :param print_loss: print loss in training loop
    :param param_lambda: regularization factor
    :param param_niter: number of iterations in training loop
    :param param_delta: learning rate
    """
    Y_true_oh = data.class_to_onehot(Y_true)

    deep_model = PTDeep(
        dimensions=dim,
        activation_function=activation_function,
        param_lambda=param_lambda
    )
    train(
        deep_model=deep_model,
        X=X,
        Y_true_oh=Y_true_oh,
        param_niter=param_niter,
        print_loss=print_loss,
        param_delta=param_delta

    )
    probs: ndarray = evaluate(deep_model, X)
    Y_pred: ndarray = np.argmax(probs, axis=1)
    statistics(Y_pred=Y_pred, Y_true=Y_true)

    if visualize_surface:
        visualize_graph_surface(deep_model, 'cpu', X, Y_pred, Y_true)


def statistics(Y_pred: ndarray, Y_true: ndarray) -> None:
    """
    Prints Accuracy, precision and recall and plots confusion matrix
    :param Y_pred: Predicted labels
    :param Y_true: True labels
    :return:
    """
    accuracy, p_r, cm = data.eval_perf_multi(Y_pred, Y_true)

    print(f"Accuracy: {accuracy}")

    for i, (precision, recall) in enumerate(p_r):
        print(f"For class {i}, Precision: {precision}, Recall: {recall}")

    display_confusion_matrix(Y_pred=Y_pred, Y_true=Y_true)


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


def test_pt_deep_model():
    X1, Y_true_1 = data.sample_gmm_2d(4, 2, 40)
    X2, Y_true_2 = data.sample_gmm_2d(6, 2, 10)

    dim_1 = [2, 2]
    dim_2 = [2, 10, 2]
    dim_3 = [2, 10, 10, 2]
    param_lambda = 0.0001

    train_and_visualize_ptdeep(
        dim=dim_1,
        param_lambda=param_lambda,
        activation_function=relu_activation_fn,
        X=X1,
        Y_true=Y_true_1,
        visualize_surface=True
    )
    train_and_visualize_ptdeep(
        dim=dim_1,
        param_lambda=param_lambda,
        activation_function=relu_activation_fn,
        X=X2,
        Y_true=Y_true_2,
        visualize_surface=True
    )
    train_and_visualize_ptdeep(
        dim=dim_2,
        param_lambda=param_lambda,
        activation_function=relu_activation_fn,
        X=X1,
        Y_true=Y_true_1,
        param_delta=0.01,
        visualize_surface=True
    )
    train_and_visualize_ptdeep(
        dim=dim_2,
        param_lambda=param_lambda,
        activation_function=relu_activation_fn,
        X=X2,
        Y_true=Y_true_2,
        param_delta=0.01,
        visualize_surface=True
    )
    train_and_visualize_ptdeep(
        dim=dim_3,
        param_lambda=param_lambda,
        activation_function=relu_activation_fn,
        X=X1,
        Y_true=Y_true_1,
        param_delta=0.001,
        visualize_surface=True
    )
    train_and_visualize_ptdeep(
        dim=dim_3,
        param_lambda=param_lambda,
        activation_function=relu_activation_fn,
        X=X2,
        Y_true=Y_true_2,
        param_delta=0.001,
        visualize_surface=True
    )


def test_same_as_lr():
    X, Y_true = data.sample_gauss_2d(2, 100)
    train_and_visualize_ptdeep(
        dim=[2, 2],
        activation_function=relu_activation_fn,
        X=X,
        Y_true=Y_true,
        visualize_surface=True
    )


def print_model(deep_model: PTDeep):
    print(f"Number of parameters: {deep_model.count_params()}")
    print("Parameters:\n")
    for name, W in deep_model.named_parameters():
        print(f"{name}:\n {W}")

    print("==========================================")


def test_sigmoid_activaction_function():
    relu_activation_fn = torch.relu
    sigmoid_activation_fn = torch.sigmoid_
    X, Y_true = data.sample_gmm_2d(nclasses=2, ncomponents=6, nsamples=10)
    train_and_visualize_ptdeep(
        dim=[2, 10, 10, 2],
        activation_function=relu_activation_fn,
        X=X,
        Y_true=Y_true,
        visualize_surface=True,
        param_lambda=0.0001,
        param_delta=0.1
    )

    train_and_visualize_ptdeep(
        dim=[2, 10, 10, 2],
        activation_function=sigmoid_activation_fn,
        X=X,
        Y_true=Y_true,
        visualize_surface=True,
        param_lambda=0.0001,
        param_delta=0.1
    )


def test_lab_example():
    np.random.seed(100)
    model = PTDeep(
        dimensions=[2, 10, 10, 2],
        param_lambda=1e-4,
        activation_function=torch.relu
    )

    X, Y_true = data.sample_gmm_2d(6, 2, 10)

    Y_true_oh = data.class_to_onehot(Y_true)

    train(
        deep_model=model,
        X=X,
        Y_true_oh=Y_true_oh,
        param_niter=10000,
        print_loss=True,
        param_delta=0.1,
    )
    probs: ndarray = evaluate(model, X)
    Y_pred: ndarray = np.argmax(probs, axis=1)
    statistics(Y_pred=Y_pred, Y_true=Y_true)
    visualize_graph_surface(model, 'cpu', X, Y_pred, Y_true)


def pt_deep_surface(model: PTDeep):
    def classify(X: ndarray) -> ndarray:
        return np.argmax(evaluate(model, X), axis=1)

    return classify


def visualize_graph_surface(
        model: PTDeep,
        device: str,
        X: ndarray,
        Y_pred: ndarray,
        Y_true: ndarray
) -> None:
    """
    Visualizes graph surface for model and data

    :param model: model, type: PTDeep
    :param device: device on which we do operations
    :param X: data
    :param Y_pred: predicted values
    :param Y_true: true values
    """

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(pt_deep_surface(model), rect, offset=0)
    data.graph_data(X, Y_true, Y_pred, special=[])
    plt.show()


if __name__ == "__main__":
    print_model_1_params = False
    print_model_2_params = False
    train_same_as_lr = False
    test_pt_deep = False
    test_softmax = True
    lab_example = False

    relu_activation_fn = torch.relu

    model = PTDeep(dimensions=[2, 3], activation_function=relu_activation_fn)
    model2 = PTDeep(dimensions=[2, 5, 3], activation_function=relu_activation_fn)

    if print_model_1_params:
        print_model(model)

    if print_model_2_params:
        print_model(model2)

    if lab_example:
        test_lab_example()

    if train_same_as_lr:
        test_same_as_lr()

    if test_pt_deep:
        print("Testing deep models")
        test_pt_deep_model()

    if test_softmax:
        print("Training models with ReLU and softmax activation function")
        test_sigmoid_activaction_function()
