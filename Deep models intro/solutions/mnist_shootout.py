import math
import os
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import torchvision

import data
import pickle

from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from timeit import default_timer as timer
from numpy import ndarray
from torch import nn
from tqdm.auto import tqdm
from pathlib import Path
from torch import Tensor
from torchvision.transforms import ToTensor
from typing import Tuple
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DATASET_ROOT = '../data/'
MODEL_ROOT = '../model/'
DATA_ROOT = '../data/model_data'


class MNISTDeepModel(nn.Module):
    def __init__(self, dimensions: list):
        super().__init__()

        if len(dimensions) < 2:
            raise ValueError("Dimensions must be at least size of 2!")

        self.layerStack: nn.Sequential = nn.Sequential(nn.Flatten())

        for i in range(len(dimensions) - 1):
            self.layerStack.append(
                nn.Linear(in_features=dimensions[i], out_features=dimensions[i + 1])
            )
            if i == len(dimensions) - 2:
                self.layerStack.append(nn.Softmax(dim=1))
            else:
                self.layerStack.append(nn.ReLU())

    def forward(self, X: Tensor) -> Tensor:
        return self.layerStack(X)


def download_data() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    downloaded: bool = Path(DATASET_ROOT + "MNIST").exists()

    mnist_train: MNIST = torchvision.datasets.MNIST(
        DATASET_ROOT,
        transform=ToTensor(),
        train=True,
        download=not downloaded
    )
    mnist_test: MNIST = torchvision.datasets.MNIST(
        DATASET_ROOT,
        transform=ToTensor(),
        train=False,
        download=not downloaded
    )

    X_train: Tensor = mnist_train.data
    Y_train: Tensor = mnist_train.targets
    X_test: Tensor = mnist_test.data
    Y_test: Tensor = mnist_test.targets
    X_train, X_test = X_train.float().div_(255.0), X_test.float().div_(255.0)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    return X_train.to(device), X_test.to(device), Y_train.to(device), Y_test.to(device)


def get_dimensions(X_train: Tensor, Y_train: Tensor) -> Tuple[int, int, int]:
    N: int = X_train.shape[0]
    D: int = X_train.shape[1] * X_train.shape[2]
    C: int = Y_train.max().add_(1).item()

    print(f"N = {N}, D = {D}, C = {C}")

    return N, D, C


def show_image(image: ndarray):
    plt.imshow(image, cmap="gray")
    plt.show()


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


def visualize_task_2(
        model_names: str,
        Y_test: ndarray,
        Y_train: ndarray,
        models_predictions_test: ndarray,
        models_predictions_train: ndarray,
        all_test_losses: float,
        all_train_losses: ndarray,
        iterations: int
) -> None:
    display_confusion_matrix(Y_pred=numpy.int64(models_predictions_train), Y_true=Y_train, title=f"{model_names} train")
    display_confusion_matrix(Y_pred=numpy.int64(models_predictions_test), Y_true=Y_test, title=f"{model_names} test")
    statistics(Y_pred=numpy.int64(models_predictions_train), Y_true=Y_train)
    statistics(Y_pred=numpy.int64(models_predictions_test), Y_true=Y_test)
    plot_loss(all_train_losses, all_test_losses, f"{model_names} train losses", iterations=iterations)


def show_image_grid(
        num_rows: int,
        num_cols: int,
        images: Tensor,
        labels: Tensor
) -> None:
    torch.manual_seed(42)
    fig = plt.figure(figsize=(12, 12))

    for i in range(1, num_rows * num_cols + 1):
        random_index = torch.randint(0, len(images), size=[1]).item()
        img = images[random_index]
        label = labels[random_index].item()
        fig.add_subplot(num_rows, num_cols, i)
        plt.imshow(img, cmap="gray")
        plt.title(label)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def print_model(deep_model: MNISTDeepModel):
    print("Parameters:\n")
    for name, module in deep_model.named_modules():
        print(f"{name}: {module}\n")

    print("==========================================")


def save_model(model: MNISTDeepModel, name: str):
    model_path = Path(MODEL_ROOT)
    model_path.mkdir(
        parents=True,
        exist_ok=True
    )

    model_save_path = MODEL_ROOT + name
    print(f"Saving model to: {model_save_path}")
    torch.save(
        obj=model.state_dict(),
        f=model_save_path
    )


def print_train_time(start: float, end: float) -> None:
    total_time = end - start
    print(f"Train time :{total_time:.3f} seconds")


def train(
        model: MNISTDeepModel,
        X_train: Tensor,
        Y_train: Tensor,
        X_validation: Tensor,
        Y_validation: Tensor,
        save_model_: bool = False,
        name: str = '',
        num_iterations: int = 1000,
        lr: float = 0.1,
        weight_decay: float = 0.001,
        tolerance: int = 5,
        check_validation_iteration: int = 100
) -> Tuple[ndarray, ndarray]:
    if num_iterations < 1:
        raise ValueError("Neispravan broj iteracija")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

    losses = np.zeros(num_iterations)
    last_validation_increase: int = -1
    last_validation_loss: float = 0.0

    train_time_start = timer()
    for i in tqdm(range(num_iterations)):
        model.train()
        Y_pred: Tensor = model(X_train)
        loss = loss_fn(Y_pred, Y_train)
        for param_name, w in model.named_parameters():
            if 'weight' in param_name:
                loss += torch.norm(w) * weight_decay

        losses[i] = loss
        if i % 100 == 0:
            print(f"Iteration: {i}, Loss: {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % check_validation_iteration == 0:
            model.eval()
            with torch.inference_mode():
                Y_pred_val: Tensor = model(X_validation)
                validation_loss: Tensor = loss_fn(Y_pred_val, Y_validation)
                if last_validation_increase == -1:
                    last_validation_increase = i
                    last_validation_loss = validation_loss.item()
                else:
                    if validation_loss < last_validation_loss:
                        last_validation_loss = validation_loss.item()
                        last_validation_increase = i
                    else:
                        if i - last_validation_increase > tolerance:
                            print(
                                f"Early stopping...Validation loss not increased in last {i - last_validation_increase} iterations.")
                            break

    train_time_end = timer()

    print_train_time(
        start=train_time_start,
        end=train_time_end
    )

    if save_model_:
        save_model(model, name if name.endswith(".pth") else name + ".pth")

    return losses, numpy.float64(Y_pred.detach().numpy())


def test(
        model: MNISTDeepModel,
        X_test: Tensor,
        Y_test: Tensor
) -> Tuple[Tensor, ndarray]:
    model.eval()
    with torch.inference_mode():
        Y_pred: Tensor = model(X_test)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(Y_pred, Y_test)
    return loss, numpy.float64(Y_pred.detach().numpy())


def to_predictions(prob: ndarray):
    return np.int64(np.argmax(prob, axis=1))


def display_confusion_matrix(Y_pred: ndarray, Y_true: ndarray, title: str = '') -> None:
    """
    Displays confusion matrix based on predicted labels Y_pred and true labels Y_true
    :param title:
    :param Y_pred: True labels
    :param Y_true: Predicted labels
    """
    classes: ndarray = np.unique(Y_true)
    cm = confusion_matrix(Y_true, Y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="YlGnBu")
    plt.title(title)
    plt.show()


def plot_loss(
        train_losses: ndarray,
        test_loss: float,
        title: str,
        iterations: int
) -> None:
    plt.plot(range(np.count_nonzero(train_losses)), train_losses[train_losses != 0])
    plt.scatter([iterations + 1], test_loss, s=20, c="r", marker="s")
    plt.title(title)
    plt.show()


def task2():
    X_train, X_test, Y_train, Y_test = download_data()
    N, D, C = get_dimensions(X_train, Y_train)

    dimensions_1: list[int] = [D, C]
    dimensions_2: list[int] = [D, 100, C]
    dimensions_3: list[int] = [D, 100, 100, C]
    dimensions_4: list[int] = [D, 100, 100, 100, C]
    dimensions: list[list] = [dimensions_1, dimensions_2, dimensions_3, dimensions_4]
    dimensions_string: list[str] = ["D_C", "D_100_C", "D_100_100_C", "D_100_100_100_C"]
    iterations: list[int] = [2000, 3000, 10000, 20000]
    lrs: list[str] = ['0.2', '0.15', '0.12', '0.08']

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    for i in range(len(dimensions)):
        model = MNISTDeepModel(dimensions[i]).to(device)
        model_name: str = f"task_2_model_lr_{lrs[i].replace('.', '_')}_iter_{iterations[i]}_dim_{dimensions_string[i]}"
        train_test_visualize(
            X_test,
            X_train,
            Y_test,
            Y_train,
            model,
            iterations[i],
            model_name,
            save=True,
            lr=float(lrs[i]),
            show_digits=True
        )


def train_test_visualize(
        X_test: Tensor,
        X_train: Tensor,
        Y_test: Tensor,
        Y_train: Tensor,
        model: MNISTDeepModel,
        model_iter: int,
        model_name: str,
        save=False,
        weight_decay=0.001,
        lr=0.001,
        show_digits: bool = False
) -> None:
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    if model_name.endswith(".pth"):
        model_path_str = MODEL_ROOT + model_name
    else:
        model_path_str = MODEL_ROOT + model_name + ".pth"
    model_path = Path(model_path_str)
    if model_path.exists():
        model.load_state_dict(torch.load(f=model_path))
        print(f"{model_path_str} model already exists... Skipping training")

    train_loss_data_path_str = DATA_ROOT + model_name + 'train_loss_data.npy'
    train_loss_data_path = Path(train_loss_data_path_str)
    train_probs_data_path_str = DATA_ROOT + model_name + 'train_probs_data.npy'
    train_probs_data_path = Path(train_probs_data_path_str)

    if train_loss_data_path.exists() and train_probs_data_path.exists():
        model_train_loss = numpy.load(train_loss_data_path)
        probs_train_model = numpy.load(train_probs_data_path)
        print("Data saved... Skipping training")
    else:
        model_train_loss, probs_train_model = train(
            model=model,
            X_train=X_train,
            Y_train=Y_train,
            X_validation=X_validation,
            Y_validation=Y_validation,
            num_iterations=model_iter,
            lr=lr,
            weight_decay=weight_decay,
            name=model_name,
            save_model_=save
        )

    model_test_loss, probs_model = test(model=model, X_test=X_test, Y_test=Y_test)
    visualize_task_2(
        model_names=model_name,
        models_predictions_test=to_predictions(probs_model),
        models_predictions_train=to_predictions(probs_train_model),
        all_test_losses=model_test_loss.detach(),
        all_train_losses=model_train_loss,
        iterations=model_iter,
        Y_test=Y_test.detach().numpy(),
        Y_train=Y_train.detach().numpy()
    )

    if not (train_loss_data_path.exists() and train_probs_data_path.exists()):
        numpy.save(train_loss_data_path, model_train_loss)
        numpy.save(train_probs_data_path, probs_train_model)

    if show_digits:
        plot_digits(model)


def regularization_task():
    X_train, X_test, Y_train, Y_test = download_data()
    N, D, C = get_dimensions(X_train, Y_train)

    dimensions: list[int] = [D, C]
    weight_decays: list[str] = ['0.0001', '0.01', '1', '100']
    num_iter: int = 1000
    model = MNISTDeepModel(dimensions=dimensions)

    for wd in weight_decays:
        model_name = f"model_1_reg_{wd.replace('.', '_')}.pth"
        train_test_visualize(
            X_test,
            X_train,
            Y_test,
            Y_train,
            model,
            num_iter,
            model_name,
            save=True,
            lr=0.1,
            weight_decay=float(wd)
        )


def make_batch(batch_size: int, X_train: Tensor, Y_train: Tensor) -> list[list]:
    data_size: int = len(Y_train)
    batches: list = []

    num_of_batches: int = int(data_size / batch_size)
    for i in range(num_of_batches):
        batches.append((X_train[i * batch_size:(i + 1) * batch_size], Y_train[i * batch_size:(i + 1) * batch_size]))

    if num_of_batches * batch_size != data_size:
        batches.append((X_train[num_of_batches * batch_size:], Y_train[num_of_batches * batch_size:]))

    return batches


def train_mb(
        X_train: Tensor,
        Y_train: Tensor,
        X_validation: Tensor,
        Y_validation: Tensor,
        model: MNISTDeepModel,
        batch_size: int = 32,
        epochs: int = 10,
        lr: float = 0.1,
        weight_decay: float = 0.001,
        use_adam: bool = False,
        use_adam_step: bool = False,
        check_validation_iteration: int = 1,
        tolerance: int = 2
):
    losses: list = []

    loss_fn = nn.CrossEntropyLoss()
    if use_adam:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - 1e-4)

    model.train()
    train_time_start = timer()
    last_validation_increase: int = -1
    last_validation_loss: float = 0.0
    predictions: ndarray = np.array([])

    for i in tqdm(range(epochs)):
        X_train_numpy = X_train.numpy()
        Y_train_numpy = Y_train.numpy()

        permutation = np.random.permutation(X_train_numpy.shape[0])
        X_train_shuffled = X_train_numpy[permutation]
        Y_train_shuffled = Y_train_numpy[permutation]

        X_train = torch.from_numpy(X_train_shuffled)
        Y_train = torch.from_numpy(Y_train_shuffled)

        batch: list[list] = make_batch(batch_size, X_train, Y_train)
        predictions: ndarray = np.array([])
        for (X, Y) in batch:
            Y_pred: Tensor = model(X)
            loss: Tensor = loss_fn(Y_pred, Y)
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if predictions.size == 0:
                predictions = Y_pred.detach().numpy()
            else:
                predictions = np.concatenate((predictions, Y_pred.detach().numpy()))

        if i % check_validation_iteration == 0:
            model.eval()
            with torch.inference_mode():
                Y_pred: Tensor = model(X_validation)
                validation_loss: Tensor = loss_fn(Y_pred, Y_validation)
                if last_validation_increase == -1:
                    last_validation_increase = i
                    last_validation_loss = validation_loss.item()
                else:
                    if validation_loss < last_validation_loss:
                        last_validation_loss = validation_loss.item()
                        last_validation_increase = i
                    else:
                        if i - last_validation_increase > tolerance:
                            print(
                                f"Early stopping...Validation loss not increased in last {i - last_validation_increase} iterations.")
                            break

        print(f"End of epoch {i + 1}., Loss: {loss}")
        if use_adam and use_adam_step:
            scheduler.step()

    train_time_end = timer()

    print_train_time(
        start=train_time_start,
        end=train_time_end
    )

    return losses, numpy.float64(predictions), Y_train_shuffled


def batch_training(
        save_model_: bool = False,
        epochs: int = 10,
        batch_size: int = 32,
        use_adam: bool = False,
        use_adam_step: bool = False
):
    X_train, X_test, Y_train, Y_test = download_data()
    N, D, C = get_dimensions(X_train, Y_train)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    model = MNISTDeepModel(dimensions=[D, C])
    model_name = f"batch_size_model_epochs_{epochs}_b_size_{batch_size}"
    if use_adam:
        model_name = model_name + "_adam_opt"
    if use_adam_step and use_adam:
        model_name = model_name + "_with_var_lr"

    if model_name.endswith(".pth"):
        model_path_str = MODEL_ROOT + model_name
    else:
        model_path_str = MODEL_ROOT + model_name + ".pth"
    model_path = Path(model_path_str)
    if model_path.exists():
        model.load_state_dict(torch.load(f=model_path))
        print(f"{model_path_str} model already exists...Importing trained weights")

    train_loss_data_path_str = DATA_ROOT + model_name + 'train_loss_data.npy'
    train_loss_data_path = Path(train_loss_data_path_str)
    train_probs_data_path_str = DATA_ROOT + model_name + 'train_probs_data.npy'
    train_probs_data_path = Path(train_probs_data_path_str)
    shuffled_Y_train_data_path_str = DATA_ROOT + model_name + 'shuffled_Y_train.npy'
    shuffled_Y_train_data_path = Path(shuffled_Y_train_data_path_str)

    if train_loss_data_path.exists() and train_probs_data_path.exists():
        train_losses = numpy.load(train_loss_data_path)
        probs_train_model = numpy.load(train_probs_data_path)
        Y_train = numpy.load(shuffled_Y_train_data_path)
        print("Data saved... Skipping training")
    else:
        train_losses, probs_train_model,Y_train  = train_mb(
            X_train=X_train,
            Y_train=Y_train,
            X_validation=X_validation,
            Y_validation=Y_validation,
            model=model,
            lr=1e-4 if use_adam else 0.1,
            epochs=epochs,
            batch_size=batch_size,
            use_adam=use_adam,
            use_adam_step=use_adam_step
        )

    test_losses, probs_model = test(model, X_test, Y_test)
    if isinstance(train_losses[0], torch.Tensor):
        train_losses = np.array([train_loss.detach() for train_loss in train_losses])

    visualize_task_2(
        model_names=model_name,
        models_predictions_test=to_predictions(probs_model),
        models_predictions_train=to_predictions(probs_train_model),
        all_test_losses=test_losses.detach(),
        all_train_losses=train_losses,
        iterations=len(train_losses),
        Y_test=Y_test.detach().numpy(),
        Y_train=Y_train
    )

    if save_model_:
        save_model(model, model_path_str)

    if not (train_loss_data_path.exists() and train_probs_data_path.exists()):
        numpy.save(train_loss_data_path, train_losses)
        numpy.save(train_probs_data_path, probs_train_model)
        numpy.save(shuffled_Y_train_data_path, Y_train)


def batch_training_task():
    # batch_training(batch_size=1000, epochs=1000) ## Ovdje se dogodi early stopping ali pri samom kraju
    batch_training(save_model_=True, batch_size=2)
    batch_training(save_model_=True, batch_size=16)
    batch_training(save_model_=True, batch_size=32, epochs=100)
    batch_training(save_model_=True, batch_size=64)
    batch_training(save_model_=True, batch_size=256, epochs=100)
    batch_training(save_model_=True, batch_size=512, epochs=100)


def adam_opt_task():
    batch_training(save_model_=True, use_adam=True)
    batch_training(save_model_=True, use_adam=True, use_adam_step=True)


def model_without_training_task():
    X_train, X_test, Y_train, Y_test = download_data()
    N, D, C = get_dimensions(X_train, Y_train)

    model = MNISTDeepModel(dimensions=[D, C])
    loss, probs_model = test(model, X_test, Y_test)
    model_predictions = to_predictions(probs_model)
    display_confusion_matrix(
        Y_pred=numpy.int64(model_predictions),
        Y_true=Y_test.detach().numpy(),
        title=f"Un trained model test"
    )
    statistics(Y_pred=numpy.int64(model_predictions), Y_true=Y_test.detach().numpy())
    print(f"Loss: {loss}")


def svm_task():
    X_train, X_test, Y_train, Y_test = download_data()
    X_train, X_test, Y_train, Y_test = X_train.detach().numpy(), X_test.detach().numpy(), Y_train.detach().numpy(), Y_test.detach().numpy()

    model_path_str = MODEL_ROOT + "svm_model"
    model_path = Path(model_path_str)
    model_saved = model_path.exists()
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    if model_saved:
        clf = pickle.load(open(model_path, "rb"))
        print(f"{model_path_str} model already exists... Skipping training")
    else:
        clf = svm.SVC(
            probability=True,
            kernel='rbf',
            decision_function_shape='ovo',
            C=1,
            gamma='auto'
        )
        train_time_start = timer()
        print("Training started.")
        clf.fit(X_train, Y_train)
        print("Training finished")
        train_time_end = timer()
        print_train_time(train_time_start, train_time_end)

    print("Predicting...")
    Y_pred = clf.predict(X_test)
    display_confusion_matrix(Y_pred=Y_pred, Y_true=Y_test)
    if not model_saved:
        print("Saving model...")
        pickle.dump(clf, open(model_path, "wb"))


def plot_digits(model: MNISTDeepModel):
    layer = 1
    for name, w in model.named_parameters():
        if 'weight' in name:
            show_digits_grid(2, 5, w.detach().numpy(), layer)
            layer += 1


def show_digits_grid(num_rows: int, num_cols: int, images: ndarray, layer: int):
    fig = plt.figure(figsize=(12, 12))

    for i in range(1, num_rows * num_cols + 1):
        img = images[i - 1]
        img = img.reshape(int(math.sqrt(len(img))), int(math.sqrt(len(img))))
        fig.add_subplot(num_rows, num_cols, i, title=f'Layer: {layer},Digit: {i - 1}')
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    task2()
    regularization_task()
    batch_training_task()
    adam_opt_task()
    model_without_training_task()
    svm_task()


if __name__ == "__main__":
    main()
