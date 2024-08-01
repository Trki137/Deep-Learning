import csv
from pathlib import Path

import numpy as np
import torch.nn
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader

from common.Arguments import Arguments
from common.CSVResult import CSVResult


def display_cfm(y_true: np.ndarray, y_pred: np.ndarray, image_name, args: Arguments) -> None:
    classes = np.ndarray([0, 1])
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)

    file_path = args.save_data_path + args.save_image_folder + "/" + image_name
    plt.savefig(file_path)
    plt.close()


def save_results_to_csv(args: Arguments, f1: float, acc: float, loss: float, seed: int, desc: str):
    filepath = args.save_data_path + args.save_csv_filename
    file_exists = Path(filepath).exists()

    with open(filepath, 'a', newline='') as file:
        headers = ['Learning Rate', 'Epochs', 'F1 Score', 'Accuracy', 'Average Loss', 'Description', 'Seed']
        writer = csv.DictWriter(file, fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        data = {
            'Learning Rate': args.lr,
            'Epochs': args.epochs,
            'F1 Score': f1,
            'Accuracy': acc,
            'Average Loss': loss,
            'Description': desc,
            'Seed': seed
        }

        writer.writerow(data)


def train(model: torch.nn.Module, data: DataLoader, optimizer: torch.optim, criterion: torch.nn, args: Arguments):
    model.train()
    for batch, y, lengths in data:
        model.zero_grad()
        logits = model(batch)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()


def evaluate(model: torch.nn.Module, data: DataLoader, criterion: torch.nn, cfm_image_name: str,
             args: Arguments, desc: str, seed: int):
    model.eval()
    loss_sum = 0.0

    all_pred: np.ndarray = np.array([], dtype=np.int32)
    all_true: np.ndarray = np.array([], dtype=np.int32)

    with torch.inference_mode():
        for batch, y, lengths in data:
            logits: torch.Tensor = model(batch)
            loss = criterion(logits, y)
            loss_sum += loss.item()
            prob = torch.sigmoid(logits).detach().numpy()
            predictions = np.where(prob > 0.5, 1, 0)
            all_pred = np.concatenate((all_pred, predictions))
            all_true = np.concatenate((all_true, y.detach().numpy().astype(int)))

    display_cfm(all_true, all_pred, cfm_image_name, args)
    f1 = f1_score(all_true, all_pred)
    acc = accuracy_score(all_true, all_pred)
    loss = loss_sum / len(data)
    save_results_to_csv(args, f1, acc, loss, seed, desc)
    return f1, acc, loss_sum / len(data)


def csv_results(filepath: str, csv_data: CSVResult):
    file_exists = Path(filepath).exists()

    with open(filepath, 'a', newline='') as file:
        headers = ['RNN type', 'Input size', 'Hidden size', 'Number of layers', 'Dropout', 'Bidirectional', 'Loss',
                   'Description', 'Vocabulary size', 'Min frequency', 'Accuracy', 'F1_score', 'Learning rate', 'Epochs',
                   'Baseline Model']
        writer = csv.DictWriter(file, fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        data = {
            'RNN type': csv_data.type,
            'Input size': csv_data.input_size,
            'Hidden size': csv_data.hidden_size,
            'Number of layers': csv_data.num_of_layers,
            'Dropout': csv_data.dropout,
            'Bidirectional': csv_data.bidirectional,
            'Loss': csv_data.loss,
            'Description': csv_data.description,
            'Vocabulary size': csv_data.vocab_size,
            'Min frequency': csv_data.min_freq,
            'Accuracy': csv_data.acc,
            'F1_score': csv_data.f1_score,
            'Learning rate': csv_data.lr,
            'Epochs': csv_data.epochs,
            'Baseline Model': csv_data.baseline_model
        }

        writer.writerow(data)


def csv_to_dict(filepath: str) -> list[dict]:
    csv_data = []
    with open(filepath, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            csv_data.append(dict(row))
    return csv_data


def save_dataframe_as_image(df, filepath: str):
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
