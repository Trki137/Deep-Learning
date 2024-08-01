import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from common.Arguments import Arguments
from baseline_model.BaselineModel import BaselineModel
from common.util import train, evaluate
from data_loading.data_util import prepare_data


def main(args: Arguments):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch_emb, train_data, valid_data, test_data = prepare_data(args)

    model = BaselineModel(torch_emb)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.epochs)):
        train(model, train_data, optimizer, criterion, args)
        f1, acc, loss = evaluate(model, valid_data, criterion, f'Epoch {epoch}.jpg', args, f'Epoch {epoch + 1}',
                                 args.seed)
        print(f"Epoch {epoch + 1}: valid accuracy = {acc * 100}")

    f1, acc, loss = evaluate(model, test_data, criterion, f'Test.jpg', args, 'Test', args.seed)
    print(f"Test accuracy = {acc * 100}")


def loop(args: Arguments):
    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch_emb, train_data, valid_data, test_data = prepare_data(args)

        model = BaselineModel(torch_emb)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in tqdm(range(args.epochs)):
            train(model, train_data, optimizer, criterion, args)
            f1, acc, loss = evaluate(model, valid_data, criterion, f'Seed_{seed}_Epoch_{epoch}.jpg', args,
                                     f'Epoch {epoch + 1}',
                                     seed)
            print(f"Epoch {epoch + 1}: valid accuracy = {acc * 100}")

        f1, acc, loss = evaluate(model, test_data, criterion, f'Seed_{seed}_Test.jpg', args, 'Test', seed)
        print(f"Test accuracy = {acc * 100}")


if __name__ == '__main__':
    args = Arguments(
        seeds=[1, 2, 3, 4, 5],
        seed=7052020,
        clip=1.0,
        lr=1e-4,
        test_batch_size=32,
        train_batch_size=10,
        valid_batch_size=32,
        save_data_path="./result_data",
        save_csv_filename="/results.csv",
        save_image_folder="/images",
        epochs=5,
        weight_decay=0.0,
        min_freq=1,
        vocabulary_size=-1
    )

    main(args)
    # loop(args)
