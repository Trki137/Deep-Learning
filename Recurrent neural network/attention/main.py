import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from common.Arguments import Arguments
from common.ModelArguments import ModelArguments
from common.util import train, evaluate
from data_loading.data_util import prepare_data
from hiperparameters.rnn_cell_comparison import create_entry
from rnn.MyRNN import MyRNN


def main(arg: Arguments, model_arg: ModelArguments):
    filepath = './result_data/results.csv'
    seed = arg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch_emb, train_data, valid_data, test_data = prepare_data(arg)

    model = MyRNN(torch_emb, model_arg)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

    for epoch in tqdm(range(arg.epochs)):
        train(model, train_data, optimizer, criterion, arg)
        f1, acc, loss = evaluate(model, valid_data, criterion,
                                 f'Attention_Epoch_{epoch}_model_n_layers{model_arg.num_of_layers}_rnn_type_{model_arg.type}.jpg',
                                 arg, f'Epoch {epoch + 1}',
                                 arg.seed)
        create_entry(f'Validation loss for epoch {epoch}.', filepath, arg, model_arg, f1,
                     acc,
                     loss, False)
        print(f"Epoch {epoch + 1}: valid accuracy = {acc * 100}")

    f1, acc, loss = evaluate(model, test_data, criterion,
                             f'Attention_Test_model_n_layers_{model_arg.num_of_layers}_lr_{arg.lr}_clip_{arg.clip}_rnn_type_{model_arg.type}.jpg',
                             arg,
                             'Test', arg.seed)

    create_entry('Test loss.', filepath, arg, model_arg, f1, acc, loss, False)
    print(f"Test accuracy = {acc * 100}")

if __name__ == '__main__':
    args = Arguments(
        seeds=[1, 2, 3, 4, 5],
        seed=4,
        clip=0.25,
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

    model_args = ModelArguments(
        input_size=300,
        hidden_size=150,
        num_of_layers=2,
        dropout=0,
        bidirectional=False,
        type="VANILLA",
        use_attention=True,
        attention_size=75
    )
    main(args, model_args)
