from dataclasses import dataclass
from typing import Literal


@dataclass
class CSVResult:
    input_size: int
    hidden_size: int
    num_of_layers: int
    dropout: float
    bidirectional: bool
    type: Literal['GRU', 'VANILLA', 'LSTM']
    loss: float
    acc: float
    f1_score: float
    lr: float
    epochs: int
    description: str
    baseline_model: bool
    min_freq: int
    vocab_size: int


def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_of_layers: int,
        dropout: float,
        bidirectional: bool,
        type: Literal['GRU', 'VANILLA', 'LSTM'],
        loss: float,
        acc: float,
        f1_score: float,
        lr: float,
        epochs: int,
        description: str,
        baseline_model:bool,
        min_freq: int,
        vocab_size: int
):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_of_layers = num_of_layers
    self.dropout = dropout
    self.bidirectional = bidirectional
    self.type = type
    self.loss = loss
    self.acc = acc
    self.f1_score = f1_score
    self.lr = lr
    self.epochs = epochs
    self.description = description
    self.baseline_model = baseline_model
    self.min_freq = min_freq
    self.vocab_size = vocab_size
