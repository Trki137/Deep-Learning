from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelArguments:
    input_size: int
    hidden_size: int
    num_of_layers: int
    dropout: float
    bidirectional: bool
    type: Literal['GRU', 'VANILLA', 'LSTM']
    use_attention: bool
    attention_size: int


def __init__(
        self,
        input_size: int = 300,
        hidden_size: int = 150,
        num_of_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
        type: Literal['GRU', 'VANILLA', 'LSTM'] = "GRU",
        use_attention: bool = False,
        attention_size: int = 75
):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_of_layers = num_of_layers
    self.dropout = dropout
    self.bidirectional = bidirectional
    self.type = type
    self.use_attention = use_attention
    self.attention_size = attention_size
