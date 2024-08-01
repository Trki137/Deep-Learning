import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_size, attention_size)
        self.W2 = nn.Linear(attention_size, 1)

    def forward(self, h):
        a = self.W2(torch.tanh(self.W1(h))).squeeze(-1)
        alpha = torch.softmax(a, dim=1)
        return torch.sum(h * alpha.unsqueeze(-1), dim=1)

