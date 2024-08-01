import torch.nn
from torch import nn
from torch.nn import ReLU, Linear


class BaselineModel(torch.nn.Module):
    def __init__(self, emb_matrix, hidden_size: int = 150):
        super().__init__()
        self.emb_matrix = emb_matrix
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.seq = torch.nn.Sequential(
            Linear(in_features=300, out_features=hidden_size),
            ReLU(),
            Linear(in_features=hidden_size, out_features=hidden_size),
            ReLU(),
            Linear(in_features=hidden_size, out_features=1)
        )

    def forward(self, X):
        # print(X.shape)
        X = self.emb_matrix(X)
        # print(X.shape)
        X = self.avg_pool(X.permute(0, 2, 1)).squeeze(-1)
        # print(X.shape)
        X = X.float()
        return torch.squeeze(self.seq(X))
