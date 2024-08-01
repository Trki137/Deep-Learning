import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        conv_layer = nn.Conv2d(in_channels=num_maps_in, out_channels=num_maps_out, kernel_size=k, bias=bias)
        batch_norm_layer = nn.BatchNorm2d(num_maps_out)
        relu_layer = nn.ReLU()

        self.add_module("convolutional_layer", conv_layer)
        self.add_module("batch_norm_layer", batch_norm_layer)
        self.add_module("relu_layer", relu_layer)


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.network = nn.Sequential(
            _BNReluConv(num_maps_in=input_channels, num_maps_out=emb_size),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _BNReluConv(num_maps_in=emb_size, num_maps_out=emb_size),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _BNReluConv(num_maps_in=emb_size, num_maps_out=emb_size),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, X):
        return self.network(X)

    def get_features(self, img):
        batch_size = img.shape[0]
        X = self.forward(img)
        return X.view(batch_size, -1)

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        anchor_positive_distance = F.pairwise_distance(a_x, p_x)
        anchor_negative_distance = F.pairwise_distance(a_x, n_x)
        diff = anchor_positive_distance - anchor_negative_distance

        return torch.maximum(diff, torch.tensor(0)).mean()


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        return torch.flatten(img)
