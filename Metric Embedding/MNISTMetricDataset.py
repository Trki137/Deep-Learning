import random

import torch
from torch import Tensor
from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision


class MNISTMetricDataset(Dataset):
    def __init__(self, root="/data/mnist/", split='train', remove_class: int | None = None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        if remove_class is not None:
            if remove_class not in range(0,10):
                raise ValueError(f"Neispravna klasa {remove_class}!")
            mask = remove_class != self.targets
            indicies = torch.argwhere(mask)
            self.images = self.images[indicies].squeeze()
            self.targets = self.targets[indicies].squeeze()

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index: int) -> int:
        target: int = self.targets[index]
        negative_index: int = random.randint(0, len(self.targets) - 1)
        while self.targets[negative_index] == target:
            negative_index = random.randint(0, len(self.targets) - 1)
        return negative_index

    def _sample_positive(self, index: int) -> int:
        target: Tensor = self.targets[index]
        positive_samples: list[int] = self.target2indices[target.item()]
        random_sample: int = choice(positive_samples)
        while random_sample == index:
            random_sample = choice(positive_samples)
        return random_sample

    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)
