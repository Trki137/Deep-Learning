from MNISTMetricDataset import MNISTMetricDataset

if __name__ == '__main__':
    dataset = MNISTMetricDataset(remove_class=0)
    dataset.__getitem__(2)