import time
import torch.optim
from torch.utils.data import DataLoader

from MNISTMetricDataset import MNISTMetricDataset
from SimpleMetricEmbeddingModel import SimpleMetricEmbedding, IdentityModel
from utils import train, evaluate, compute_representations

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False

def main(model_name, remove_class):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    mnist_download_root = "./mnist/"
    ds_train = MNISTMetricDataset(mnist_download_root, split='train', remove_class=remove_class)
    ds_train_repr = MNISTMetricDataset(mnist_download_root, split='train')

    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    train_loader_repr = DataLoader(
            ds_train_repr,
            batch_size=64,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            drop_last=True
        )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    use_identity_model = False
    if use_identity_model:
        identity_model = IdentityModel().to(device)
        repr = compute_representations(identity_model, train_loader, num_classes, 28 * 28, device)
        acc = evaluate(identity_model, repr, traineval_loader, device)
        print(f"Acc: {round(acc * 100, 2)}%")
    else:
        emb_size = 32
        model = SimpleMetricEmbedding(1, emb_size).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )

        epochs = 3
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            t0 = time.time_ns()
            train_loss = train(model, optimizer, train_loader, device, model_name=model_name)
            print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
            if EVAL_ON_TEST or EVAL_ON_TRAIN:
                print("Computing mean representations for evaluation...")
                representations = compute_representations(model, train_loader_repr, num_classes, emb_size, device)
            if EVAL_ON_TRAIN:
                print("Evaluating on training set...")
                acc1 = evaluate(model, representations, traineval_loader, device)
                print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
            if EVAL_ON_TEST:
                print("Evaluating on test set...")
                acc1 = evaluate(model, representations, test_loader, device)
                print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
            t1 = time.time_ns()
            print(f"Epoch time (sec): {(t1 - t0) / 10 ** 9:.1f}")

if __name__ == '__main__':
    main(model_name="model_with_all_classes.pt", remove_class=None)
    main(model_name="model_without_zero_class.pt", remove_class=0)
