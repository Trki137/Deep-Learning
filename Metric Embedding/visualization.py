import os.path
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
from MNISTMetricDataset import MNISTMetricDataset
from SimpleMetricEmbeddingModel import SimpleMetricEmbedding

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_colormap():
    colormap = np.zeros((10, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    return colormap


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")
    emb_size = 32
    model = SimpleMetricEmbedding(1, emb_size).to(device)

    model_dir_path = './models'
    model_name = 'model_with_all_classes.pt'
    model_path = os.path.join(model_dir_path, model_name)
    if not Path(model_path).exists():
        raise ValueError(f"Model with path {model_path} doesn't exist")

    model.load_state_dict(torch.load(Path(model_path)))

    colormap = get_colormap()
    mnist_download_root = "./mnist/"
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    X = ds_test.images
    Y = ds_test.targets

    print("Fitting PCA directly from images...")
    test_img_rep2d = torch.pca_lowrank(X.view(-1, 28 * 28), 2)[0]
    plt.figure()
    scatter = plt.scatter(test_img_rep2d[:, 0], test_img_rep2d[:, 1], c=Y, cmap='tab10', s=5)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)), title="Classes")
    plt.savefig("./images/model_with_all_classes")
    plt.show()
    plt.figure()

    print("Fitting PCA from feature representation")
    with torch.no_grad():
        model.eval()
        test_rep = model.get_features(X.unsqueeze(1))
        test_rep2d = torch.pca_lowrank(test_rep, 2)[0]
        plt.figure()
        scatter = plt.scatter(test_rep2d[:, 0], test_rep2d[:, 1], c=Y, cmap='tab10', s=5)
        plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)), title="Classes")
        plt.savefig("./images/model_with_all_classes_img2")
        plt.show()
