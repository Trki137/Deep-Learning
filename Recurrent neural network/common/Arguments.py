from dataclasses import dataclass


@dataclass
class Arguments:
    seeds: list[int]
    seed: int
    clip: float
    lr: float
    weight_decay: float
    test_batch_size: int
    train_batch_size: int
    valid_batch_size: int
    epochs: int
    save_data_path: str
    save_image_folder: str
    save_csv_filename: str
    min_freq: int
    vocabulary_size: int


def __init__(
        self,
        seeds: list[int],
        seed: int,
        clip: float,
        lr: float,
        test_batch_size: int,
        train_batch_size: int,
        valid_batch_size: int,
        save_data_path: str,
        save_image_folder: str,
        save_csv_filename: str,
        epochs: int,
        min_freq: int = 1,
        vocabulary_size: int = -1,
        weight_decay: float = 0.0

):
    self.seeds = seeds
    self.seed = seed
    self.clip = clip
    self.lr = lr,
    self.weight_decay = weight_decay,
    self.test_batch_size = test_batch_size
    self.train_batch_size = train_batch_size
    self.valid_batch_size = valid_batch_size
    self.save_data_path = save_data_path
    self.save_image_folder = save_image_folder
    self.save_csv_filename = save_csv_filename
    self.epochs = epochs
    self.min_freq = min_freq
    self.vocabulary_size = vocabulary_size
