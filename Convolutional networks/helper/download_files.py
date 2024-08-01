import requests
from pathlib import Path

files = [
    "check_grads.py",
    "im2col_cython.pyx",
    "layers.py",
    "nn.py",
    "setup_cython.py",
    "train.py",
    "train_l2reg.py"
]
BASE_PATH = "..\\github_files"


def download_file(file_name: str, path: str):
    if file_name is None:
        raise ValueError("Missing file name")
    if path is None:
        raise ValueError("Missing path")

    if Path(path).is_file():
        print(f"{file_name} already exists, skipping download")
    else:
        print(f"Downloading {file_name}")
        request = requests.get(
            f"https://raw.githubusercontent.com/ivankreso/fer-deep-learning/master/lab2/{file_name}"
        )
        with open(path, "wb") as f:
            f.write(request.content)


def download_files():
    Path(BASE_PATH).mkdir(parents=True, exist_ok=True)
    for file in files:
        download_file(file, f"{BASE_PATH}\\{file}")


if __name__ == '__main__':
    download_files()
