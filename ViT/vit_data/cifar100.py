"""CIFAR-100 (50k train / 10k test, 32x32, 100 fine classes) from the Hugging Face mirror.
PNGs are decoded once and cached as a ~180 MB uint8 .npz. Uses the `fine_label` column."""
import os

from .common import cached_npz, download, parquet_split

NUM_CLASSES = 100
MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)
URL = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/{}-00000-of-00001.parquet"


def load(data_dir):
    def build():
        arrays = {}
        for split in ("train", "test"):
            path = download(URL.format(split), os.path.join(data_dir, f"cifar100_{split}.parquet"))
            arrays[f"x{split}"], arrays[f"y{split}"] = parquet_split([path], "img", "fine_label")
        return arrays

    return cached_npz(os.path.join(data_dir, "cifar100_uint8.npz"), build)
