"""CIFAR-10 (50k train / 10k test, 32x32, 10 classes) from the Hugging Face mirror.
The Toronto server can crawl at ~100 KB/s. PNGs are decoded once and cached as a ~180 MB uint8 .npz."""
import os

from .common import cached_npz, download, parquet_split

NUM_CLASSES = 10
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"


def load(data_dir):
    def build():
        arrays = {}
        for split in ("train", "test"):
            path = download(URL.format(split), os.path.join(data_dir, f"cifar10_{split}.parquet"))
            arrays[f"x{split}"], arrays[f"y{split}"] = parquet_split([path], "img", "label")
        return arrays

    return cached_npz(os.path.join(data_dir, "cifar10_uint8.npz"), build)
