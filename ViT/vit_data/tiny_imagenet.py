"""Tiny-ImageNet (100k train / 10k val, 64x64, 200 classes) from the Hugging Face mirror zh-plus/tiny-imagenet.
The official test split has no labels, so the labelled 'valid' split (50 img/class) is used as the test set.
Some images are grayscale; they are converted to RGB. Decoded once (multiprocessing) and cached as a uint8 .npz."""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .common import cached_npz, decode_images, download

NUM_CLASSES = 200
MEAN = (0.4802, 0.4481, 0.3975)  # computed on the train set
STD = (0.2764, 0.2689, 0.2816)
BASE = "https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main/data/"
FILES = {
    "train": "train-00000-of-00001-1359597a978bc4fa.parquet",
    "valid": "valid-00000-of-00001-70d52db3c749a935.parquet",
}


def _decode_chunk(cells):
    return decode_images(cells)


def _read_split(path):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    t = pf.read(columns=["image", "label"]).to_pydict()
    cells, y = t["image"], np.asarray(t["label"], dtype=np.int64)
    chunks = [cells[i:i + 2000] for i in range(0, len(cells), 2000)]
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as ex:
        x = np.concatenate(list(ex.map(_decode_chunk, chunks)))
    return x, y


def load(data_dir):
    def build():
        arrays = {}
        for split, fname in FILES.items():
            path = download(BASE + fname, os.path.join(data_dir, f"tiny_imagenet_{split}.parquet"))
            key = "test" if split == "valid" else "train"
            arrays[f"x{key}"], arrays[f"y{key}"] = _read_split(path)
        # the parquet files carry no class names, so check the label space is the expected one
        for k in ("train", "test"):
            assert set(np.unique(arrays[f"y{k}"])) == set(range(NUM_CLASSES))
        return arrays

    return cached_npz(os.path.join(data_dir, "tiny_imagenet_uint8.npz"), build)
