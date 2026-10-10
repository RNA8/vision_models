"""Shared helpers for dataset modules.

Every dataset module in vit_data/ exposes:
    NUM_CLASSES: int
    MEAN, STD:   per-channel tuples (RGB, in [0, 1] units)
    load(data_dir) -> (xtrain, ytrain, xtest, ytest)
        x*: uint8 torch tensors, shape (N, H, W, 3)   (NHWC; train_vit.py moves them to the GPU)
        y*: int64 torch tensors, shape (N,)
train_vit.py picks a module with --dataset <module name>; image size comes from the data.
"""
import io
import os
import urllib.request

import numpy as np
import torch


def download(url, path, headers=None):
    """Download `url` to `path` atomically (via a .part file); skip if it already exists."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url} -> {path}", flush=True)
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    os.replace(path + ".part", path)
    return path


def decode_images(image_cells, size=None):
    """Decode a list of HF image cells ({'bytes': ...}) to a uint8 (N, H, W, 3) array.
    Converts grayscale/CMYK to RGB; resizes to `size` x `size` (bicubic) if given."""
    from PIL import Image

    out = None
    for i, cell in enumerate(image_cells):
        im = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        if size is not None and im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        a = np.asarray(im)
        if out is None:
            out = np.empty((len(image_cells),) + a.shape, dtype=np.uint8)
        out[i] = a
    return out


def parquet_split(paths, image_col, label_col, size=None):
    """Read one split stored as one or more parquet files -> (uint8 NHWC array, int64 labels)."""
    import pyarrow.parquet as pq

    xs, ys = [], []
    for p in paths:
        t = pq.read_table(p, columns=[image_col, label_col]).to_pydict()
        xs.append(decode_images(t[image_col], size))
        ys.append(np.asarray(t[label_col], dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def cached_npz(cache_path, build):
    """Return tensors from `cache_path`, calling build() -> dict(xtrain, ytrain, xtest, ytest)
    of numpy arrays to create it on first use."""
    if not os.path.exists(cache_path):
        arrays = build()
        np.savez(cache_path + ".part.npz", **arrays)
        os.replace(cache_path + ".part.npz", cache_path)
    d = np.load(cache_path)
    return tuple(torch.from_numpy(d[k]) for k in ("xtrain", "ytrain", "xtest", "ytest"))
