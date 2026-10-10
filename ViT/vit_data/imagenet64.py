"""ImageNet-1k at 64x64 (1,281,167 train / 50,000 official val used as test, 1000 classes).

SOURCE: the official gated HF dataset `ILSVRC/imagenet-1k` (full-res JPEG parquet shards: 294 train + 14 val,
~150 GB). No legitimate pre-downsampled 64x64 copy exists on HF behind the ImageNet terms (the 64x64 ones
are ungated re-uploads, which we do not use; image-net.org's Downsampled-ImageNet also needs a login), so we
download the shards and downsample ourselves. Each shard is processed as soon as it is downloaded, written
as a small per-shard npz (resumable), the raw shard is deleted, and finally everything is assembled into
preallocated uint8 arrays and cached to data/imagenet64_uint8.npz (~16 GB).

Resize: whole image squashed to 64x64 (no crop, like the official Downsampled-ImageNet), using
PIL JPEG draft mode (DCT-domain downscale to >=2x target, fast) followed by a BOX (area) filter, which is
alias-free for downscaling and is what Chrabaszcz et al. used for the official 64x64 variant.

ACCESS: needs accepted terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k and a token
(HF_TOKEN env var or `hf auth login`). Prepare ahead of training with (from ViT/):
    ../.venv/bin/python -m vit_data.imagenet64 --prepare
"""
import argparse
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

import numpy as np

NUM_CLASSES = 1000
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 64
REPO = "ILSVRC/imagenet-1k"
SPLITS = {"train": ("train", 294), "test": ("validation", 14)}  # our name -> (hf prefix, n shards)
N_TRAIN, N_TEST = 1281167, 50000


def get_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    p = os.path.join(home, "token")
    return open(p).read().strip() if os.path.exists(p) else None


def shard_name(split, i):
    prefix, n = SPLITS[split]
    return f"data/{prefix}-{i:05d}-of-{n:05d}.parquet"


def _resize(b):
    from PIL import Image

    im = Image.open(io.BytesIO(b))
    if im.format == "JPEG":
        im.draft("RGB", (2 * SIZE, 2 * SIZE))  # never below 2x target
    im = im.convert("RGB").resize((SIZE, SIZE), Image.BOX)
    return np.asarray(im)


def process_shard(args):
    """Decode one raw parquet shard -> small npz (x uint8 NHWC, y int64). Runs in a Pool worker."""
    raw, out = args
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(raw)
    xs, ys = [], []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["image", "label"]).to_pydict()
        xs.append(np.stack([_resize(c["bytes"]) for c in t["image"]]))
        ys.append(np.asarray(t["label"], dtype=np.int64))
    np.savez(out + ".part.npz", x=np.concatenate(xs), y=np.concatenate(ys))
    os.replace(out + ".part.npz", out)
    os.remove(raw)
    return out


def prepare_shards(data_dir, workers=26, dl_threads=8):
    """Download (threads) + decode (process pool) every shard; skips shards already processed."""
    from huggingface_hub import hf_hub_download

    token = get_token()
    if token is None:
        sys.exit(
            "No Hugging Face token found. ImageNet-1k is gated: accept the terms at "
            "https://huggingface.co/datasets/ILSVRC/imagenet-1k then run `.venv/bin/hf auth login` "
            "(or export HF_TOKEN=...)."
        )
    raw_dir = os.path.join(data_dir, "imagenet64_raw")
    shard_dir = os.path.join(data_dir, "imagenet64_shards")
    os.makedirs(shard_dir, exist_ok=True)
    jobs = []  # (split, i, filename, out)
    for split, (prefix, n) in SPLITS.items():
        for i in range(n):
            out = os.path.join(shard_dir, f"{prefix}-{i:05d}.npz")
            if not os.path.exists(out):
                jobs.append((split, i, shard_name(split, i), out))
    print(f"{len(jobs)} shards to do", flush=True)
    if not jobs:
        return shard_dir

    def fetch(job):
        for attempt in range(5):
            try:
                return job, hf_hub_download(REPO, job[2], repo_type="dataset", token=token, local_dir=raw_dir)
            except Exception as e:  # noqa: BLE001
                if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
                    sys.exit(f"Access denied for {REPO} ({e}). Accept the terms on the dataset page and log in.")
                print(f"retry {job[2]} ({attempt}): {e}", flush=True)
        raise RuntimeError(f"failed to download {job[2]}")

    done = 0
    with Pool(workers) as pool, ThreadPoolExecutor(dl_threads) as tp:
        pending = []
        futs = [tp.submit(fetch, j) for j in jobs]
        for f in as_completed(futs):
            job, raw = f.result()
            pending.append(pool.apply_async(process_shard, ((raw, job[3]),)))
            # reap finished decodes (re-raises worker errors) and bound the backlog of raw shards on disk
            while len([p for p in pending if not p.ready()]) > workers + 4:
                pending[0].wait(1)
            done_now = [p for p in pending if p.ready()]
            for p in done_now:
                p.get()
                pending.remove(p)
                done += 1
            print(f"downloaded {job[2]}; decoded {done}/{len(jobs)}", flush=True)
        for p in pending:
            p.get()
            done += 1
            print(f"decoded {done}/{len(jobs)}", flush=True)
    return shard_dir


def build(data_dir):
    shard_dir = prepare_shards(data_dir)
    arrays = {}
    for split, (prefix, n) in SPLITS.items():
        total = N_TRAIN if split == "train" else N_TEST
        x = np.empty((total, SIZE, SIZE, 3), np.uint8)
        y = np.empty(total, np.int64)
        k = 0
        for i in range(n):
            d = np.load(os.path.join(shard_dir, f"{prefix}-{i:05d}.npz"))
            m = len(d["y"])
            x[k:k + m], y[k:k + m] = d["x"], d["y"]
            k += m
        assert k == total, f"{split}: got {k} images, expected {total}"
        assert 0 <= y.min() and y.max() < NUM_CLASSES
        arrays[f"x{split}"], arrays[f"y{split}"] = x, y
    return arrays


def load(data_dir):
    from .common import cached_npz

    return cached_npz(os.path.join(data_dir, "imagenet64_uint8.npz"), lambda: build(data_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="download + decode + build the npz cache")
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    a = ap.parse_args()
    os.makedirs(a.data_dir, exist_ok=True)
    from vit_data.common import cached_npz

    path = os.path.join(a.data_dir, "imagenet64_uint8.npz")
    cached_npz(path, lambda: build(a.data_dir))
    print("done:", path)
