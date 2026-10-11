"""ImageNet-21K-P (winter21) at 64x64: 10,450 classes, 11,060,239 train / 522,500 val images (~136 GB uint8).

Prepared by prepare_in21k_64.py into data/in21k64_shards/{train,validation}-{i:05d}.{x,y,sem,ids}.npy.
Unlike the other vit_data modules the train set does NOT fit on the GPU, so this module is STREAMING:
`STREAMING = True` and `Stream(data_dir, device, seed)` replaces `load()` (train_vit.py checks the flag).

Design
  * Shards are np.load(mmap_mode='r'): nothing is read eagerly; the OS page cache keeps the hot set in RAM
    (the whole set fits in ~140 GB of cache on the 221 GB box, so epochs after the first are RAM-speed;
    the first epoch is read from disk). Process RSS stays at a few GB.
  * Per epoch: the shard order is shuffled (seed, epoch), consecutive WINDOW shards (default 4, ~46k images,
    ~570 MB) are copied from the mmaps into one RAM array (a thread, one window ahead) and shuffled as one block.
    Shuffle quality: shards are already a random mix of the data (each holds ~6.8k of the 10,450 classes in random
    order, checked on shard 0), so a batch is a random draw from ~46k images of ~4 shards instead of from all 11M.
    Within an epoch every image is seen exactly once (a few < batch-size remainders carry into the next window);
    the sample order across epochs is fully re-randomised by the shard permutation. This deviates from i.i.d.
    sampling only by correlation between batches of the same window (same shards), which is negligible for SGD
    at batch 256 with 240 batches/window.
  * A second thread gathers each batch (np.take into a ring of pinned buffers), copies it to the GPU on a side
    stream (non_blocking), permutes NHWC->NCHW and hands over a uint8 CUDA tensor through a small queue. The
    train loop then runs its normal GPU augmentation (crop/flip/RandAugment/mix) on these tensors, exactly as for
    GPU-resident datasets; torch.compile/CUDA graphs copy from them into their static input buffers as usual.
  * Resume: the iterator is rebuilt from (seed, epoch) at every epoch start, so a restart simply restarts that epoch.
  * Validation: a fixed, deterministic 50,000-image subset of the 522,500 val images (np.random.RandomState(0),
    sorted indices), GPU-resident (614 MB). A 10,000-image subset of train (RandomState(1)) is also GPU-resident
    for the "clean train accuracy" metric.
  * IN21K64_TRAIN_SHARDS=<n> env var uses only the first n train shards (smoke tests); IN21K64_VAL_SHARDS likewise.
"""
import json
import os
import queue
import threading
import time

import numpy as np
import torch

NUM_CLASSES = 10450
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 64
STREAMING = True
N_VAL_SUBSET = 50000
N_CLEAN = 10000
WINDOW = int(os.environ.get("IN21K64_WINDOW", 4))


def shard_dir(data_dir):
    return os.path.join(data_dir, "in21k64_shards")


def list_shards(data_dir, split):
    d = shard_dir(data_dir)
    n = sum(1 for f in os.listdir(d) if f.startswith(split + "-") and f.endswith(".x.npy"))
    lim = os.environ.get("IN21K64_TRAIN_SHARDS" if split == "train" else "IN21K64_VAL_SHARDS")
    if lim:
        n = min(n, int(lim))
    return [os.path.join(d, f"{split}-{i:05d}") for i in range(n)]


def _gather(prefixes, counts, idx):
    """Rows `idx` (global, sorted) of the concatenation of the shards -> (uint8 NHWC array, int labels)."""
    starts = np.concatenate([[0], np.cumsum(counts)])
    xs, ys = [], []
    for i, p in enumerate(prefixes):
        sel = idx[(idx >= starts[i]) & (idx < starts[i + 1])] - starts[i]
        if len(sel):
            xs.append(np.load(p + ".x.npy", mmap_mode="r")[sel])
            ys.append(np.load(p + ".y.npy")[sel])
    return np.concatenate(xs), np.concatenate(ys).astype(np.int64)


def load_val(data_dir, n=N_VAL_SUBSET):
    """Deterministic validation subset: (uint8 NHWC, int64 labels) as numpy arrays."""
    pre = list_shards(data_dir, "validation")
    counts = [len(np.load(p + ".y.npy")) for p in pre]
    total = sum(counts)
    idx = np.sort(np.random.RandomState(0).choice(total, min(n, total), replace=False))
    return _gather(pre, counts, idx)


def class_names(data_dir):
    with open(os.path.join(shard_dir(data_dir), "classes.json")) as f:
        return json.load(f)["synsets"]


class Stream:
    """Streaming train set + GPU-resident val subset. See the module docstring."""

    def __init__(self, data_dir, device, seed=0, prefetch=6, window=WINDOW):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.wait_s = 0.0  # time the consumer (train loop) spent blocked on the loader; ~0 means GPU-bound
        self.device, self.seed, self.prefetch, self.window = device, seed, prefetch, window
        self.shards = list_shards(data_dir, "train")
        self.counts = [len(np.load(p + ".y.npy")) for p in self.shards]
        self.labels = [np.load(p + ".y.npy") for p in self.shards]  # 4 bytes/img, ~44 MB in total
        self.n = int(sum(self.counts))
        xv, yv = load_val(data_dir)
        xc, yc = _gather(self.shards, self.counts,
                         np.sort(np.random.RandomState(1).choice(self.n, min(N_CLEAN, self.n), replace=False)))
        to_dev = lambda x: torch.from_numpy(x).to(device).permute(0, 3, 1, 2).contiguous()  # noqa: E731
        self.xte, self.yte = to_dev(xv), torch.from_numpy(yv).to(device)
        self.xclean, self.yclean = to_dev(xc), torch.from_numpy(yc).to(device)

    # ---- host side: windows of shards, block-shuffled
    def _windows(self, epoch, stop):
        order = np.random.RandomState([self.seed, epoch]).permutation(len(self.shards))
        for w in range(0, len(order), self.window):
            if stop.is_set():
                return
            ids = order[w : w + self.window]
            n = sum(self.counts[i] for i in ids)
            x = np.empty((n, SIZE, SIZE, 3), np.uint8)
            k = 0
            for i in ids:
                m = self.counts[i]
                x[k : k + m] = np.load(self.shards[i] + ".x.npy", mmap_mode="r")
                k += m
            y = np.concatenate([self.labels[i] for i in ids]).astype(np.int64)
            perm = np.random.RandomState([self.seed, epoch, w]).permutation(n)
            yield x, y, perm

    def steps(self, bs):
        return self.n // bs

    def epoch(self, epoch, bs):
        """Iterator over exactly n // bs batches (uint8 NCHW cuda, int64 cuda), reshuffled per epoch."""
        n_steps = self.steps(bs)
        stop = threading.Event()
        wq = queue.Queue(maxsize=1)  # window prefetch: one finished window waiting while one is consumed
        bq = queue.Queue(maxsize=self.prefetch)
        err = []

        def put(q, item):
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    pass
            return False

        def window_thread():
            try:
                for win in self._windows(epoch, stop):
                    if not put(wq, win):
                        return
                put(wq, None)
            except BaseException as e:  # noqa: BLE001
                err.append(e)
                put(wq, None)

        def batch_thread():
            try:
                torch.cuda.set_device(self.device)
                stream = torch.cuda.Stream(self.device)
                ring = [torch.empty((bs, SIZE, SIZE, 3), dtype=torch.uint8).pin_memory() for _ in range(self.prefetch + 2)]
                ring_y = [torch.empty(bs, dtype=torch.int64).pin_memory() for _ in range(self.prefetch + 2)]
                r = 0
                cur, pos = None, 0
                for _ in range(n_steps):
                    bx, by = ring[r].numpy(), ring_y[r].numpy()
                    got = 0
                    while got < bs:  # a batch may straddle two windows
                        if cur is None or pos >= len(cur[2]):
                            cur, pos = None, 0
                            while cur is None and not stop.is_set():
                                try:
                                    cur = wq.get(timeout=0.5)
                                except queue.Empty:
                                    continue
                                if cur is None:  # window thread finished / failed
                                    return
                            if cur is None:
                                return
                        x, y, perm = cur
                        take = min(bs - got, len(perm) - pos)
                        sel = np.sort(perm[pos : pos + take])  # sorted -> sequential memory access
                        np.take(x, sel, axis=0, out=bx[got : got + take])
                        np.take(y, sel, out=by[got : got + take])
                        pos += take
                        got += take
                    with torch.cuda.stream(stream):
                        gx = ring[r].to(self.device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
                        gy = ring_y[r].to(self.device, non_blocking=True)
                    stream.synchronize()  # ring slot is reusable after this; consumer sees finished tensors
                    if not put(bq, (gx, gy)):
                        return
                    r = (r + 1) % len(ring)
                put(bq, None)
            except BaseException as e:  # noqa: BLE001
                err.append(e)
                put(bq, None)

        threads = [threading.Thread(target=window_thread, daemon=True), threading.Thread(target=batch_thread, daemon=True)]
        for t in threads:
            t.start()
        cons = torch.cuda.current_stream(self.device)
        try:
            for _ in range(n_steps):
                t_w = time.time()
                item = bq.get()
                self.wait_s += time.time() - t_w
                if item is None:
                    raise RuntimeError(f"in21k64 loader stopped early: {err[0] if err else 'out of data'}")
                gx, gy = item
                gx.record_stream(cons)  # allocated on the loader's stream, used on ours
                gy.record_stream(cons)
                yield gx, gy
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=5)
