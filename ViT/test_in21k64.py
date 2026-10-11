"""Tests for vit_data/in21k64.py (needs data/in21k64_shards). Run from ViT/:  ../.venv/bin/python test_in21k64.py
Uses 6 train shards (IN21K64_TRAIN_SHARDS) so it takes ~1 min."""
import json
import os

os.environ.setdefault("IN21K64_TRAIN_SHARDS", "6")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from vit_data import in21k64 as d  # noqa: E402

DATA = os.environ.get("VIT_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
dev = "cuda"


def test_val_subset():
    x, y = d.load_val(DATA)
    x2, y2 = d.load_val(DATA)
    assert x.shape == (d.N_VAL_SUBSET, 64, 64, 3) and x.dtype == np.uint8 and y.dtype == np.int64
    assert (x == x2).all() and (y == y2).all(), "val subset must be deterministic"
    assert 0 <= y.min() and y.max() < d.NUM_CLASSES and len(np.unique(y)) > 10000
    print("val subset ok:", x.shape, "classes", len(np.unique(y)))


def test_labels_vs_classes_and_ids():
    syn = d.class_names(DATA)
    assert len(syn) == d.NUM_CLASSES
    p = d.list_shards(DATA, "validation")[0]
    y, ids = np.load(p + ".y.npy"), np.load(p + ".ids.npy")
    for i in [0, 1, 17, 5000, len(y) - 1]:  # image_id is '<synset>_<n>': must agree with classes.json[label]
        assert ids[i].split("_")[0] == syn[y[i]], (ids[i], syn[y[i]])
    x = np.load(p + ".x.npy", mmap_mode="r")
    assert x.shape[0] == len(y) and x[0].std() > 5, "image looks blank"
    print("label/synset/id spot checks ok")


def test_stream():
    st = d.Stream(DATA, dev, seed=0, prefetch=3)
    assert st.xte.shape == (d.N_VAL_SUBSET, 3, 64, 64) and st.xte.dtype == torch.uint8
    assert st.xclean.shape[0] == d.N_CLEAN and st.yte.max() < d.NUM_CLASSES
    bs = 256
    nb = st.steps(bs)
    seen = 0
    first = []
    for i, (x, y) in enumerate(st.epoch(0, bs)):
        assert x.shape == (bs, 3, 64, 64) and x.dtype == torch.uint8 and x.is_cuda
        assert y.shape == (bs,) and y.dtype == torch.int64 and 0 <= y.min() and y.max() < d.NUM_CLASSES
        if i < 3:
            first.append((x[:4].cpu().clone(), y[:4].cpu().clone()))
        seen += 1
    assert seen == nb, (seen, nb)
    # every sample is seen once per epoch: the multiset of labels over an epoch is (nearly) the dataset's
    lab = np.bincount(np.concatenate(st.labels), minlength=d.NUM_CLASSES)
    # spot check image/label pairing: the first streamed batch rows must exist in the shards with the same label
    x0, y0 = first[0]
    found = 0
    for p, ys in zip(st.shards, st.labels):
        xs = np.load(p + ".x.npy", mmap_mode="r")
        for j in range(4):
            cand = np.nonzero(ys == y0[j].item())[0]
            for c in cand:
                if (xs[c] == x0[j].permute(1, 2, 0).numpy()).all():
                    found += 1
                    break
    assert found >= 4, f"streamed images not found with matching labels ({found}/4)"
    # determinism per (seed, epoch), and reshuffle across epochs
    it = st.epoch(0, bs)
    x_a, y_a = next(it)
    it.close()
    assert torch.equal(y_a[:4].cpu(), first[0][1]) and torch.equal(x_a[:4].cpu(), first[0][0]), "epoch not deterministic"
    it = st.epoch(1, bs)
    _, y_b = next(it)
    it.close()
    assert not torch.equal(y_a[:16].cpu(), y_b[:16].cpu())
    print(f"stream ok: {seen} batches x {bs}, labels spot-check {found}/4, total images {st.n}, bincount sum {lab.sum()}")


if __name__ == "__main__":
    for t in (test_val_subset, test_labels_vs_classes_and_ids, test_stream):
        t()
    print("ALL OK")
