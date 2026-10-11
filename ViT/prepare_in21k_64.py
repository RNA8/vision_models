"""ImageNet-21K-P (winter21) at 64x64, uint8 shards, for pretraining small ViTs.

SOURCE: the official gated HF dataset `timm/imagenet-w21-p` (accept the ImageNet terms at
https://huggingface.co/datasets/timm/imagenet-w21-p; token from `hf auth login` or HF_TOKEN). It is the
ImageNet winter21 release filtered with the ImageNet-21K-P recipe (Ridnik et al. 2021,
https://github.com/Alibaba-MIIL/ImageNet21K): 10,450 classes, 11,060,239 train + 522,500 validation
images (50/class), stored as 961 + 44 parquet shards (~497 GB total, images re-encoded as WEBP).
Columns: image, label (0..10449, synset names in the parquet HF metadata), label_tree, semantic_labels
(11 hierarchy levels, -1 padded; for semantic-softmax training), image_id.
We picked -P over the full winter21 (`timm/imagenet-w21-wds`, 19,167 classes, 1.17 TB tar) because it is
the standard 21k pretraining set, half the download, and ships an official held-out split.

HELD-OUT SPLIT: the dataset's own `validation` split (50 imgs/class, disjoint from train) is kept as
`validation-*`. No extra hold-out is carved from train.

RESIZE: identical to our ImageNet-64 (vit_data/imagenet64.py): whole image squashed to 64x64 (no crop,
like Chrabaszcz et al. Downsampled-ImageNet) with a PIL BOX (area) filter. JPEG inputs use draft mode to
>=2x target first; these shards are WEBP (no draft), so BOX runs on the full-res image (exact area mean).

OUTPUT (default ViT/data/in21k64_shards/), per source shard `{split}-{i:05d}`:
    .x.npy    uint8  (N, 64, 64, 3)  NHWC RGB
    .y.npy    int32  (N,)            label in [0, 10450)
    .sem.npy  int16  (N, 11)         semantic_labels (hierarchy, -1 = padding)
    .ids.npy  str                    image_id (e.g. n04497570_11144)
written atomically (x last; a shard is "done" iff its .x.npy exists), plus classes.json (synset per label)
and, at the end, manifest.json (per-shard counts). Final size ~12.3 KB/img -> ~142 GB.
Each raw parquet shard is deleted right after decoding; at most --max-inflight raw shards (~0.5 GB each)
are on disk at any time.

RUN / RESUME (from repo root; rerunning the same command skips finished shards and retries failed ones):
    source ViT/env.sh
    .venv/bin/python ViT/prepare_in21k_64.py                    # full run, 14 decode workers
    .venv/bin/python ViT/prepare_in21k_64.py --limit 2          # quick test on 2 shards
"""
import argparse
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np

REPO = "timm/imagenet-w21-p"
SIZE = 64
NUM_CLASSES = 10450
SPLITS = [("validation", 44), ("train", 961)]  # validation first so the eval set is ready early
N_EXPECTED = {"train": 11060239, "validation": 522500}
HERE = os.path.dirname(os.path.abspath(__file__))


class AccessDenied(Exception):
    pass


def _resize(b):
    from PIL import Image

    im = Image.open(io.BytesIO(b))
    if im.format == "JPEG":
        im.draft("RGB", (2 * SIZE, 2 * SIZE))
    im = im.convert("RGB").resize((SIZE, SIZE), Image.BOX)
    return np.asarray(im)


def _save(path, arr):
    tmp = path[:-4] + ".part.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def process_shard(raw, out):
    """Decode one raw parquet shard -> per-shard .npy files; deletes the raw shard. Runs in a worker process."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(raw)
    xs, ys, sems, ids = [], [], [], []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["image", "label", "semantic_labels", "image_id"]).to_pydict()
        xs.append(np.stack([_resize(c["bytes"]) for c in t["image"]]))
        ys.append(np.asarray(t["label"], dtype=np.int32))
        sems.append(np.asarray(t["semantic_labels"], dtype=np.int16))
        ids.extend(t["image_id"])
    x, y = np.concatenate(xs), np.concatenate(ys)
    assert 0 <= y.min() and y.max() < NUM_CLASSES
    _save(out + ".y.npy", y)
    _save(out + ".sem.npy", np.concatenate(sems))
    _save(out + ".ids.npy", np.asarray(ids))
    _save(out + ".x.npy", x)  # last: marks the shard done
    os.remove(raw)
    return len(y)


def _worker_init():
    os.nice(10)  # leave CPU priority to the training jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--workers", type=int, default=14, help="decode processes")
    ap.add_argument("--dl-threads", type=int, default=6, help="concurrent shard downloads")
    ap.add_argument("--max-inflight", type=int, default=40, help="max raw shards on disk (~0.5 GB each)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N pending shards (testing)")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    raw_dir = os.path.join(a.data_dir, "in21k_raw")
    out_dir = os.path.join(a.data_dir, "in21k64_shards")
    os.makedirs(out_dir, exist_ok=True)

    jobs = []  # (hf filename, out prefix)
    for prefix, n in SPLITS:
        for i in range(n):
            out = os.path.join(out_dir, f"{prefix}-{i:05d}")
            if not os.path.exists(out + ".x.npy"):
                jobs.append((f"data/{prefix}-{i:05d}-of-{n:05d}.parquet", out))
    total_jobs = len(jobs)
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"[{time.strftime('%F %T')}] {total_jobs} shards pending, doing {len(jobs)}; "
          f"workers={a.workers} dl_threads={a.dl_threads} max_inflight={a.max_inflight}", flush=True)

    def fetch(fname):
        for attempt in range(10):
            try:
                return hf_hub_download(REPO, fname, repo_type="dataset", local_dir=raw_dir)
            except (GatedRepoError, RepositoryNotFoundError) as e:
                raise AccessDenied(f"Access denied for {REPO}: accept the terms at "
                                   f"https://huggingface.co/datasets/{REPO} and `hf auth login`. ({e})")
            except Exception as e:  # noqa: BLE001 - network errors: back off and retry
                wait = min(300, 10 * 2 ** attempt)
                print(f"retry {fname} in {wait}s (attempt {attempt}): {type(e).__name__}: {str(e)[:200]}", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"failed to download {fname}")

    dl_sem = threading.Semaphore(a.dl_threads)
    stop = threading.Event()

    with ProcessPoolExecutor(a.workers, initializer=_worker_init) as pp:
        def pipeline(job):  # one thread per in-flight shard: download (bounded) -> decode in the process pool
            fname, out = job
            if stop.is_set():
                return job, 0
            with dl_sem:
                raw = fetch(fname)
            return job, pp.submit(process_shard, raw, out).result()

        t0, n_img, n_done, failed = time.time(), 0, 0, []
        with ThreadPoolExecutor(a.max_inflight) as tp:
            futs = [tp.submit(pipeline, j) for j in jobs]
            for f in as_completed(futs):
                try:
                    job, m = f.result()
                except AccessDenied as e:
                    print(str(e), flush=True)
                    stop.set()
                    for g in futs:
                        g.cancel()
                    os._exit(2)
                except Exception as e:  # noqa: BLE001 - keep going; a rerun retries this shard
                    failed.append(str(e)[:300])
                    print(f"!! shard failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
                    continue
                n_done += 1
                n_img += m
                dt = time.time() - t0
                rate = n_done / dt
                eta = (len(jobs) - n_done) / rate if rate else 0
                print(f"[{time.strftime('%F %T')}] {os.path.basename(job[1])} ({m} imgs) | {n_done}/{len(jobs)} shards, "
                      f"{n_img} imgs, {n_img / dt:.0f} img/s, {rate * 3600:.0f} shards/h, ETA {eta / 3600:.2f} h", flush=True)

    # class index (synset per label) from the parquet's HF metadata of any remaining raw file or via the hub
    cls_path = os.path.join(out_dir, "classes.json")
    if not os.path.exists(cls_path):
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq

        fs = HfFileSystem()
        with fs.open(f"datasets/{REPO}/data/validation-00000-of-00044.parquet", "rb") as fh:
            md = json.loads(pq.ParquetFile(fh).schema_arrow.metadata[b"huggingface"])
        names = md["info"]["features"]["label"]["names"]
        assert len(names) == NUM_CLASSES
        with open(cls_path + ".part", "w") as fh:
            json.dump({"source": REPO, "num_classes": NUM_CLASSES, "synsets": names}, fh)
        os.replace(cls_path + ".part", cls_path)

    # manifest once everything is done
    shards = {p: [f"{p}-{i:05d}" for i in range(n)] for p, n in SPLITS}
    if all(os.path.exists(os.path.join(out_dir, s + ".x.npy")) for v in shards.values() for s in v):
        man = {"source": REPO, "size": SIZE, "num_classes": NUM_CLASSES, "splits": {}}
        for p, names in shards.items():
            counts = [len(np.load(os.path.join(out_dir, s + ".y.npy"), mmap_mode="r")) for s in names]
            man["splits"][p] = {"shards": names, "counts": counts, "total": sum(counts)}
            ok = "OK" if sum(counts) == N_EXPECTED[p] else f"MISMATCH (expected {N_EXPECTED[p]})"
            print(f"{p}: {sum(counts)} images {ok}", flush=True)
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(man, fh)
        print("ALL DONE", flush=True)
    if failed:
        print(f"{len(failed)} shards failed; rerun the same command to retry them.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
