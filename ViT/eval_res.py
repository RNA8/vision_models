#!/usr/bin/env python3
"""FixRes-style test-resolution sweep (no training): evaluate a finished finetune.py run at several test
resolutions (CIFAR images upsampled on the GPU). usage: eval_res.py runs/<name> 224 256 288 [--out f.json]"""
import argparse, json, os, sys
from argparse import Namespace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from finetune import load_model, evaluate
from train_vit import load_data

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("sizes", type=int, nargs="+")
ap.add_argument("--dataset", default="cifar100"); ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = "cuda"
d = os.path.dirname(os.path.abspath(__file__))
_, _, xte, yte, _ = load_data(Namespace(dataset=a.dataset, data_dir=os.path.join(d, "data"), fake_data=False,
                                        train_subset=0, test_subset=0), dev)
amp = lambda: torch.autocast(device_type=dev, dtype=torch.bfloat16)
res = {}
model, spec = load_model(a.run)  # uses EMA weights when they beat the raw ones (same rule as KD teachers)
model = model.to(dev).eval()
mean = torch.tensor(spec["mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(spec["std"], device=dev).view(1, 3, 1, 1)
for s in a.sizes:
    res[s] = evaluate(model, xte, yte, s, mean, std, amp)
    print(f"{a.run} test@{s}: {res[s]:.2f}", flush=True)
out = a.out or os.path.join(a.run, "eval_res.json")
json.dump({"run": a.run, "weights": "ema-if-present", "acc_by_size": res}, open(out + ".tmp", "w"), indent=1)
os.replace(out + ".tmp", out)
