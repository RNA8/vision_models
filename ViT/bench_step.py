#!/usr/bin/env python3
"""Benchmark one training step of train_vit.py's pipeline under different speed settings.

  ../.venv/bin/python bench_step.py --batch-size 256 --model tiny
Prints ms/step and img/s for: eager aug + compiled model (current), compiled aug, and
compiled aug + CUDA-graphed model (torch.compile mode="max-autotune-no-cudagraphs" / "reduce-overhead").
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

import train_vit as T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--model", default="tiny")
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--variants", nargs="+", default=["baseline", "compile_aug", "compile_aug_cg"])
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev, bs = "cuda", a.batch_size
    args = argparse.Namespace(model=a.model, patch_size=a.patch_size, drop_path=0.1, mixup=0.8, cutmix=1.0,
                              mix_prob=1.0, lr=1e-3, weight_decay=0.05, r18_stem="cifar")
    xu = torch.randint(0, 256, (50000, 3, 32, 32), dtype=torch.uint8, device=dev)
    yu = torch.randint(0, 10, (50000,), device=dev)
    mean = torch.tensor((0.49, 0.48, 0.45), device=dev).view(1, 3, 1, 1)
    std = torch.tensor((0.25, 0.24, 0.26), device=dev).view(1, 3, 1, 1)

    def aug(xb):
        x = xb.float().div_(255.0)
        x = T.random_crop_flip(x, pad=4)
        x = T.rand_augment(x, 2, 0.9)
        return (x - mean) / std

    for variant in a.variants:
        torch._dynamo.reset()
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        model = T.build_model(args, 10, 32).to(dev)
        mode = "reduce-overhead" if variant.endswith("_cg") else None
        cmodel = torch.compile(model, mode=mode)
        opt = T.make_optimizer(model, args, fused=True)
        aug_fn = aug if variant.startswith("baseline") else torch.compile(aug)

        def step():
            idx = torch.randint(0, 50000, (bs,), device=dev)
            x = aug_fn(xu[idx])
            t = T.smooth_one_hot(yu[idx], 10, 0.1)
            x, t = T.mix_batch(x, t, args, rng)
            with torch.autocast("cuda", torch.bfloat16):
                logits = cmodel(x)
            loss = torch.sum(-t * F.log_softmax(logits.float(), dim=-1), dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        for _ in range(10):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(a.steps):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / a.steps
        print(f"{variant:16s} bs {bs:5d}  {dt * 1000:7.2f} ms/step  {bs / dt:9.0f} img/s  "
              f"-> {50000 // bs * dt:5.2f} s/epoch", flush=True)


if __name__ == "__main__":
    main()
