#!/usr/bin/env python3
"""
visualize.py - plots for a trained ViT run (CPU by default).

  1. {prefix}pos_embed_similarity.png  position-embedding cosine-similarity grid
  2. {prefix}cls_attention.png         CLS-to-patch attention per layer (head-averaged, recomputed from qkv)
  3. {prefix}acc_curves.png            train(clean) vs val accuracy + gap, all curve runs
  4. {prefix}loss_curves.png, {prefix}throughput.png

Usage (from ViT/, after `source env.sh`):
  ../.venv/bin/python visualize.py                       # auto-pick best ViT run, ablation ladder curves
  ../.venv/bin/python visualize.py --prefix c100_        # CIFAR-100 runs
  ../.venv/bin/python visualize.py --run t6_full_300ep --n-images 8
"""
import argparse
import argparse as _ap
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LADDER = ["t1_plain", "t2_ls_wd", "t3_randaug", "t4_mix", "t5_droppath", "t6_full_300ep", "r18_full"]
OTHER_PREFIXES = ["c100_", "tin_", "in64_"]
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# --------------------------------------------------------------------------- io
def load_results(runs_dir, name):
    p = os.path.join(runs_dir, name, "results.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: cannot read {p}: {e}")
        return None


def pick_run(runs_dir, prefix):
    best, best_acc = None, -1.0
    for name in sorted(os.listdir(runs_dir)):
        if name.startswith("smoke") or not name.startswith(prefix):
            continue
        if prefix == "" and any(name.startswith(o) for o in OTHER_PREFIXES):
            continue
        r = load_results(runs_dir, name)
        if not r or r.get("args", {}).get("model") == "resnet18":
            continue
        if not os.path.isfile(os.path.join(runs_dir, name, "model.pt")):
            continue
        acc = r.get("best_val_acc") or 0.0
        if acc > best_acc:
            best, best_acc = name, acc
    return best


def load_model_and_data(runs_dir, name, res, device):
    from train_vit import build_model
    import importlib

    a = res["args"]
    ds = importlib.import_module(f"vit_data.{a['dataset']}")
    _, _, xte, yte = ds.load(a.get("data_dir") or os.path.join(HERE, "data"))
    xte = xte.permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
    model = build_model(_ap.Namespace(**a), ds.NUM_CLASSES, xte.size(-1))
    sd = torch.load(os.path.join(runs_dir, name, "model.pt"), map_location="cpu")
    model.load_state_dict(sd)
    return model.to(device).eval().float(), xte, yte, ds


# --------------------------------------------------------------------------- 1. pos embed
def plot_pos_embed(model, title_extra, path):
    pe = model.pos_embed.detach().float()[0]
    pe = pe[model.num_prefix_tokens:]
    g = math.isqrt(pe.size(0))
    pn = F.normalize(pe, dim=-1)
    sim = (pn @ pn.T).reshape(g, g, g, g).cpu().numpy()
    cell = 0.9 if g <= 8 else max(0.45, 7.2 / g)
    fig, axes = plt.subplots(g, g, figsize=(g * cell + 1.2, g * cell + 1.0), squeeze=False)
    for r in range(g):
        for c in range(g):
            ax = axes[r, c]
            im = ax.imshow(sim[r, c], vmin=-1, vmax=1, cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.3)
    fig.suptitle(f"Position-embedding cosine similarity ({title_extra})\n"
                 f"cell (row, col) = similarity of that patch's embedding to every patch ({g}x{g} grid)",
                 fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.88, top=0.90, bottom=0.01, wspace=0.05, hspace=0.05)
    cax = fig.add_axes([0.91, 0.15, 0.025, 0.7])
    fig.colorbar(im, cax=cax, label="cosine similarity")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- 2. attention
def manual_attention(attn, x):
    """Recompute timm Attention: returns (output after proj, weights B,H,N,N)."""
    B, N, C = x.shape
    H = attn.num_heads
    qkv = attn.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    w = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    out = (w @ v).transpose(1, 2).reshape(B, N, C)
    return attn.proj(out), w


@torch.no_grad()
def cls_attention(model, xn, verify=True):
    """xn: normalized (B,3,H,W). Returns (logits, attn maps (L,B,g,g) head-averaged, max_diff)."""
    inputs, outputs = {}, {}
    hooks = []
    def make_hook(i):
        def hook(m, a, o):  # must return None so the output is not replaced
            inputs[i] = a[0].detach()
            outputs[i] = o.detach()
        return hook

    for i, blk in enumerate(model.blocks):
        hooks.append(blk.attn.register_forward_hook(make_hook(i)))
    try:
        logits = model(xn)
    finally:
        for h in hooks:
            h.remove()
    npre = model.num_prefix_tokens
    maps, max_diff = [], 0.0
    for i, blk in enumerate(model.blocks):
        out, w = manual_attention(blk.attn, inputs[i])
        if verify:
            max_diff = max(max_diff, (out - outputs[i]).abs().max().item())
        a = w[:, :, 0, npre:].mean(1)  # CLS row, patch columns, mean over heads
        g = math.isqrt(a.size(-1))
        maps.append(a.reshape(-1, g, g))
    return logits, torch.stack(maps), max_diff


def pick_indices(yte, n):
    """Fixed test indices with varied classes: first test image of each of n distinct classes."""
    seen, idx = set(), []
    for i, y in enumerate(yte.tolist()):
        if y not in seen:
            seen.add(y); idx.append(i)
        if len(idx) == n:
            break
    for i in range(len(yte)):  # fewer classes than n: top up
        if len(idx) >= n:
            break
        if i not in idx:
            idx.append(i)
    return idx


def plot_attention(model, xte, yte, ds, n_images, device, title_extra, path):
    idx = pick_indices(yte, n_images)
    imgs = xte[idx]
    mean = torch.tensor(ds.MEAN).view(1, 3, 1, 1)
    std = torch.tensor(ds.STD).view(1, 3, 1, 1)
    xn = ((imgs.float() / 255 - mean) / std).to(device)
    logits, maps, diff = cls_attention(model, xn)
    print(f"Attention recompute vs timm module output: max abs diff = {diff:.2e} "
          f"({'OK' if diff < 1e-4 else 'MISMATCH'})")
    pred = logits.argmax(1).cpu()
    names = CIFAR10_CLASSES if ds.__name__.endswith("cifar10") else None
    nm = (lambda k: names[k]) if names else (lambda k: str(k))
    L, n = maps.size(0), len(idx)
    H, W = imgs.shape[2:]
    maps_up = F.interpolate(maps.reshape(L * n, 1, *maps.shape[-2:]), size=(H, W), mode="bilinear",
                            align_corners=False).reshape(L, n, H, W).cpu().numpy()
    fig, axes = plt.subplots(n, L + 1, figsize=(1.15 * (L + 1) + 0.6, 1.2 * n + 1.2), squeeze=False)
    for r in range(n):
        im = imgs[r].permute(1, 2, 0).numpy()
        ax = axes[r, 0]
        ax.imshow(im, interpolation="nearest")
        t, p = int(yte[idx[r]]), int(pred[r])
        ax.set_title(f"true {nm(t)}\npred {nm(p)}", fontsize=6, color="green" if t == p else "red", pad=2)
        for l in range(L):
            m = maps_up[l, r]
            m = (m - m.min()) / (m.max() - m.min() + 1e-12)
            ax = axes[r, l + 1]
            ax.imshow(im, interpolation="nearest")
            ax.imshow(m, cmap="jet", alpha=0.55, vmin=0, vmax=1)
            if r == 0:
                ax.set_title(f"L{l + 1}", fontsize=7, pad=2)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"CLS-token attention to image patches, per layer, mean over heads ({title_extra})\n"
                 f"each map min-max normalised and overlaid on the image; L1 = first block", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.0 / (1.2 * n + 1.2) * 1.0, bottom=0.01,
                        wspace=0.04, hspace=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- 3/4. curves
def collect_curves(runs_dir, names):
    runs = []
    for n in names:
        r = load_results(runs_dir, n)
        if r is None:
            print(f"WARNING: no results.json for '{n}', skipping")
            continue
        if not r.get("history"):
            print(f"WARNING: '{n}' has empty history, skipping")
            continue
        runs.append((n, r))
    return runs


def plot_acc_curves(runs, title_extra, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    cmap = plt.get_cmap("tab10")
    drawn = 0
    for i, (n, r) in enumerate(runs):
        h = [e for e in r["history"] if "val_acc" in e and "train_acc_clean" in e]
        if not h:
            print(f"WARNING: '{n}' has no val_acc entries, skipping accuracy curves")
            continue
        ep = [e["epoch"] for e in h]
        va = np.array([e["val_acc"] for e in h]); ta = np.array([e["train_acc_clean"] for e in h])
        col = cmap(i % 10)
        mk = "o" if len(ep) < 15 else None
        a1.plot(ep, va, "-", color=col, marker=mk, ms=3, label=f"{n} (final val {va[-1]:.2f})")
        a1.plot(ep, ta, "--", color=col, marker=mk, ms=3)
        a2.plot(ep, ta - va, "-", color=col, marker=mk, ms=3, label=n)
        drawn += 1
    a1.set_xlabel("epoch"); a1.set_ylabel("accuracy (%)")
    a1.set_title("Accuracy: solid = validation, dashed = train (clean, no augmentation)", fontsize=9)
    a2.set_xlabel("epoch"); a2.set_ylabel("train acc - val acc (points)")
    a2.set_title("Generalisation gap (train clean - val); larger = more overfitting", fontsize=9)
    a2.axhline(0, color="k", lw=0.5)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    if drawn:
        a1.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Train vs validation accuracy {title_extra}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def plot_loss_and_throughput(runs, title_extra, loss_path, thr_path):
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (n, r) in enumerate(runs):
        h = r["history"]
        ax.plot([e["epoch"] for e in h], [e["train_loss"] for e in h], color=cmap(i % 10), label=n,
                marker="o" if len(h) < 15 else None, ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss (on augmented / mixed batches)")
    ax.set_title(f"Training loss per epoch {title_extra}\n(not comparable across runs with different "
                 f"label smoothing / MixUp)", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(loss_path, dpi=130); plt.close(fig)
    print(f"Saved {loss_path}")

    th = [(n, r.get("img_per_sec_steady")) for n, r in runs]
    th = [(n, v) for n, v in th if v]
    if not th:
        print("WARNING: no throughput data")
        return
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(th) + 2), 4.2))
    names = [n for n, _ in th]; vals = [v for _, v in th]
    bars = ax.bar(range(len(th)), vals, color=[cmap(i % 10) for i in range(len(th))])
    ax.bar_label(bars, labels=[f"{v / 1000:.1f}k" for v in vals], fontsize=7)
    ax.set_xticks(range(len(th))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("images / sec (steady state, training)")
    ax.set_title(f"Training throughput {title_extra}", fontsize=10)
    fig.tight_layout(); fig.savefig(thr_path, dpi=130); plt.close(fig)
    print(f"Saved {thr_path}")


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Visualize ViT runs")
    p.add_argument("--run", default=None, help="run name (default: best ViT run matching --prefix)")
    p.add_argument("--prefix", default="", help="dataset prefix filter, e.g. 'c100_'")
    p.add_argument("--curve-runs", nargs="+", default=None,
                   help="runs for curves (default: ablation ladder for the prefix that exist)")
    p.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    p.add_argument("--out", default=os.path.join(HERE, "report_assets"))
    p.add_argument("--n-images", type=int, default=8)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pre = args.prefix
    out = lambda f: os.path.join(args.out, pre + f)  # noqa: E731

    torch.manual_seed(0)
    name = args.run or pick_run(args.runs_dir, pre)
    if name is None:
        print(f"WARNING: no ViT run found for prefix '{pre}'; skipping plots 1-2")
    else:
        res = load_results(args.runs_dir, name)
        if res is None:
            print(f"WARNING: no results.json for run '{name}'; skipping plots 1-2")
        elif res["args"]["model"] == "resnet18":
            print(f"Run '{name}' is a ResNet (no pos_embed/attention); skipping plots 1-2")
        else:
            print(f"Using run '{name}' (best val {res.get('best_val_acc')})")
            try:
                model, xte, yte, ds = load_model_and_data(args.runs_dir, name, res, args.device)
                a = res["args"]
                extra = f"{name}: {a['model']}, patch {a['patch_size']}, val {res.get('final_val_acc')}%"
                plot_pos_embed(model, extra, out("pos_embed_similarity.png"))
                plot_attention(model, xte, yte, ds, args.n_images, args.device, extra, out("cls_attention.png"))
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"WARNING: plots 1-2 failed for '{name}': {e}")

    names = args.curve_runs or [pre + n for n in LADDER if os.path.isfile(os.path.join(args.runs_dir, pre + n, "results.json"))]
    runs = collect_curves(args.runs_dir, names)
    if not runs:
        print("WARNING: no runs for curves; done.")
        return
    extra = f"[{pre.rstrip('_') or 'cifar10'}]"
    plot_acc_curves(runs, extra, out("acc_curves.png"))
    plot_loss_and_throughput(runs, extra, out("loss_curves.png"), out("throughput.png"))


if __name__ == "__main__":
    main()
