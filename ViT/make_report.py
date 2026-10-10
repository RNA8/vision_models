#!/usr/bin/env python
"""Generate REPORT.md and a self-contained REPORT.html from ViT/runs/*/results.json.

  cd ViT && source env.sh && ../.venv/bin/python make_report.py [--include-smoke]
        [--runs-dir runs] [--assets report_assets] [--out-dir .] [--rate 3.29]
"""
import argparse, base64, html, json, os, re, subprocess, sys, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = {"cifar10": 32, "cifar100": 32, "tiny_imagenet": 64, "imagenet64": 64}

SECTIONS = [  # (key, title, prefix, dataset, asset prefix, expected run base names)
    ("cifar10", "CIFAR-10", "", "cifar10", "",
     ["t1_plain", "t2_ls_wd", "t3_randaug", "t4_mix", "t5_droppath", "t6_full_300ep", "r18_full",
      "t5b_mix_half", "t5c_bs256", "s_small_200ep", "s_small_200ep_lr1e-3", "p2_tiny", "p8_tiny",
      "bf1_slvit", "bf1b_tiny_samerecipe", "bf2_slvit_init", "bf3_slvit_300ep", "bf4_slvit_300ep_mixhalf",
      "bf1_slvit_lr1e-3", "bf1_slvit_lr5e-4",
      "rd_pilot_a", "rd_pilot_b", "rd_pilot_noinject", "rd_flat6"]),
    ("cifar100", "CIFAR-100", "c100_", "cifar100", "c100_",
     ["t1_plain", "t2_ls_wd", "t3_randaug", "t4_mix", "t5_droppath", "t6_full_300ep", "r18_full",
      "s_small_200ep"]),
    ("tin", "Tiny-ImageNet", "tin_", "tiny_imagenet", "tin_",
     ["t1_plain", "t2_ls_wd", "t3_randaug", "t4_mix", "t5_droppath", "r18_full", "s_small_200ep",
      "p4_tiny"]),
    ("in64", "ImageNet-64", "in64_", "imagenet64", "in64_",
     ["t1_plain", "t5_full", "r18_full", "s_small_full"]),
]
PREFIXES = ["c100_", "tin_", "in64_"]
LADDER_RE = re.compile(r"^t\d+_")

# ---------------------------------------------------------------- loading
def load_runs(runs_dir, include_smoke):
    runs = {}
    if not os.path.isdir(runs_dir):
        return runs
    for name in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, name, "results.json")
        if not os.path.isfile(p):
            continue
        if name.startswith("smoke") and not include_smoke:
            continue
        try:
            with open(p) as f:
                runs[name] = json.load(f)
        except Exception as e:
            print(f"warn: cannot read {p}: {e}", file=sys.stderr)
    return runs

def section_of(name, r):
    ds = (r.get("args") or {}).get("dataset")
    for key, _, prefix, dataset, _, _ in SECTIONS:
        if prefix and name.startswith(prefix):
            return key
    if name.startswith("smoke"):
        for key, _, _, dataset, _, _ in SECTIONS:
            if dataset == ds:
                return key
    return "cifar10"

def base_name(name, prefix):
    n = name[len(prefix):] if prefix and name.startswith(prefix) else name
    return n

def sort_key(base, expected):
    if base in expected:
        return (0, expected.index(base), base)
    return (1, 0, base)

# ---------------------------------------------------------------- derived text
def f1(x):
    return "n/a" if x is None else f"{x:.2f}"

def sgn(x, nd=1):
    return f"{x:+.{nd}f}"

def gap(r):
    t, v = r.get("final_train_acc_clean"), r.get("final_val_acc")
    return None if t is None or v is None else t - v

def a(r):
    return r.get("args") or {}

def describe_changes(prev, cur):
    """What changed in the recipe going prev -> cur (short text)."""
    p, c = a(prev), a(cur)
    parts = []
    if c.get("model") != p.get("model"):
        parts.append(f"model {p.get('model')}->{c.get('model')}")
    if c.get("patch_size") != p.get("patch_size"):
        parts.append(f"patch {p.get('patch_size')}->{c.get('patch_size')}")
    ls_wd = []
    if (c.get("label_smoothing") or 0) != (p.get("label_smoothing") or 0):
        ls_wd.append("label smoothing")
    if (c.get("weight_decay") or 0) != (p.get("weight_decay") or 0):
        ls_wd.append("weight decay")
    if ls_wd:
        parts.append("+" + "+".join(ls_wd))
    if bool(c.get("randaug")) != bool(p.get("randaug")):
        parts.append("+RandAugment" if c.get("randaug") else "-RandAugment")
    if (c.get("mixup") or 0) != (p.get("mixup") or 0) or (c.get("cutmix") or 0) != (p.get("cutmix") or 0):
        parts.append("+MixUp/CutMix")
    if (c.get("drop_path") or 0) != (p.get("drop_path") or 0):
        parts.append("+drop-path")
    if c.get("epochs") != p.get("epochs"):
        parts.append(f"epochs {p.get('epochs')}->{c.get('epochs')}")
    return ", ".join(parts) if parts else "no recipe change"

def delta_text(prev, cur):
    dv = cur["final_val_acc"] - prev["final_val_acc"]
    g0, g1 = gap(prev), gap(cur)
    s = f"val {sgn(dv)} pts"
    if g0 is not None and g1 is not None:
        verb = "shrinks" if g1 < g0 else ("grows" if g1 > g0 else "unchanged")
        s += f", gap {verb} {g0:.0f}→{g1:.0f}" if verb != "unchanged" else f", gap unchanged ({g1:.0f})"
    return s

def tokens(r):
    ps = a(r).get("patch_size")
    img = IMG_SIZE.get(a(r).get("dataset"), 32)
    return None if not ps else (img // ps) ** 2


ARG_LABELS = [("model", "model"), ("patch_size", "patch"), ("batch_size", "batch"), ("lr", "lr"), ("epochs", "epochs"),
              ("warmup_epochs", "warmup"), ("mix_prob", "mix-prob"), ("init", "init"), ("pool", "pool"), ("lsa", "LSA"),
              ("mix_off_epoch", "mix-off")]

def arg_diff(prev, cur):
    p, c = a(prev), a(cur)
    out = []
    for k, lab in ARG_LABELS:
        if p.get(k) != c.get(k) and not (p.get(k) is None and c.get(k) in ("default", "cls", False, 0)):
            out.append(f"{lab} {p.get(k)}->{c.get(k)}")
    return ", ".join(out) if out else "no recipe change"

# base name -> (parent base, what the step was for)
FOLLOWUPS = {
    "t5b_mix_half": ("t5_droppath",
        "t5 underfit (train acc ~ val acc, both low) so MixUp/CutMix was too strong for a 100-epoch budget; "
        "applying it to only half the batches makes the task easier"),
    "t5c_bs256": ("t5_droppath",
        "4x more optimizer steps for the same 100 epochs (lr scaled 2e-3->1e-3); steps, not regularization, were the limit. "
        "img/s is lower because this run was not CUDA-graphed"),
    "s_small_200ep_lr1e-3": ("s_small_200ep",
        "the lr 2e-3 Small run stalled (loss plateau ~2.1); halving the LR trains it, so that failure was optimization, not capacity"),
    "p2_tiny": ("t5_droppath",
        "patch 2 = (32/2)^2 = 256 tokens vs 64; same heavy (underfitting) recipe, so this shows what the extra tokens buy when steps are fixed; "
        "img/s was measured with other queues on the GPU, so treat it as indicative"),
    "p8_tiny": ("t5_droppath",
        "patch 8 = (32/8)^2 = 16 tokens vs 64; same heavy (underfitting) recipe; cheaper but loses spatial detail; "
        "img/s was measured with other queues on the GPU, so treat it as indicative"),
    "bf1_slvit": ("t5c_bs256",
        "SL-ViT shape (9 layers, 12 heads, MLP 2, 2.7M params) with the SL-ViT recipe (batch 128, lr 3e-3, warmup 10, reprob 0.25), "
        "compared with the best same-budget Tiny run"),
    "bf1b_tiny_samerecipe": ("bf1_slvit",
        "shape effect: identical SL-ViT recipe, but our 12-layer / 3-head / MLP-4 Tiny; train acc is also low, so it is an optimization stall, not overfitting"),
    "bf2_slvit_init": ("bf1_slvit",
        "init effect: identical to bf1 but impulse init; train acc is also low, so training stalled (suspect lr 3e-3 with this init; "
        "bf1_slvit_lr* will tell)"),
}

def followup_lesson(base, r, bb):
    parent, note = FOLLOWUPS[base]
    p = bb.get(parent)
    if p is None or p.get("final_val_acc") is None:
        return f"parent {parent} not available"
    dv = r["final_val_acc"] - p["final_val_acc"]
    g0, g1 = gap(p), gap(r)
    s = f"vs {parent}: {arg_diff(p, r)}: val {sgn(dv)} pts"
    if g0 is not None and g1 is not None:
        s += f", gap {g0:.0f}->{g1:.0f}"
    if r.get("img_per_sec_steady") and p.get("img_per_sec_steady"):
        s += f", img/s x{r['img_per_sec_steady'] / p['img_per_sec_steady']:.2f}"
    return s + ". " + note

def progress_text(runs_dir, name):
    """For a run with no results.json: parse the last 'ep i/N' line of train.log if present."""
    p = os.path.join(runs_dir, name, "train.log")
    if not os.path.isfile(p):
        return "not started"
    try:
        with open(p, errors="replace") as f:
            lines = f.read().splitlines()[-40:]
        for ln in reversed(lines):
            m = re.match(r"\s*ep\s+(\d+)/(\d+)", ln)
            if m:
                age = (datetime.datetime.now().timestamp() - os.path.getmtime(p)) / 60
                return f"running (epoch {m.group(1)}/{m.group(2)}, log updated {age:.0f} min ago); result pending"
    except Exception:
        pass
    return "started; result pending"

def lesson(base, r, ctx):
    """ctx: dict with prev_ladder (run or None), ref (full-recipe tiny run or None), vit_match."""
    if r is None:
        return ""
    if r.get("final_val_acc") is None:
        return "incomplete results"
    if base in FOLLOWUPS:
        return followup_lesson(base, r, ctx.get("bb") or {})
    if LADDER_RE.match(base) or base == "t5_full":
        prev = ctx.get("prev_ladder")
        if prev is None:
            return "baseline (crop+flip only)" if not a(r).get("randaug") and not a(r).get("weight_decay") else "first ladder run; no earlier step to compare"
        return f"{describe_changes(prev, r)}: {delta_text(prev, r)}"
    if a(r).get("model") == "resnet18":
        m = ctx.get("ref")
        if m is None:
            return "ResNet-18 control (no matching ViT run yet)"
        dv = r["final_val_acc"] - m["final_val_acc"]
        g0, g1 = gap(m), gap(r)
        gs = f", gap {g1:.0f} vs {g0:.0f}" if g0 is not None and g1 is not None else ""
        tm = f", {r['total_minutes'] / m['total_minutes']:.1f}x the time" if m.get("total_minutes") else ""
        who = "better" if dv > 0 else "worse"
        return (f"vs matching ViT ({m['run_name']}): val {sgn(dv)} pts ({who}){gs}{tm}; "
                f"conv inductive bias {'helps' if dv > 0 else 'does not help'} at this budget")
    if a(r).get("model") == "small":
        m = ctx.get("ref")
        if m is None:
            return "ViT-Small (no Tiny full-recipe run to compare yet)"
        dv = r["final_val_acc"] - m["final_val_acc"]
        return (f"ViT-Small vs Tiny ({m['run_name']}): val {sgn(dv)} pts, "
                f"{r['params_M']:.1f}M vs {m['params_M']:.1f}M params, "
                f"{a(r).get('epochs')} vs {a(m).get('epochs')} epochs")
    if "patch_size" in a(r) and a(r).get("patch_size") != a(ctx.get("ref") or {}).get("patch_size") and ctx.get("ref"):
        m = ctx["ref"]
        n1, n0 = tokens(r), tokens(m)
        dv = r["final_val_acc"] - m["final_val_acc"]
        sp = r["img_per_sec_steady"] / m["img_per_sec_steady"] if r.get("img_per_sec_steady") and m.get("img_per_sec_steady") else None
        s = (f"patch {a(r)['patch_size']}: tokens = ({IMG_SIZE.get(a(r).get('dataset'), 32)}/{a(r)['patch_size']})^2 = {n1} "
             f"vs {n0} for patch {a(m)['patch_size']}; val {sgn(dv)} pts")
        if sp:
            s += f", {sp:.2f}x img/s ({'faster' if sp > 1 else 'slower'})"
        s += "; " + ("more tokens cost throughput for accuracy" if n1 > n0 else "fewer tokens buy speed at some accuracy cost") \
            if dv * (n1 - n0) > 0 else "; token count change did not pay off in accuracy"
        return s.replace("; ; ", "; ")
    return "no comparison available"


# ---------------------------------------------------------------- published references
# (label, params, setting, acc, type, source text, url).  CIFAR-10 rows are from REFERENCES.md (verified 2026-10-10).
# CIFAR-100 rows come from RESEARCH_DIRECTIONS.md (fetched by research agents, not in the verified REFERENCES.md tables).
REFS = {
 "cifar10": [
  ("ViT-Tiny 192/12/3 p4, plain init (mimetic-init paper baseline)", "~5.4M", "100 ep, batch 512, RandAug+Cutout", 88.43, "ViT", "arXiv 2305.09828 Tab. 7", "https://arxiv.org/html/2305.09828"),
  ("DeiT-T patch 2 (NesT authors)", "5.3M", "n/s", 88.39, "ViT", "arXiv 2105.12723 Tab. A3", "https://arxiv.org/html/2105.12723"),
  ("ViT-Lite-7/4, no aug / no tuning", "3.72M", "300 ep", 83.59, "ViT", "arXiv 2104.05704 Tab. 8", "https://arxiv.org/html/2104.05704v4"),
  ("ViT-12/16", "85.6M", "300 ep, batch 128", 83.04, "ViT", "arXiv 2104.05704 Tab. 1", "https://arxiv.org/html/2104.05704v4"),
  ("omihub777/ViT-CIFAR (7L, dim 384)", "6.3M", "200 ep, batch 128", 90.92, "ViT", "GitHub", "https://github.com/omihub777/ViT-CIFAR"),
  ("ViT-Lite-7/4, full DeiT-style recipe", "3.72M", "300 ep, batch 128", 93.57, "ViT", "arXiv 2104.05704 Tab. 1", "https://arxiv.org/html/2104.05704v4"),
  ("ViT baseline (depth 9, dim 192), SPT/LSA paper", "2.8M", "100 ep, batch 128", 93.58, "ViT", "arXiv 2112.13492 Tab. 2", "https://arxiv.org/html/2112.13492v1"),
  ("SL-ViT (ViT + SPT + LSA)", "2.9M", "100 ep, batch 128", 94.53, "ViT (locality)", "arXiv 2112.13492 Tab. 2", "https://arxiv.org/html/2112.13492v1"),
  ("CCT-7/3x1 (conv tokenizer, hybrid)", "3.76M", "300 ep, batch 128", 96.53, "hybrid", "arXiv 2104.05704 Tab. 1", "https://arxiv.org/html/2104.05704v4"),
  ("ResNet-110 (He et al.)", "1.7M", "~164 ep, batch 128", 93.57, "CNN", "arXiv 1512.03385 Tab. 6", "https://ar5iv.labs.arxiv.org/html/1512.03385"),
  ("ResNet18 (kuangliu/pytorch-cifar)", "11M", "n/s", 93.02, "CNN", "GitHub", "https://github.com/kuangliu/pytorch-cifar"),
  ("ResNet18 (Cutout paper), crop+flip", "11M", "200 ep, batch 128", 95.28, "CNN", "arXiv 1708.04552", "https://ar5iv.labs.arxiv.org/html/1708.04552"),
  ("WRN-28-10", "36.5M", "200 ep", 96.00, "CNN", "arXiv 1605.07146", "https://ar5iv.labs.arxiv.org/html/1605.07146"),
  ("airbench94 (Keller Jordan)", "small CNN", "3.29 s on 1x A100", 94.0, "CNN", "arXiv 2404.00498", "https://arxiv.org/abs/2404.00498"),
 ],
 "cifar100": [
  ("ViT 12L/3h/MLP4 (same shape as ours), Gani et al. recipe [not re-checked]", "n/s", "n/s", 68.09, "ViT", "arXiv 2210.07240 Tab. 15 (quoted by research agent, RESEARCH_DIRECTIONS.md)", "https://arxiv.org/abs/2210.07240"),
  ("L-ViT (+LSA), SL-ViT paper ablation [not re-checked]", "n/s", "n/s", 74.87, "ViT (locality)", "arXiv 2112.13492 (quoted by research agent, RESEARCH_DIRECTIONS.md)", "https://arxiv.org/abs/2112.13492"),
  ("ViTRM (Vision Tiny Recursion Model) [verified]", "~3.6M", "n/s", 72.1, "ViT (recurrent)", "arXiv 2603.19503 (REFERENCES.md)", "https://arxiv.org/abs/2603.19503"),
  ("ViT baseline (depth 9, dim 192), SPT/LSA paper [verified]", "2.8M", "100 ep, batch 128", 73.81, "ViT", "arXiv 2112.13492 Tab. 2 (REFERENCES.md)", "https://arxiv.org/html/2112.13492v1"),
  ("SL-ViT (ViT + SPT + LSA) [verified]", "2.9M", "100 ep, batch 128", 76.92, "ViT (locality)", "arXiv 2112.13492 Tab. 2 (REFERENCES.md)", "https://arxiv.org/html/2112.13492v1"),
  ("ViT-Lite-7/4 [verified]", "3.72M", "300 ep, batch 128", 73.94, "ViT", "arXiv 2104.05704 Tab. 1 (REFERENCES.md)", "https://arxiv.org/html/2104.05704v4"),
  ("ViT + SSL pretraining on CIFAR (Gani et al.) [verified]", "2.8M", "200 SSL + 100 ep", 79.15, "ViT, 2-stage", "arXiv 2210.07240 Tab. 3 (REFERENCES.md)", "https://arxiv.org/html/2210.07240"),
  ("CCT-7/3x1 (conv tokenizer, hybrid) [verified]", "3.76M", "300 ep, batch 128", 80.92, "hybrid", "arXiv 2104.05704 Tab. 1 (REFERENCES.md)", "https://arxiv.org/html/2104.05704v4"),
  ("CCT-7/3x1, long schedule [verified]", "3.76M", "5000 ep", 82.87, "hybrid", "arXiv 2104.05704 Tab. 2 (REFERENCES.md)", "https://arxiv.org/html/2104.05704v4"),
 ],
}
REF_NOTE = {
 "cifar10": "Reference rows are from `REFERENCES.md` (verified 2026-10-10; epochs/batch differ from ours, so this is a rough yardstick, not a controlled comparison).",
 "cifar100": "Rows tagged [verified] are from `REFERENCES.md` (CIFAR-100 section, re-checked against the source pages). The same-shape ViT (68.09) and L-ViT (74.87) rows were quoted by a research agent in `RESEARCH_DIRECTIONS.md` and are not independently re-checked. "
             "No verified published CIFAR-100 ResNet-18 number is recorded, so the CNN comparison is our own r18_full only.",
}

def best_of(runs, key, prefix, cnn):
    best = None
    for n, r in runs.items():
        if section_of(n, r) != key or r.get("final_val_acc") is None:
            continue
        is_cnn = a(r).get("model") == "resnet18"
        if is_cnn != cnn:
            continue
        if best is None or r["final_val_acc"] > best["final_val_acc"]:
            best = r
    return best

def build_comparison(key, runs, prefix):
    """-> (header, rows, link_col_idx) ; rows: [label, params, setting, acc, delta vs our best ViT, delta vs our best CNN, source, url]"""
    vit, cnn = best_of(runs, key, prefix, False), best_of(runs, key, prefix, True)
    refs = REFS.get(key)
    if not refs:
        return None
    rows = []
    for r, kind in ((vit, "ViT"), (cnn, "CNN")):
        if r is not None:
            rows.append([f"**ours: {r['run_name']}** ({a(r).get('model')}, {a(r).get('epochs')} ep, batch {a(r).get('batch_size')})",
                         f"{r['params_M']:.1f}M", "this repo", f"{r['final_val_acc']:.2f}", "", "", kind, "this repo", ""])
    for lab, par, setting, acc, typ, src, url in refs:
        d1 = f"{vit['final_val_acc'] - acc:+.1f}" if vit else "n/a"
        d2 = f"{cnn['final_val_acc'] - acc:+.1f}" if cnn else "n/a"
        rows.append([lab, par, setting, f"{acc:.2f}", d1, d2, typ, src, url])
    return rows

CMP_HEAD = ["model", "params", "setting", "acc (%)", "our best ViT minus ref", "our best CNN minus ref", "type", "source"]

# ---------------------------------------------------------------- wall-clock
def run_intervals(runs_dir, runs):
    """(start, end) epoch-seconds per completed run: end = results.json mtime, start = end - total_minutes."""
    iv = []
    for n, r in runs.items():
        p = os.path.join(runs_dir, n, "results.json")
        if not os.path.isfile(p) or not r.get("total_minutes"):
            continue
        end = os.path.getmtime(p)
        start = end - 60 * r["total_minutes"]  # fallback: training minutes only
        try:  # run-dir birth time (mkdir at launch) also covers startup, compile and final eval
            out = subprocess.run(["stat", "-c", "%W", os.path.join(runs_dir, n)], capture_output=True, text=True).stdout.strip()
            if out.isdigit() and int(out) > 0 and int(out) < end:
                start = min(start, float(out))
        except Exception:
            pass
        iv.append((start, end, n))
    # runs in progress (train.log updated in the last 15 min, no results.json): count up to now
    now_t = datetime.datetime.now().timestamp()
    for n in sorted(os.listdir(runs_dir)):
        d, lg = os.path.join(runs_dir, n), os.path.join(runs_dir, n, "train.log")
        if n in runs or n.startswith("smoke") or not os.path.isfile(lg) or os.path.isfile(os.path.join(d, "results.json")):
            continue
        if now_t - os.path.getmtime(lg) > 900:
            continue
        out = subprocess.run(["stat", "-c", "%W", d], capture_output=True, text=True).stdout.strip()
        if out.isdigit() and int(out) > 0:
            iv.append((float(out), now_t, n + " (running)"))
    return iv

def wallclock(iv):
    if not iv:
        return None
    t0, t1 = min(i[0] for i in iv), max(i[1] for i in iv)
    union, cur_s, cur_e = 0.0, None, None
    for st, en, _ in sorted(iv):
        if cur_e is None or st > cur_e:
            if cur_e is not None:
                union += cur_e - cur_s
            cur_s, cur_e = st, en
        else:
            cur_e = max(cur_e, en)
    union += cur_e - cur_s
    # peak concurrency
    ev = sorted([(i[0], 1) for i in iv] + [(i[1], -1) for i in iv])
    c = mx = 0
    for _, d in ev:
        c += d; mx = max(mx, c)
    return {"start": t0, "end": t1, "span_min": (t1 - t0) / 60, "busy_min": union / 60, "max_conc": mx}

# ---------------------------------------------------------------- tables
HEAD = ["run", "epochs", "params (M)", "clean train acc", "val acc", "train-val gap", "minutes", "img/s",
        "what this step taught"]

def build_section(sec, runs, runs_dir=None):
    key, title, prefix, dataset, aprefix, expected = sec
    mine = {n: r for n, r in runs.items() if section_of(n, r) == key}
    bases = {base_name(n, prefix): n for n in mine}
    order = list(expected) + sorted(b for b in bases if b not in expected)
    # full-recipe tiny reference
    ref = None
    for cand in ("t5_droppath", "t5_full"):
        if cand in bases:
            ref = mine[bases[cand]]
            break
    rows, prev_ladder = [], None
    for b in sorted(order, key=lambda b: sort_key(b, expected)):
        n = bases.get(b)
        r = mine.get(n) if n else None
        disp = (prefix + b) if r is None else n
        if r is None:
            rows.append([disp] + ["pending"] * 7 + [progress_text(runs_dir, disp) if runs_dir else ""])
            continue
        ctx = {"prev_ladder": prev_ladder, "ref": ref, "bb": {b2: mine[n2] for b2, n2 in bases.items()}}
        is_ladder = bool(LADDER_RE.match(b)) or b == "t5_full"
        if not is_ladder and a(r).get("model") == "tiny" and ref is not None and r is not ref \
           and a(r).get("patch_size") == a(ref).get("patch_size"):
            pass
        row_ref = ref
        if r is ref and not is_ladder:
            row_ref = None
        ctx["ref"] = row_ref
        v, best = r.get("final_val_acc"), r.get("best_val_acc")
        vs = f1(v) if best is None or v is None or abs(best - v) < 0.005 else f"{v:.2f} ({best:.2f})"
        g = gap(r)
        rows.append([disp, str(a(r).get("epochs", "?")), f"{r.get('params_M', float('nan')):.2f}",
                     f1(r.get("final_train_acc_clean")), vs, "n/a" if g is None else f"{g:.1f}",
                     f"{r.get('total_minutes', 0):.1f}", f"{r.get('img_per_sec_steady', 0):,.0f}",
                     lesson(b, r, ctx)])
        if is_ladder and v is not None:
            prev_ladder = r
    mins = sum(r.get("total_minutes") or 0 for r in mine.values())
    n_pending = sum(1 for row in rows if row[1] == "pending")
    return {"key": key, "title": title, "rows": rows, "minutes": mins, "n": len(mine), "pending": n_pending,
            "aprefix": aprefix}

# ---------------------------------------------------------------- assets
def find_assets(assets_dir, aprefix, all_prefixes):
    if not os.path.isdir(assets_dir):
        return []
    out = []
    for f in sorted(os.listdir(assets_dir)):
        if not f.lower().endswith(".png"):
            continue
        has_pref = next((p for p in all_prefixes if f.startswith(p)), "")
        if has_pref == aprefix:
            out.append(f)
    skip = ("followups_loss", "followups_thr", "bf_loss", "bf_thr")  # keep the report compact: accuracy curves only
    out = [f for f in out if not any(f[len(aprefix):].startswith(k) for k in skip)]
    order = ["acc_curves", "followups_acc_curves", "bf_acc_curves", "loss_curves", "throughput", "pos_embed", "cls_attention"]
    return sorted(out, key=lambda f: next((i for i, o in enumerate(order) if f[len(aprefix):].startswith(o)), 99))

CAPTIONS = {"acc_curves": "Ablation ladder: train (dashed) vs validation (solid) accuracy and gap",
            "followups_acc_curves": "Follow-up runs: t5 vs t5b/t5c, ViT-Small LR fix, patch 2/8, ResNet-18",
            "bf_acc_curves": "Baseline-fix runs: SL-ViT shape/recipe (bf1) vs Tiny shape (bf1b) vs impulse init (bf2), with t5c for reference"}

def pretty(f):
    base = re.sub(r"^(c100_|tin_|in64_)", "", os.path.splitext(f)[0])
    if base in CAPTIONS:
        return CAPTIONS[base]
    return os.path.splitext(f)[0].replace("_", " ")

# ---------------------------------------------------------------- markdown -> html
def inline(s):
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", s)
    return s

def md_to_html(text):
    out, para, in_list, in_code = [], [], False, False
    def flush():
        nonlocal para
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para = []
    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            flush(); close_list()
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line)); continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush(); close_list()
            lvl = min(len(m.group(1)) + 1, 6)  # shift down: report owns h1
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>"); continue
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            flush()
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{inline(m.group(1))}</li>"); continue
        if not line.strip():
            flush(); close_list(); continue
        para.append(line.strip())
    flush(); close_list()
    if in_code:
        out.append("</pre>")
    return "\n".join(out)

CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1d2330;--mut:#5d6677;--bd:#d9dde5;--hd:#f1f3f7;--ac:#2a5bd7}
@media(prefers-color-scheme:dark){:root{--bg:#14171d;--fg:#e4e7ee;--mut:#9aa3b5;--bd:#2f3542;--hd:#1d222b;--ac:#7aa2ff}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}
h1,h2,h3{line-height:1.25}h2{border-bottom:1px solid var(--bd);padding-bottom:.25rem;margin-top:2.2rem}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:.8rem 0}
th,td{border:1px solid var(--bd);padding:.35rem .5rem;text-align:left;vertical-align:top}
th{background:var(--hd)}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.pend td{color:var(--mut);font-style:italic}code,pre{background:var(--hd);border-radius:4px}code{padding:0 .25rem}pre{padding:.6rem;overflow-x:auto}
img{max-width:100%;height:auto;border:1px solid var(--bd);border-radius:4px;background:#fff}
figure{margin:1rem 0}figcaption{color:var(--mut);font-size:13px}.mut{color:var(--mut)}
"""

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--assets", default=os.path.join(HERE, "report_assets"))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--notes", default=os.path.join(HERE, "REPORT_NOTES.md"))
    ap.add_argument("--include-smoke", action="store_true")
    ap.add_argument("--rate", type=float, default=3.29, help="GPU $/hour")
    args = ap.parse_args()

    runs = load_runs(args.runs_dir, args.include_smoke)
    secs = [build_section(s, runs, args.runs_dir) for s in SECTIONS]
    secs = [s for s in secs if s["n"] > 0 or s["key"] == "cifar10" or True]
    total_min = sum(s["minutes"] for s in secs)
    cost = lambda m: m / 60 * args.rate
    all_pref = PREFIXES
    notes = None
    if os.path.isfile(args.notes):
        with open(args.notes) as f:
            notes = f.read()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    md = ["# ViT training report", "",
          f"_Generated {now} from `{args.runs_dir}`; {len(runs)} runs included._", "",
          "Gap = clean train acc minus val acc (points). Val acc shows final epoch, best in parentheses when different.", "",
          "## Compute summary", "", "| section | runs done | pending | GPU minutes | cost |", "|---|---|---|---|---|"]
    for s in secs:
        md.append(f"| {s['title']} | {s['n']} | {s['pending']} | {s['minutes']:.1f} | ${cost(s['minutes']):.2f} |")
    md.append(f"| **Total (sum of run-minutes)** | {len(runs)} | {sum(s['pending'] for s in secs)} | **{total_min:.1f}** ({total_min/60:.2f} h) | ${cost(total_min):.2f} at ${args.rate}/h (upper bound) |")
    iv = run_intervals(args.runs_dir, runs)
    wc = wallclock(iv)
    fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
    wc_lines = []
    if wc:
        n_run = sum(1 for i in iv if i[2].endswith("(running)"))
        run_min = sum((i[1] - i[0]) / 60 for i in iv if i[2].endswith("(running)"))
        wc_done = wallclock([i for i in iv if not i[2].endswith("(running)")])
        why = (f"Completed runs rarely overlapped (peak {wc_done['max_conc']}), so the span exceeds the sum of run-minutes by idle gaps and "
               f"startup/eval time (the sum will overstate wall-clock once the concurrent queues complete); "
               f"{n_run} runs currently in progress ({run_min:.0f} min elapsed so far; peak concurrency now {wc['max_conc']}) are counted in the span but not in the sum")
        wc_lines = [
            f"**Wall-clock (estimated, counting in-progress runs up to now):** {wc['span_min']:.0f} min ({wc['span_min']/60:.2f} h) from the earliest run start ({fmt(wc['start'])}) "
            f"to the latest run end ({fmt(wc['end'])}); **GPU cost on wall-clock: ${cost(wc['span_min']):.2f}** at ${args.rate}/h.",
            f"Sum of completed run-minutes is {total_min:.0f} min. {why}; time with at least one run active is {wc['busy_min']:.0f} min (${cost(wc['busy_min']):.2f}). "
            "Start = run-directory creation time (falls back to results.json mtime minus total_minutes), end = results.json mtime (in-progress runs end at report time). "
            "Per-run minutes and img/s of overlapping runs are inflated by GPU sharing."]
    md.append("")
    md += wc_lines + [""]
    hsecs = [f"<h1>ViT training report</h1><p class='mut'>Generated {now}; {len(runs)} runs included. "
             "Gap = clean train acc minus val acc (points). Val acc shows final epoch, best in parentheses when different.</p>",
             "<h2>Compute summary</h2><div class='tw'><table><tr><th>section</th><th>runs done</th><th>pending</th><th>GPU minutes</th><th>cost</th></tr>"]
    for s in secs:
        hsecs.append(f"<tr><td>{s['title']}</td><td class='n'>{s['n']}</td><td class='n'>{s['pending']}</td>"
                     f"<td class='n'>{s['minutes']:.1f}</td><td class='n'>${cost(s['minutes']):.2f}</td></tr>")
    hsecs.append(f"<tr><td><b>Total (sum of run-minutes)</b></td><td class='n'>{len(runs)}</td><td class='n'>{sum(s['pending'] for s in secs)}</td>"
                 f"<td class='n'><b>{total_min:.1f}</b> ({total_min/60:.2f} h)</td><td class='n'>${cost(total_min):.2f} at ${args.rate}/h (upper bound)</td></tr></table></div>")
    for ln in wc_lines:
        hsecs.append("<p>" + inline(ln) + "</p>")

    for s in secs:
        md += [f"## {s['title']}", ""]
        hsecs.append(f"<h2>{s['title']}</h2>")
        if s["n"] == 0:
            md += ["_No runs yet; all rows pending._", ""]
        md.append("| " + " | ".join(HEAD) + " |")
        md.append("|" + "---|" * len(HEAD))
        hsecs.append("<div class='tw'><table><tr>" + "".join(f"<th>{h}</th>" for h in HEAD) + "</tr>")
        for row in s["rows"]:
            pend = row[1] == "pending"
            md.append("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |")
            cells = "".join(f"<td{'' if i in (0, 8) else ' class=n'}>{html.escape(c)}</td>" for i, c in enumerate(row))
            hsecs.append(f"<tr{' class=pend' if pend else ''}>{cells}</tr>")
        hsecs.append("</table></div>")
        md += ["", f"Subtotal: **{s['minutes']:.1f} GPU min** ({s['minutes']/60:.2f} h), **${cost(s['minutes']):.2f}**", ""]
        hsecs.append(f"<p>Subtotal: <b>{s['minutes']:.1f} GPU min</b> ({s['minutes']/60:.2f} h), <b>${cost(s['minutes']):.2f}</b></p>")
        cmp_rows = build_comparison(s["key"], runs, s["aprefix"])
        if cmp_rows:
            md += ["### Comparison with published results", "", REF_NOTE[s["key"]], "",
                   "| " + " | ".join(CMP_HEAD) + " |", "|" + "---|" * len(CMP_HEAD)]
            hsecs.append("<h3>Comparison with published results</h3><p>" + inline(REF_NOTE[s["key"]]) + "</p>"
                         "<div class='tw'><table><tr>" + "".join(f"<th>{h}</th>" for h in CMP_HEAD) + "</tr>")
            for lab, par, setting, acc, d1, d2, typ, src, url in cmp_rows:
                srcmd = f"[{src}]({url})" if url else src
                md.append("| " + " | ".join([lab, par, setting, acc, d1, d2, typ, srcmd]) + " |")
                srch = f"<a href=\"{html.escape(url)}\">{html.escape(src)}</a>" if url else html.escape(src)
                cells = [inline(lab), html.escape(par), html.escape(setting), acc, d1, d2, html.escape(typ), srch]
                hsecs.append("<tr>" + "".join(f"<td{' class=n' if i in (3, 4, 5) else ''}>{c}</td>" for i, c in enumerate(cells)) + "</tr>")
            hsecs.append("</table></div>")
            md.append("")
        pngs = find_assets(args.assets, s["aprefix"], all_pref)
        for f in pngs:
            p = os.path.join(args.assets, f)
            rel = os.path.relpath(p, args.out_dir)
            md += [f"![{pretty(f)}]({rel})", ""]
            with open(p, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            hsecs.append(f"<figure><img alt=\"{html.escape(pretty(f))}\" src=\"data:image/png;base64,{b64}\">"
                         f"<figcaption>{html.escape(pretty(f))}</figcaption></figure>")

    if notes:
        md += ["## What I learned", "", notes.strip(), ""]
        hsecs.append("<h2>What I learned</h2>" + md_to_html(notes))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(args.out_dir, "REPORT.html"), "w") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                f"<title>ViT training report</title><style>{CSS}</style></head><body>\n" + "\n".join(hsecs) + "\n</body></html>")
    print(f"wrote {os.path.join(args.out_dir, 'REPORT.md')} and REPORT.html "
          f"({len(runs)} runs, {total_min:.1f} GPU min, ${cost(total_min):.2f})")

if __name__ == "__main__":
    main()
