#!/usr/bin/env python3
"""Build the final dashboard + step-by-step educational report from everything under ViT/runs.

    ../.venv/bin/python make_dashboard.py          # from ViT/ (after `source env.sh`)

Writes (atomically):
  DASHBOARD.html           standalone HTML document for the repo
  dashboard_artifact.html  the same page as a fragment (<title>, <style>, content) for a claude.ai Artifact

Reads runs/*/results.json, recurrent_eval.json, edge_bench.json, train.log (unfinished runs), the run_*.sh queue
scripts (for the exact CLI flags), REFERENCES.md / research/*.md (every published number is checked to appear
in its source file), REPORT_NOTES.md (bench numbers) and git log (dates). Charts are inline SVG built here;
no CDN, no JS. Safe to rerun at any time: missing runs show up as running / queued.
"""
import base64
import datetime as dt
import html
import io
import itertools
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PRICE = 3.29  # $/GPU-hour (1x H100 PCIe)
NOW = time.time()

esc = html.escape


# ----------------------------------------------------------------------------------------------- data loading
def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def wandb_start(d):
    """Start time of a run from its wandb dir name (run-YYYYMMDD_HHMMSS-id), else None."""
    best = None
    for p in (d / "wandb").glob("run-*") if (d / "wandb").is_dir() else []:
        m = re.match(r"run-(\d{8}_\d{6})-", p.name)
        if m:
            t = time.mktime(time.strptime(m.group(1), "%Y%m%d_%H%M%S"))
            best = t if best is None else min(best, t)
    return best


def load_runs():
    runs = {}
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name in ("smoke_real", "int8_audit"):
            continue
        r = {"name": d.name, "dir": d, "res": jload(d / "results.json"), "rec": jload(d / "recurrent_eval.json")}
        eb = jload(d / "edge_bench.json")
        if isinstance(eb, list):
            eb = eb[-1] if eb else None
        r["edge"] = eb
        log = d / "train.log"
        if r["res"]:
            r["status"] = "done"
            end = (d / "results.json").stat().st_mtime
            r["end"] = end
            r["start"] = end - 60 * r["res"].get("total_minutes", 0)
        elif log.exists():
            age = NOW - log.stat().st_mtime
            r["status"] = "running" if age < 1800 else "stalled"
            r["start"] = wandb_start(d) or log.stat().st_mtime
            r["end"] = log.stat().st_mtime
            r["progress"] = log_progress(log)
        else:
            r["status"] = "queued"
        runs[d.name] = r
    return runs


def log_progress(log):
    """Last 'ep  N/M | loss L' line of a train.log -> (N, M, loss) or None."""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, log.stat().st_size - 20000))
            tail = f.read().decode("utf8", "replace").replace("\r", "\n")
    except Exception:
        return None
    hits = re.findall(r"ep\s+(\d+)/(\d+)\s*\|\s*loss\s+([\d.]+)", tail)
    if not hits:
        return None
    n, m, loss = hits[-1]
    return int(n), int(m), float(loss)


# ------------------------------------------------------------------------------- exact CLI flags from run_*.sh
def parse_scripts():
    """Map run name -> (script, expanded flag string) by expanding the simple bash in run_*.sh."""
    out = {}
    for sh in sorted(ROOT.glob("run_*.sh")):
        txt = sh.read_text()
        txt = re.sub(r"\\\n\s*", " ", txt)
        vars_ = {}
        for m in re.finditer(r'^\s*([A-Z][A-Z0-9_]*)="(.*?)"', txt, re.S | re.M):
            vars_[m.group(1)] = " ".join(m.group(2).split())
        for m in re.finditer(r"^\s*([A-Z][A-Z0-9_]*)=([^\s\"'$(][^\s;]*)\s*$", txt, re.M):
            vars_.setdefault(m.group(1), m.group(2))

        def expand(s, extra):
            env = dict(vars_, **extra)
            for _ in range(5):
                s = re.sub(r"\$\{(\w+)/([^/}]*)/([^}]*)\}", lambda m: env.get(m.group(1), "").replace(m.group(2), m.group(3), 1), s)
                s = re.sub(r"\$\{(\w+)\}|\$(\w+)", lambda m: env.get(m.group(1) or m.group(2), m.group(0)), s)
            return " ".join(s.split())

        # run() helpers that build the name from a prefix ("local name=c100_rd_c2_$1") or take the script first ("run <x.py> <name>")
        mp = re.search(r"local name=([\w.-]*)\$1", txt)
        name_prefix = mp.group(1) if mp else ""
        loops = []
        body = txt.split("\n")
        for line in body:
            line = re.sub(r"(^|\s)#.*$", "", line) if not line.lstrip().startswith("#!") else ""
            for stmt in re.split(r";;|;", line):
                stmt = stmt.strip()
                stmt = re.sub(r"^\w[\w|]*\)\s*", "", stmt)  # case label "a)"
                if stmt.startswith("do "):
                    stmt = stmt[3:].strip()
                m = re.match(r"for (\w+) in (.+)$", stmt)
                if m:
                    loops.append((m.group(1), m.group(2).split()))
                    continue
                if stmt == "done" or stmt.startswith("done"):
                    if loops:
                        loops.pop()
                    continue
                stmt = re.sub(r"^(run|train|ft)\s+\S+\.py\s+", r"\1 ", stmt)
                m = re.match(r"(run|train|ft)\s+(\S+)\s*(.*)$", stmt)
                if not m or m.group(2).startswith("<") or m.group(2).startswith('"$'):
                    continue
                names = [v for v, _ in loops]
                for combo in itertools.product(*[vals for _, vals in loops]) if loops else [()]:
                    extra = dict(zip(names, combo))
                    name = name_prefix + expand(m.group(2), extra).strip('"')
                    if "$" in name:
                        continue
                    flags = expand(m.group(3), extra)
                    common = {"run": "COMMON", "train": None, "ft": "FT"}[m.group(1)]
                    pre = []
                    if sh.name == "run_seeds_ablation.sh":
                        pre = [vars_.get("C10", "")]
                    elif common and common in vars_:
                        pre = [vars_[common]]
                    elif m.group(1) == "train":
                        pre = ["--wandb --eval-every 5 --cudagraphs"]
                    tool = "finetune.py" if (m.group(1) == "ft" or sh.name == "run_edge_finetune.sh") else "train_vit.py"
                    out.setdefault(name, (sh.name, f"{tool} --run-name {name} " + " ".join(p for p in pre + [flags] if p)))
    return out


def flags_from_args(r):
    """Fallback when a run was launched by a loop the parser cannot expand (edge sweeps)."""
    a = (r["res"] or {}).get("args", {})
    if not a:
        return None
    if "img_size" in a:  # finetune.py
        mdl = a.get("model") or ""
        src = f"--model {mdl}" if mdl else f"--init-ckpt {a.get('init_ckpt')}"
        s = f"finetune.py --run-name {r['name']} {src} --img-size {a['img_size']} --lr {a['lr']:g}"
        if a.get("lr_decay") and a["lr_decay"] != 0.75 or "vit_" in mdl or "deit" in mdl:
            s += f" --lr-decay {a.get('lr_decay', 0):g}" if a.get("lr_decay") else ""
        if a.get("teacher"):
            s += f" --teacher {a['teacher']} --kd {a['kd']} --kd-alpha {a['kd_alpha']:g} --kd-tau {a['kd_tau']:g}"
        s += (f" --dataset {a['dataset']} --epochs {a['epochs']} --batch-size {a['batch_size']} --warmup-epochs "
              f"{a['warmup_epochs']} --weight-decay {a['weight_decay']:g} --randaug --drop-path {a['drop_path']:g} --ema {a['ema']:g}")
        return s
    return None


# ----------------------------------------------------------------------------- published numbers (checked)
SRC_FILES = {"REF": ROOT / "REFERENCES.md", "C100": ROOT / "research/cifar100_sota_small_models.md",
             "IMP": ROOT / "research/small_vit_improvements.md", "DIR": ROOT / "RESEARCH_DIRECTIONS.md",
             "EDGE": ROOT / "research/edge_architectures.md"}
# key: (label, value, setting, url, file key, needle that must appear in the file)
PUB = {
    "vitlite_c10": ("ViT-Lite-7/4 (CCT paper)", 93.57, "CIFAR-10, scratch, 3.72M, 300 ep", "https://arxiv.org/html/2104.05704v4", "REF", "93.57"),
    "sptvit_c10": ("ViT baseline, SPT/LSA paper", 93.58, "CIFAR-10, scratch, 2.8M, 100 ep (x3 repeated aug)", "https://arxiv.org/html/2112.13492v1", "REF", "93.58"),
    "slvit_c10": ("SL-ViT (ViT + SPT + LSA)", 94.53, "CIFAR-10, scratch, 2.9M", "https://arxiv.org/html/2112.13492v1", "REF", "94.53"),
    "cvt_c10": ("CVT-7/4 (SeqPool)", 94.01, "CIFAR-10, scratch, 3.72M", "https://arxiv.org/html/2104.05704v4", "REF", "94.01"),
    "cct_c10": ("CCT-7/3x1 (conv tokenizer, hybrid)", 96.53, "CIFAR-10, scratch, 3.76M, 300 ep", "https://arxiv.org/html/2104.05704v4", "REF", "96.53"),
    "cct5k_c10": ("CCT-7/3x1, 5000 epochs", 98.00, "CIFAR-10, scratch, 3.76M", "https://arxiv.org/html/2104.05704v4", "REF", "98.00"),
    "mimetic_c10": ("ViT-Tiny 192/12/3 p4, plain init", 88.43, "CIFAR-10, scratch, ~5.4M, 100 ep", "https://arxiv.org/html/2305.09828", "REF", "88.43"),
    "r18cutout_c10": ("ResNet-18 (Cutout paper, crop+flip)", 95.28, "CIFAR-10, 200 ep, batch 128", "https://ar5iv.labs.arxiv.org/html/1708.04552", "REF", "95.28"),
    "r18kuangliu_c10": ("ResNet18 (kuangliu/pytorch-cifar)", 93.02, "CIFAR-10", "https://github.com/kuangliu/pytorch-cifar", "REF", "93.02"),
    "airbench": ("airbench94: 94% in 3.29 s on one A100", 94.0, "CIFAR-10, small CNN", "https://arxiv.org/abs/2404.00498", "REF", "3.29"),
    "structinit_c10": ("ViT-Tiny + impulse init, 300 ep", 94.67, "CIFAR-10, scratch", "https://arxiv.org/abs/2505.19985", "DIR", "94.67"),
    "vitrm_c10": ("ViTRM (recursion model)", 93.1, "CIFAR-10, scratch, ~3.6M", "https://arxiv.org/abs/2603.19503", "DIR", "93.1"),
    "gani_c10": ("ViT + SSL pretraining on CIFAR (Gani et al.)", 96.41, "CIFAR-10, 2.8M, 2-stage", "https://arxiv.org/html/2210.07240", "REF", "96.41"),
    "sptvit_c100": ("ViT baseline, SPT/LSA paper", 73.81, "CIFAR-100, scratch, 2.8M", "https://arxiv.org/html/2112.13492v1", "REF", "73.81"),
    "vitlite_c100": ("ViT-Lite-7/4", 73.94, "CIFAR-100, scratch, 3.72M, 300 ep", "https://arxiv.org/html/2104.05704v4", "REF", "73.94"),
    "slvit_c100": ("SL-ViT", 76.92, "CIFAR-100, scratch, 2.9M", "https://arxiv.org/html/2112.13492v1", "REF", "76.92"),
    "vitrm_c100": ("ViTRM (recursion model)", 72.1, "CIFAR-100, scratch, ~3.6M", "https://arxiv.org/abs/2603.19503", "REF", "72.1"),
    "gani_c100": ("ViT + SSL pretraining on CIFAR (Gani et al.)", 79.15, "CIFAR-100, 2.8M, 2-stage", "https://arxiv.org/html/2210.07240", "REF", "79.15"),
    "gani_tin": ("ViT + SSL pretraining (Gani et al.)", 63.36, "Tiny-ImageNet, 2.8M, 2-stage", "https://arxiv.org/html/2210.07240", "REF", "63.36"),
    "cct_c100": ("CCT-7/3x1 (hybrid)", 80.92, "CIFAR-100, scratch, 3.76M, 300 ep", "https://arxiv.org/html/2104.05704v4", "REF", "80.92"),
    "cct5k_c100": ("CCT-7/3x1, 5000 epochs (hybrid)", 82.87, "CIFAR-100, scratch, 3.76M", "https://arxiv.org/html/2104.05704v4", "REF", "82.87"),
    "asam_c100": ("PyramidNet-272 + ShakeDrop + AA + CutMix + ASAM", 89.90, "CIFAR-100 from-scratch SOTA, 26.0M, 900 ep", "https://arxiv.org/html/2102.11600", "C100", "89.90"),
    "effl2sam_c100": ("EfficientNet-L2 + SAM (JFT pretraining, 475 px)", 96.08, "CIFAR-100 pretrained SOTA, 480M", "https://arxiv.org/html/2010.01412", "C100", "96.08"),
    "effl2_c100": ("EfficientNet-L2 without SAM", 95.93, "CIFAR-100 pretrained, 480M", "https://arxiv.org/html/2010.01412", "C100", "95.93"),
    "effl2_params": ("EfficientNet-L2 parameter count (Noisy Student)", 480, "params (M)", "https://arxiv.org/html/1911.04252", "C100", "480M"),
    "vith_c100": ("ViT-H/14, JFT-300M, 384 px", 94.55, "CIFAR-100 pretrained, 632M", "https://arxiv.org/html/2010.11929", "C100", "94.55"),
    "effb0_c100": ("EfficientNet-B0, IN-1k", 88.1, "CIFAR-100 pretrained, 4M", "https://arxiv.org/html/1905.11946", "C100", "88.1"),
    "mnv3_c100": ("MobileNetV3-L, IN-21k-P, 224 px", 88.5, "CIFAR-100 pretrained, 4.2M", "https://arxiv.org/html/2104.10972", "C100", "88.5"),
    "deitt_c100": ("DeiT-Tiny fine-tune, 224 px (ToE, 1000 ep)", 86.78, "CIFAR-100 pretrained, 5.5M", "https://arxiv.org/html/2404.00672", "C100", "86.78"),
    "effv2s_c100": ("EfficientNetV2-S, IN-1k", 91.5, "CIFAR-100 pretrained, 24M", "https://arxiv.org/html/2104.00298", "C100", "91.5"),
    "deith_c100": ("DeiT-Ti + hard distillation from ResNet-56", 73.25, "CIFAR-100, 300 ep, 32 px teacher", "https://arxiv.org/abs/2207.10026", "IMP", "73.25"),
    "deitb_c100": ("DeiT-Ti baseline in the same study", 65.08, "CIFAR-100, 300 ep", "https://arxiv.org/abs/2207.10026", "IMP", "65.08"),
    "vit1216_c10": ("ViT-12/16 (CCT paper)", 83.04, "CIFAR-10, scratch, 85.6M, patch 16", "https://arxiv.org/html/2104.05704v4", "REF", "83.04"),
    "slvit_tin": ("SL-ViT", 61.07, "Tiny-ImageNet, scratch", "https://arxiv.org/html/2112.13492", "IMP", "61.07"),
    "spt_gain_c100": ("SPT alone on top of plain ViT", 76.29, "CIFAR-100, 2.8M, 100 ep", "https://arxiv.org/html/2112.13492", "IMP", "76.29"),
}
SRC_TXT = {k: (p.read_text() if p.exists() else "") for k, p in SRC_FILES.items()}
PUB_MISSING = [k for k, v in PUB.items() if v[5] not in SRC_TXT.get(v[4], "")]
USED_PUB = []


def P(key):
    """Published value, registered for the Sources section."""
    if key not in USED_PUB:
        USED_PUB.append(key)
    return PUB[key][1]


def cite(key, text=None):
    lab, val, setting, url, f, _ = PUB[key]
    P(key)
    return f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(text or lab)}</a>'


# ------------------------------------------------------------------------------------------ helpers
RUNS_D = load_runs()
FLAGS = parse_scripts()


def R(name):
    return RUNS_D.get(name, {}).get("res")


def acc(name, key="final_val_acc"):
    r = R(name)
    return None if not r or r.get(key) is None else float(r[key])


def A(name, key="final_val_acc", nd=2):
    v = acc(name, key)
    return '<span class="pend">pending</span>' if v is None else f"{v:.{nd}f}"


def num(v, nd=2, sign=False):
    if v is None:
        return '<span class="pend">n/a</span>'
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def D(a, b, nd=2):
    """Signed difference acc(a) - acc(b) as text."""
    x, y = acc(a), acc(b)
    return '<span class="pend">pending</span>' if x is None or y is None else f"{x - y:+.{nd}f}"


def flag_of(name):
    if name in FLAGS:
        return FLAGS[name][1]
    r = RUNS_D.get(name)
    return flags_from_args(r) if r and r.get("res") else None


def code_flags(*names):
    rows = []
    for n in names:
        f = flag_of(n)
        if f:
            rows.append(esc(f))
    if not rows:
        return ""
    return '<pre class="cli"><code>' + "\n".join(rows) + "</code></pre>"


def fmt_t(t, fmt="%b %d %H:%M"):
    return time.strftime(fmt, time.localtime(t)) if t else "?"


def chapter_date(names):
    ts = [RUNS_D[n]["start"] for n in names if n in RUNS_D and RUNS_D[n].get("start")]
    return fmt_t(min(ts)) if ts else "pending"


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else None), len(vals)


def pill(ours, ref, tol=0.5):
    if ours is None:
        return '<span class="pill pend-pill">pending</span>'
    d = ours - ref
    if d > tol:
        return f'<span class="pill beat">beat {d:+.2f}</span>'
    if d >= -tol:
        return f'<span class="pill match">matched {d:+.2f}</span>'
    return f'<span class="pill below">below {d:+.2f}</span>'


def table(headers, rows, num_cols=(), cls="", caption=None):
    h = [f'<div class="tw"><table class="{cls}">']
    if caption:
        h.append(f"<caption>{caption}</caption>")
    h.append("<thead><tr>" + "".join(f'<th{" class=n" if i in num_cols else ""}>{c}</th>' for i, c in enumerate(headers)) + "</tr></thead><tbody>")
    for r in rows:
        h.append("<tr>" + "".join(f'<td{" class=n" if i in num_cols else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>")
    h.append("</tbody></table></div>")
    return "".join(h)


# ------------------------------------------------------------------------------------------- SVG charts
FALLBACK = {"vit": "#2a78d6", "hyb": "#eb6834", "cnn": "#1baf7a", "ours": "#6250d6", "mute": "#8a96a3",
            "ink": "#15202b", "ink2": "#4f5d6b", "grid": "#dfe5ea", "ref": "#9a6700", "bad": "#c62828", "good": "#1a7f37"}


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


def log_ticks(lo, hi):
    out = []
    for e in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo <= v <= hi:
                out.append(v)
    return out


def tfmt(v):
    if v >= 100 or abs(v - round(v)) < 1e-9:
        return f"{v:.0f}"
    return f"{v:g}"


class Chart:
    def __init__(self, w, h, label, ml=56, mr=24, mt=16, mb=44):
        self.w, self.h, self.ml, self.mr, self.mt, self.mb = w, h, ml, mr, mt, mb
        self.parts = []
        self.label = label

    def setx(self, lo, hi, log=False):
        self.xlo, self.xhi, self.xlog = lo, hi, log

    def sety(self, lo, hi):
        self.ylo, self.yhi = lo, hi

    def X(self, v):
        a, b = (math.log10(self.xlo), math.log10(self.xhi)) if self.xlog else (self.xlo, self.xhi)
        v = math.log10(max(v, 1e-12)) if self.xlog else v
        return self.ml + (v - a) / (b - a) * (self.w - self.ml - self.mr)

    def Y(self, v):
        return self.mt + (self.yhi - v) / (self.yhi - self.ylo) * (self.h - self.mt - self.mb)

    def axes(self, xlabel, ylabel, xticks=None, yticks=None):
        xt = xticks or (log_ticks(self.xlo, self.xhi) if self.xlog else nice_ticks(self.xlo, self.xhi, 6))
        yt = yticks or nice_ticks(self.ylo, self.yhi, 5)
        p = self.parts
        for t in yt:
            y = self.Y(t)
            p.append(f'<line class="grid" x1="{self.ml}" x2="{self.w - self.mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="{FALLBACK["grid"]}" fill="none"/>')
            p.append(f'<text class="tick" x="{self.ml - 6}" y="{y + 4:.1f}" text-anchor="end" fill="{FALLBACK["ink2"]}">{tfmt(t)}</text>')
        yb = self.h - self.mb
        p.append(f'<line class="axis" x1="{self.ml}" x2="{self.w - self.mr}" y1="{yb}" y2="{yb}" stroke="{FALLBACK["ink2"]}" fill="none"/>')
        for t in xt:
            x = self.X(t)
            p.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{yb}" y2="{yb + 4}" stroke="{FALLBACK["ink2"]}" fill="none"/>')
            p.append(f'<text class="tick" x="{x:.1f}" y="{yb + 17}" text-anchor="middle" fill="{FALLBACK["ink2"]}">{tfmt(t)}</text>')
        p.append(f'<text class="axl" x="{(self.ml + self.w - self.mr) / 2:.1f}" y="{self.h - 6}" text-anchor="middle" fill="{FALLBACK["ink2"]}">{esc(xlabel)}</text>')
        p.append(f'<text class="axl" transform="translate(13,{(self.mt + yb) / 2:.1f}) rotate(-90)" text-anchor="middle" fill="{FALLBACK["ink2"]}">{esc(ylabel)}</text>')

    def hline(self, v, label, cls="ref", dash="5 4", anchor="end", below=False):
        if not (self.ylo <= v <= self.yhi):
            return
        y = self.Y(v)
        x0, x1 = self.ml, self.w - self.mr
        self.parts.append(f'<line class="hl {cls}" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke-dasharray="{dash}" stroke="{FALLBACK["ref"]}" fill="none"/>')
        tx = x1 - 4 if anchor == "end" else x0 + 4
        ty = y + 13 if below else y - 4
        self.parts.append(f'<text class="hlt {cls}" x="{tx}" y="{ty:.1f}" text-anchor="{anchor}" fill="{FALLBACK["ink2"]}">{esc(label)}</text>')

    def vline(self, v, label):
        x = self.X(v)
        self.parts.append(f'<line class="hl ref" x1="{x:.1f}" x2="{x:.1f}" y1="{self.mt}" y2="{self.h - self.mb}" stroke-dasharray="3 3" stroke="{FALLBACK["ref"]}" fill="none"/>')
        self.parts.append(f'<text class="hlt" x="{x + 5:.1f}" y="{self.h - self.mb - 6}" fill="{FALLBACK["ink2"]}">{esc(label)}</text>')

    def line(self, xs, ys, cls, name=None, label_end=True, marker=True, dash=None):
        pts = [(self.X(x), self.Y(y)) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            return
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path class="ln {cls}" d="{d}" fill="none" stroke="{FALLBACK.get(cls, "#555")}"{da}><title>{esc(name or "")}</title></path>')
        if marker:
            x, y = pts[-1]
            self.parts.append(f'<circle class="pt {cls}" cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{FALLBACK.get(cls, "#555")}"/>')
        if label_end and name:
            x, y = pts[-1]
            self.parts.append(f'<text class="dl" x="{x + 7:.1f}" y="{y + 4:.1f}" fill="{FALLBACK["ink"]}">{esc(name)}</text>')

    def dot(self, x, y, cls, tip, shape="circle", hollow=False, r=5):
        cx, cy = self.X(x), self.Y(y)
        fill = "none" if hollow else FALLBACK.get(cls, "#555")
        hcls = " hollow" if hollow else ""
        if shape == "diamond":
            s = r + 1.5
            el = f'<path class="pt {cls}{hcls}" d="M{cx:.1f},{cy - s:.1f} L{cx + s:.1f},{cy:.1f} L{cx:.1f},{cy + s:.1f} L{cx - s:.1f},{cy:.1f} Z" fill="{fill}" stroke="{FALLBACK.get(cls)}" stroke-width="2">'
        elif shape == "square":
            el = f'<rect class="pt {cls}{hcls}" x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r}" height="{2 * r}" rx="1.5" fill="{fill}" stroke="{FALLBACK.get(cls)}" stroke-width="2">'
        else:
            el = f'<circle class="pt {cls}{hcls}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{FALLBACK.get(cls)}" stroke-width="2">'
        tag = el.split(" ", 1)[0][1:]
        self.parts.append(el + f"<title>{esc(tip)}</title></{tag}>")

    def text(self, x, y, s, anchor="start", cls="dl", dx=0, dy=0):
        self.parts.append(f'<text class="{cls}" x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" text-anchor="{anchor}" fill="{FALLBACK["ink"]}">{esc(s)}</text>')

    def svg(self):
        return (f'<svg class="chart" viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{esc(self.label)}" '
                f'xmlns="http://www.w3.org/2000/svg"><title>{esc(self.label)}</title>' + "".join(self.parts) + "</svg>")


def legend(items):
    """items: [(cls, label, shape)] -> HTML legend row (identity never by color alone: shapes + labels)."""
    out = ['<div class="legend">']
    for cls, lab, shape in items:
        out.append(f'<span><i class="sw {cls} {shape}"></i>{esc(lab)}</span>')
    out.append("</div>")
    return "".join(out)


def hbars(items, label, xlo=None, xhi=None, refs=(), unit="", w=640, nd=2, lw=190):
    """items: [(label, value, cls, tip, ghost)]: ghost = optional second value drawn as an outline bar
    (clean-train accuracy). Missing values render as 'pending'."""
    vals = [v for _, v, *_ in items if v is not None] + [g for *_, g in items if g is not None]
    vals += [r[0] for r in refs]
    lo = xlo if xlo is not None else 0
    hi = xhi if xhi is not None else (max(vals) * 1.05 if vals else 1)
    rowh = 26
    h = 26 + rowh * len(items) + 30
    ch = Chart(w, h, label, ml=lw, mr=58, mt=16, mb=30)
    ch.setx(lo, hi)
    ch.sety(0, 1)
    p = ch.parts
    yb = h - ch.mb
    for t in nice_ticks(lo, hi, 5):
        x = ch.X(t)
        p.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{ch.mt}" y2="{yb}" stroke="{FALLBACK["grid"]}" fill="none"/>')
        p.append(f'<text class="tick" x="{x:.1f}" y="{yb + 16}" text-anchor="middle" fill="{FALLBACK["ink2"]}">{tfmt(t)}{unit}</text>')
    for i, (lab, v, cls, tip, ghost) in enumerate(items):
        y = ch.mt + 6 + i * rowh
        p.append(f'<text class="bl" x="{ch.ml - 8}" y="{y + 13}" text-anchor="end" fill="{FALLBACK["ink"]}">{esc(lab)}</text>')
        x0 = ch.X(lo)
        if ghost is not None:
            gx = ch.X(min(max(ghost, lo), hi))
            p.append(f'<rect class="ghost {cls}" x="{x0:.1f}" y="{y + 1}" width="{max(gx - x0, 1):.1f}" height="16" rx="3" fill="none" stroke="{FALLBACK.get(cls)}" stroke-width="1.5"><title>{esc(lab)}: clean train {ghost:.2f}</title></rect>')
        if v is None:
            p.append(f'<text class="pendt" x="{x0 + 4:.1f}" y="{y + 13}" fill="{FALLBACK["ink2"]}">pending</text>')
            continue
        xv = ch.X(min(max(v, lo), hi))
        p.append(f'<rect class="bar {cls}" x="{x0:.1f}" y="{y + 4}" width="{max(xv - x0, 1.5):.1f}" height="10" rx="2" fill="{FALLBACK.get(cls)}"><title>{esc(tip or lab)}: {v:.{nd}f}{unit}</title></rect>')
        tx = max(xv, gx if ghost is not None else xv)
        p.append(f'<text class="bv" x="{tx + 5:.1f}" y="{y + 13}" fill="{FALLBACK["ink"]}">{v:.{nd}f}{unit}</text>')
    for v, lab in refs:
        if lo <= v <= hi:
            x = ch.X(v)
            p.append(f'<line class="hl ref" x1="{x:.1f}" x2="{x:.1f}" y1="{ch.mt - 4}" y2="{yb}" stroke-dasharray="4 3" stroke="{FALLBACK["ref"]}" fill="none"/>')
            p.append(f'<text class="hlt" x="{x:.1f}" y="{ch.mt - 6 + 0:.1f}" text-anchor="middle" fill="{FALLBACK["ink2"]}">{esc(lab)}</text>')
    return ch.svg()


def curves(names, label, cls_map=None, ylo=None, yhi=None, w=640, h=280, key="val_acc", labels=None):
    series = []
    for i, n in enumerate(names):
        r = R(n)
        if not r:
            continue
        pts = [(e["epoch"], e[key]) for e in r["history"] if e.get(key) is not None]
        if pts:
            series.append((n, pts))
    if not series:
        return '<p class="pend">No finished runs to plot yet.</p>'
    ys = [y for _, pts in series for _, y in pts]
    xs = [x for _, pts in series for x, _ in pts]
    lo = ylo if ylo is not None else math.floor(min(ys) / 5) * 5
    hi = yhi if yhi is not None else math.ceil(max(ys) / 5) * 5
    ch = Chart(w, h, label, mr=150)
    ch.setx(0, max(xs))
    ch.sety(lo, hi)
    ch.axes("epoch", "val acc (%)")
    ends = []
    for i, (n, pts) in enumerate(series):
        cls = (cls_map or {}).get(n, ["vit", "hyb", "cnn", "ours", "mute"][i % 5])
        ch.line([p[0] for p in pts], [p[1] for p in pts], cls, name=None, label_end=False)
        ends.append([ch.Y(pts[-1][1]), (labels or {}).get(n, n), pts[-1][1], cls])
    ends.sort()
    for i in range(1, len(ends)):  # de-overlap direct labels
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
    for y, lab, v, cls in ends:
        ch.parts.append(f'<text class="dl" x="{ch.w - ch.mr + 8}" y="{y + 4:.1f}" fill="{FALLBACK["ink"]}">{esc(lab)} {v:.1f}</text>')
    return ch.svg()


def fig(svg, caption, legend_html=""):
    return f'<figure class="fig">{legend_html}<div class="svgw">{svg}</div><figcaption>{caption}</figcaption></figure>'


def png_uri(path, max_w=900):
    """Embed a PNG as a downscaled data URI (keeps the page small)."""
    try:
        from PIL import Image
        im = Image.open(path)
        if im.width > max_w:
            im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# --------------------------------------------------------------------------------- run groups and families
def dataset_of(name):
    r = R(name)
    if r:
        return r.get("args", {}).get("dataset", "?")
    if name.startswith(("c100_", "ec100_")):
        return "cifar100"
    if name.startswith("tin_"):
        return "tiny_imagenet"
    if name.startswith(("in64", "in64pre")):
        return "imagenet64"
    return "cifar10"


EDGE_FAMILY = {  # short name stem -> (family class, display)
    "deit_t": ("vit", "DeiT-Ti"), "vit_t_ar": ("vit", "ViT-Ti AugReg"), "T_vitb_ar2": ("vit", "ViT-B/16 teacher"),
    "T_eva02s": ("vit", "EVA02-S teacher"), "tinyvit_5m": ("hyb", "TinyViT-5M"), "iformer_t": ("hyb", "iFormer-T"),
    "effformv2_s0": ("hyb", "EffFormerV2-S0"), "fastvit_t8": ("hyb", "FastViT-T8"), "effvit_m0": ("hyb", "EfficientViT-M0"),
    "effvit_m1": ("hyb", "EfficientViT-M1"), "mvitv2_075": ("hyb", "MobileViTv2-0.75"), "mvit_xs": ("hyb", "MobileViT-XS"),
    "edgenext_xs": ("hyb", "EdgeNeXt-XS"), "mnv4_hm": ("hyb", "MNv4-HybridM"), "starnet_s1": ("cnn", "StarNet-S1"),
    "mnv4_cs": ("cnn", "MNv4-ConvS"), "repvit_m0_9": ("cnn", "RepViT-M0.9"), "convnext_at": ("cnn", "ConvNeXt-Atto"),
    "in64pre_slvit": ("ours", "our SL-ViT, IN-64 pre"), "in64pre_tiny": ("ours", "our ViT-Tiny, IN-64 pre"),
    "in64pre_tiny_p4": ("ours", "our ViT-Tiny p4 (FlexiViT switch), IN-64 pre"),
    "upcyc_in64pre_tiny_e4": ("ours", "our ViT-Tiny MoE-upcycled E4, IN-64 pre"),
    "dense_in64pre_tiny": ("ours", "our ViT-Tiny, IN-64 pre (dense control)"),
    "moe_repvit_k4": ("cnn", "RepViT-M0.9 CondConv K4"), "moe_repvit_k8": ("cnn", "RepViT-M0.9 CondConv K8"),
    "fm_repvit": ("cnn", "RepViT-M0.9 function-matching KD"), "fm_tinyvit5m": ("hyb", "TinyViT-5M function-matching KD"),
    "kd60_soft_repvit": ("cnn", "RepViT-M0.9 KD soft 60ep"), "kd60_dist_repvit": ("cnn", "RepViT-M0.9 KD DIST 60ep"),
    "kd60_lsd_repvit": ("cnn", "RepViT-M0.9 KD logit-std 60ep"),
}


def edge_info(name):
    s = name[len("ec100_"):]
    kd = None
    if s.startswith("kdhard_"):
        kd, s = "hard", s[7:]
    elif s.startswith("kd_"):
        kd, s = "soft", s[3:]
    m = re.match(r"(.+?)_(\d+)(_kd)?(_lr[^_]*)?(?:_e\d+|_\d+ep)?$", s)
    stem, px = (m.group(1), int(m.group(2))) if m else (s, None)
    if m and m.group(3):
        kd = "soft"
    fam, disp = EDGE_FAMILY.get(stem, ("mute", stem))
    teacher = stem.startswith("T_")
    lr = m.group(4)[3:] if m and m.group(4) else None
    # lr-sweep runs are variants, not the default-recipe run; the MNv4-HybridM lr 1e-4 run replaces the diverged lr 5e-4 default
    variant = bool(lr) and not (stem == "mnv4_hm")
    if lr:
        disp += f" (lr {lr})"
    return dict(stem=stem, px=px, kd=kd, fam=fam, disp=disp, teacher=teacher, lr=lr, variant=variant, superseded=False)


EDGE = []
SUPERSEDED = {"ec100_mnv4_hm_128": "ec100_mnv4_hm_128_lr1e-4", "ec100_mnv4_hm_224": "ec100_mnv4_hm_224_lr1e-4"}
for n, r in RUNS_D.items():
    if not n.startswith("ec100_"):
        continue
    info = edge_info(n)
    info["name"] = n
    info["status"] = r["status"]
    info["superseded"] = n in SUPERSEDED and SUPERSEDED[n] in RUNS_D and bool(RUNS_D[SUPERSEDED[n]]["res"])
    res = r["res"]
    if res:
        info["acc"] = res["final_val_acc"]
        info["best"] = res.get("best_val_acc")
        info["ema"] = res.get("best_ema_val_acc")
        info["params"] = res.get("params_M")
        info["minutes"] = res.get("total_minutes")
        info["collapsed"] = res["final_val_acc"] < 20
    eb = r.get("edge")
    if eb:
        info["gmacs"] = eb.get("GMACs")
        info["params_b"] = eb.get("params_M")
        info["int8_t1"] = eb.get("ort_lat_ms_int8_t1")
        info["int8_t4"] = eb.get("ort_lat_ms_int8_t4")
        info["fp32_t1"] = eb.get("ort_lat_ms_fp32_t1")
        info["fp32_t4"] = eb.get("ort_lat_ms_fp32_t4")
        info["act"] = eb.get("act_MB_est_bs1_fp32")
    EDGE.append(info)

TEACHERS = [e for e in EDGE if e["teacher"] and e.get("acc")]
TEACH_ACC = max((e["acc"] for e in TEACHERS), default=None)
TEACH = max(TEACHERS, key=lambda e: e["acc"]) if TEACHERS else None
STUDENTS = [e for e in EDGE if not e["teacher"] and e.get("acc") is not None and not e.get("collapsed") and not e["superseded"]]
SMALL = [e for e in STUDENTS if (e.get("params") or 99) <= 6.0]
BEST_SMALL = max(SMALL, key=lambda e: e["acc"]) if SMALL else None
SOTA_PRE = P("effl2sam_c100")
SOTA_SCR = P("asam_c100")


# ------------------------------------------------------------------------------------------------ cost
def cost_summary():
    done = [r for r in RUNS_D.values() if r["status"] == "done"]
    run_min = sum(r["res"].get("total_minutes", 0) for r in done)
    live = [r for r in RUNS_D.values() if r["status"] in ("running", "stalled")]
    live_min = sum(max(0, (r["end"] - r["start"]) / 60) for r in live)
    iv = sorted((r["start"], r["end"]) for r in RUNS_D.values() if r.get("start") and r.get("end"))
    busy, cur = 0.0, None
    for s, e in iv:
        if cur is None or s > cur[1]:
            if cur:
                busy += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur:
        busy += cur[1] - cur[0]
    span = (iv[-1][1] - iv[0][0]) if iv else 0
    by_ds = {}
    for r in done:
        ds = "edge (CIFAR-100 fine-tune)" if r["name"].startswith("ec100_") else dataset_of(r["name"])
        by_ds.setdefault(ds, [0, 0.0])
        by_ds[ds][0] += 1
        by_ds[ds][1] += r["res"].get("total_minutes", 0)
    return dict(run_min=run_min, live_min=live_min, busy_min=busy / 60, span_min=span / 60, first=iv[0][0] if iv else None,
                last=iv[-1][1] if iv else None, by_ds=by_ds, n_done=len(done), n_live=len(live))


COST = cost_summary()


def git_dates():
    try:
        out = subprocess.run(["git", "log", "--format=%ad|%s", "--date=iso"], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout
        lines = [l.split("|", 1) for l in out.strip().splitlines() if "|" in l]
        return lines[-1][0][:16], lines[0][0][:16], len(lines)
    except Exception:
        return "?", "?", 0


GIT_FIRST, GIT_LAST, GIT_N = git_dates()


# ======================================================================================== DASHBOARD section
def scoreboard():
    rows = []

    def row(setting, fam, ours_lab, ours, ref_key, extra=""):
        ref = P(ref_key)
        lab, _, rset, url, *_ = PUB[ref_key]
        pct = f"{100 * ours / ref:.1f}%" if ours is not None else "—"
        rows.append([setting, fam, ours_lab + extra, num(ours), f'{cite(ref_key)} <span class="sub">{esc(rset)}</span>', f"{ref:.2f}", pct, pill(ours, ref)])

    b3 = acc("bf3b_slvit_300ep_lr5e-4")
    ts = acc("abl_tiny_shape")
    for k in ("vitlite_c10", "sptvit_c10", "slvit_c10"):
        row("CIFAR-10", "pure ViT, scratch", "bf3b (SL-ViT shape, 2.7M)", b3, k)
    row("CIFAR-10", "pure ViT, scratch", "abl_tiny_shape (ViT-Tiny shape, 5.4M)", ts, "slvit_c10")
    row("CIFAR-10", "hybrid, scratch", "not run (our best pure ViT shown)", ts, "cct_c10")
    row("CIFAR-10", "CNN control", "r18_full (100 ep, batch 1024)", acc("r18_full"), "r18cutout_c10")
    c3 = acc("c100_bf3b_slvit_300ep_lr5e-4")
    for k in ("sptvit_c100", "vitlite_c100", "slvit_c100"):
        row("CIFAR-100", "pure ViT, scratch", "c100_bf3b (SL-ViT shape)", c3, k)
    ct = acc("c100_tinyshape_bf3b")
    row("CIFAR-100", "pure ViT, scratch", "c100_tinyshape_bf3b (ViT-Tiny shape)", ct, "slvit_c100")
    row("CIFAR-100", "hybrid, scratch", "not run (best pure ViT shown)", max([v for v in (c3, ct) if v is not None], default=None), "cct_c100")
    row("CIFAR-100", "any, scratch", "best from-scratch run", max([v for v in (c3, ct, acc("c100_r18_full")) if v is not None], default=None), "asam_c100")
    if TEACH:
        row("CIFAR-100", "pretrained", f"teacher {TEACH['disp']} @224 ({TEACH['params']:.0f}M)", TEACH["acc"], "effl2sam_c100")
    if BEST_SMALL:
        kd = " + KD" if BEST_SMALL["kd"] else ""
        row("CIFAR-100", "pretrained, ≤6M", f"{BEST_SMALL['disp']}{kd} @{BEST_SMALL['px']} ({BEST_SMALL['params']:.1f}M)", BEST_SMALL["acc"], "effl2sam_c100")
        row("CIFAR-100", "pretrained, ≤6M", f"{BEST_SMALL['disp']}{kd} @{BEST_SMALL['px']}", BEST_SMALL["acc"], "effb0_c100")
    return table(["dataset", "setting", "ours", "acc", "published", "acc", "% of it", "status"], rows, num_cols=(3, 5, 6), cls="score")


def pareto_chart(xkey, xlabel, title):
    pts = [e for e in EDGE if e.get("acc") is not None and not e.get("collapsed") and not e["superseded"] and e.get(xkey)]
    if not pts:
        return '<p class="pend">No benchmarked edge runs yet.</p>'
    xs = [e[xkey] for e in pts]
    lo, hi = min(xs) / 1.4, max(xs) * 1.6
    ylo = math.floor(min(70, min(e["acc"] for e in pts)) / 2) * 2
    yhi = 97
    ch = Chart(680, 360, title, mr=24, mt=14)
    ch.setx(lo, hi, log=True)
    ch.sety(ylo, yhi)
    ch.axes(xlabel + " (log scale)", "CIFAR-100 val acc (%)")
    if TEACH_ACC:
        ch.hline(0.99 * TEACH_ACC, f"99% of our teacher ({0.99 * TEACH_ACC:.2f})", anchor="start")
    ch.hline(0.95 * SOTA_PRE, f"95% of SOTA {SOTA_PRE} ({0.95 * SOTA_PRE:.2f})", dash="2 3", anchor="start", below=True)
    ch.hline(0.90 * SOTA_PRE, f"90% of SOTA ({0.90 * SOTA_PRE:.2f})", dash="2 3", anchor="start")
    # Pareto frontier (lower cost, higher acc)
    front, best = [], -1
    for e in sorted(pts, key=lambda e: (e[xkey], -e["acc"])):
        if e["acc"] > best:
            front.append(e)
            best = e["acc"]
    ch.line([e[xkey] for e in front], [e["acc"] for e in front], "mute", marker=False, label_end=False, dash="1 0")
    shapes = {"vit": "circle", "hyb": "square", "cnn": "diamond", "ours": "circle"}
    for e in pts:
        kd = f", KD {e['kd']}" if e["kd"] else ""
        tip = f"{e['name']}: {e['disp']} @{e['px']}px{kd}; {e['acc']:.2f}%; {e[xkey]:.3g} {xlabel}"
        ch.dot(e[xkey], e["acc"], e["fam"], tip, shape=shapes.get(e["fam"], "circle"), hollow=e["teacher"] or e["px"] == 128, r=5)
    # reserved boxes: reference-line labels on the left
    boxes = []
    for v, below in ((0.99 * TEACH_ACC if TEACH_ACC else None, False), (0.95 * SOTA_PRE, True), (0.90 * SOTA_PRE, False)):
        if v is not None and ch.ylo <= v <= ch.yhi:
            yy = ch.Y(v) + (13 if below else -4)
            boxes.append((ch.ml, ch.ml + 260, yy - 10, yy + 2))
    for e in pts:  # markers
        x, y = ch.X(e[xkey]), ch.Y(e["acc"])
        boxes.append((x - 6, x + 6, y - 6, y + 6))
    ours = [e for e in pts if e["fam"] == "ours"]
    extra = [max(ours, key=lambda e: e["acc"])] if ours else []
    for e in sorted(front + [x for x in extra if x not in front], key=lambda e: -e["acc"]):
        x, y = ch.X(e[xkey]), ch.Y(e["acc"])
        lab = ("\u2605 best \u22646M: " if BEST_SMALL and e["name"] == BEST_SMALL["name"] else "") + f"{e['disp']}{' KD' if e['kd'] else ''} @{e['px']}"
        wlab = 6.8 * len(lab)
        best = None
        for dy in (4, -8, 16, -20, 28, -32, 40, -44, 52):
            for side in (("start", "end") if x < ch.w - wlab - 40 else ("end", "start")):
                x0 = x + 9 if side == "start" else x - 9 - wlab
                if x0 < ch.ml or x0 + wlab > ch.w - 4:
                    continue
                bb = (x0, x0 + wlab, y + dy - 10, y + dy + 2)
                if not any(bb[0] < b[1] and b[0] < bb[1] and bb[2] < b[3] and b[2] < bb[3] for b in boxes if b != (x - 6, x + 6, y - 6, y + 6)):
                    best = (side, dy, bb)
                    break
            if best:
                break
        if not best:
            continue  # no free spot: the tooltip still names it
        side, dy, bb = best
        boxes.append(bb)
        ch.parts.append(f'<text class="dl" x="{x + (9 if side == "start" else -9):.1f}" y="{y + dy:.1f}" text-anchor="{side}" fill="{FALLBACK["ink"]}">{esc(lab)}</text>')
    return ch.svg()


PARETO_LEGEND = legend([("vit", "plain ViT", "circle"), ("hyb", "hybrid conv+attention", "square"), ("cnn", "CNN", "diamond"),
                        ("mute", "hollow = 128 px input or teacher; filled = 224 px student", "ring"),
                        ("mute", "Pareto frontier", "line"), ("ref", "reference lines", "dash")])


def seed_table():
    groups = [("bf3b (SL-ViT, CIFAR-10)", ["bf3b_slvit_300ep_lr5e-4", "bf3b_seed1", "bf3b_seed2"]),
              ("bf4 (lighter mixing, CIFAR-10)", ["bf4_slvit_300ep_mixhalf", "bf4_seed1", "bf4_seed2"]),
              ("rd_deep_r8_add_ds (recurrent, CIFAR-10)", ["rd_deep_r8_add_ds", "rd_deep_r8_add_ds_seed1", "rd_deep_r8_add_ds_seed2"]),
              ("abl_tiny_shape (ViT-Tiny shape, CIFAR-10)", ["abl_tiny_shape", "abl_tiny_shape_seed1", "abl_tiny_shape_seed2"])]
    rows = []
    for lab, names in groups:
        vals = [acc(n) for n in names]
        m, s, k = mean_std(vals)
        rows.append([lab] + [num(v) if v is not None else '<span class="pend">pending</span>' for v in vals]
                    + [f"{m:.2f}" if m is not None else "—", f"± {s:.2f}" if s is not None else "—", str(k)])
    return table(["recipe", "seed 0", "seed 1", "seed 2", "mean", "std (n-1)", "n"], rows, num_cols=(1, 2, 3, 4, 5, 6))


def ds_table(ds, prefix_ok=lambda n: True):
    rows = []
    for n, r in RUNS_D.items():
        if n.startswith("ec100_") or not prefix_ok(n):
            continue
        if dataset_of(n) != ds:
            continue
        res = r["res"]
        if res:
            best = res.get("best_val_acc")
            v = f"{res['final_val_acc']:.2f}" + (f' <span class="sub">({best:.2f})</span>' if best and abs(best - res["final_val_acc"]) > 0.05 else "")
            rows.append((r["start"] or 0, [f"<code>{n}</code>", f"{res['params_M']:.2f}", str(len(res["history"])), v,
                                           num(res.get("final_train_acc_clean")), f"{res['total_minutes']:.1f}", f"{res.get('img_per_sec_steady', 0):,.0f}"]))
        else:
            pr = r.get("progress")
            st = f"{r['status']}" + (f", epoch {pr[0]}/{pr[1]}" if pr else "")
            rows.append((r.get("start") or NOW, [f"<code>{n}</code>", "—", "—", f'<span class="pill pend-pill">{st}</span>', "—", "—", "—"]))
    rows.sort(key=lambda x: x[0])
    return table(["run", "params (M)", "epochs", "val acc (best)", "clean train", "minutes", "img/s"], [r for _, r in rows], num_cols=(1, 2, 3, 4, 5, 6))


def edge_table():
    rows = []
    for e in sorted(EDGE, key=lambda e: (e["teacher"] is False, -(e.get("acc") or 0))):
        if e.get("acc") is None:
            pr = RUNS_D[e["name"]].get("progress")
            st = e["status"] + (f", epoch {pr[0]}/{pr[1]}, loss {pr[2]:.2f}" if pr else "")
            rows.append([f"<code>{e['name']}</code>", e["disp"], fam_name(e["fam"]), str(e["px"]), e["kd"] or "—", "—", "—", "—", "—", f'<span class="pill pend-pill">{esc(st)}</span>', "—", "—", "—"])
            continue
        a = e["acc"]
        accs = f"{a:.2f}" + (' <span class="pill below">collapsed</span>' if e.get("collapsed") else "") + (' <span class="pill below">superseded: lr 5e-4 diverged</span>' if e["superseded"] else "")
        mark = (' <span class="pill match">best &le;6M</span>' if BEST_SMALL and e["name"] == BEST_SMALL["name"] else "") + \
               (' <span class="pill pend-pill">pretrained by us on ImageNet-64</span>' if e["fam"] == "ours" else "")
        rows.append([f"<code>{e['name']}</code>", e["disp"] + (" (teacher)" if e["teacher"] else "") + mark, fam_name(e["fam"]), str(e["px"]), e["kd"] or "—",
                     num(e.get("params"), 2), num(e.get("gmacs"), 3), num(e.get("int8_t1"), 1), num(e.get("fp32_t1"), 1), accs,
                     num(e.get("ema")), f"{100 * a / TEACH_ACC:.1f}%" if TEACH_ACC else "—", f"{100 * a / SOTA_PRE:.1f}%"])
    return table(["run", "model", "family", "px", "KD", "params (M)", "GMACs", "int8 ms (1 thr)", "fp32 ms (1 thr)", "val acc", "EMA acc", "% teacher", "% SOTA"],
                 rows, num_cols=(3, 5, 6, 7, 8, 9, 10, 11, 12))


def fam_name(f):
    return {"vit": "plain ViT", "hyb": "hybrid", "cnn": "CNN", "ours": "our ViT", "mute": "?"}[f]


def smallest_within():
    rows = []
    targets = []
    if TEACH_ACC:
        targets += [(f"99% of our teacher ({TEACH['disp']} {TEACH_ACC:.2f})", 0.99 * TEACH_ACC), (f"95% of our teacher", 0.95 * TEACH_ACC)]
    targets += [(f"99% of published SOTA ({SOTA_PRE})", 0.99 * SOTA_PRE), ("95% of published SOTA", 0.95 * SOTA_PRE), ("90% of published SOTA", 0.90 * SOTA_PRE)]
    for lab, thr in targets:
        ok = [e for e in STUDENTS if e["acc"] >= thr]
        if ok:
            e = min(ok, key=lambda e: e["params"])
            who = f"{e['disp']}{' + KD' if e['kd'] else ''} @{e['px']} px (<code>{e['name']}</code>)"
            rows.append([lab, f"{thr:.2f}", who, f"{e['params']:.2f}", f"{e['acc']:.2f}"])
        else:
            near = max(STUDENTS, key=lambda e: e["acc"]) if STUDENTS else None
            rows.append([lab, f"{thr:.2f}", "none of our students" + (f" (closest: {near['disp']} {near['acc']:.2f})" if near else ""), "—", "—"])
    return table(["target", "threshold", "smallest student that clears it", "params (M)", "acc"], rows, num_cols=(1, 3, 4))


def cost_block():
    c = COST
    usd = lambda m: m / 60 * PRICE
    rows = [[esc(k), str(v[0]), f"{v[1]:.0f}", f"{v[1] / 60:.2f}", f"${usd(v[1]):.2f}"] for k, v in sorted(c["by_ds"].items())]
    rows.append(["<b>sum of finished run-minutes</b>", f"<b>{c['n_done']}</b>", f"<b>{c['run_min']:.0f}</b>", f"<b>{c['run_min'] / 60:.2f}</b>", f"<b>${usd(c['run_min']):.2f}</b>"])
    rows.append([f"runs still going ({c['n_live']}), elapsed so far", str(c["n_live"]), f"{c['live_min']:.0f}", f"{c['live_min'] / 60:.2f}", f"${usd(c['live_min']):.2f}"])
    t = table(["group", "runs", "GPU-min", "GPU-h", f"$ at ${PRICE}/h"], rows, num_cols=(1, 2, 3, 4))
    note = (f"<p>The GPU is billed by wall-clock, not by run-minutes. From the first run start ({fmt_t(c['first'])}) to the latest "
            f"activity ({fmt_t(c['last'])}) is <b>{c['span_min'] / 60:.1f} h</b> (${usd(c['span_min']):.2f}). The GPU had at least one job "
            f"on it for <b>{c['busy_min'] / 60:.1f} h</b> (${usd(c['busy_min']):.2f}). The sum of run-minutes ({(c['run_min'] + c['live_min']) / 60:.1f} h) "
            f"is larger than both because up to four queues shared the GPU at once, so the run-minute total overstates the bill. "
            f"Run start times are estimated as results.json mtime minus run length.</p>")
    return t + note


def pending_block():
    live = [(n, r) for n, r in RUNS_D.items() if r["status"] in ("running", "stalled")]
    known = set(RUNS_D)
    queued = sorted(n for n in FLAGS if n not in known and not n.startswith("smoke"))
    items = []
    for n, r in sorted(live):
        pr = r.get("progress")
        s = f"<code>{n}</code>: {r['status']}" + (f", epoch {pr[0]}/{pr[1]}, loss {pr[2]:.2f}" if pr else "")
        items.append(f"<li>{s}</li>")
    for f in sorted(d.name for d in RUNS.iterdir() if d.is_dir() and d.name.startswith("_failed_")):
        n = f[len("_failed_"):]
        if n not in known:
            items.append(f"<li><code>{n}</code>: rerun queued after the fix (failed attempt kept in <code>runs/{f}</code>)</li>")
    for n in queued:
        items.append(f"<li><code>{n}</code>: queued in <code>{FLAGS[n][0]}</code>, not started</li>")
    if not items:
        return "<p>Every queued run has finished.</p>"
    return "<ul class=\"plist\">" + "".join(items) + "</ul>"


# ============================================================================================== STORY chapters
def chap(num_, cid, title, date, q, change, why, result, lesson, published):
    return f"""
<section class="chapter" id="{cid}" aria-labelledby="{cid}-h">
  <header class="ch-head"><span class="ch-num">{num_:02d}</span><div><h3 id="{cid}-h">{title}</h3><p class="ch-date">first run {date}</p></div></header>
  <div class="ch-grid">
    <div class="part q"><h4>Question</h4>{q}</div>
    <div class="part"><h4>What we changed</h4>{change}</div>
    <div class="part"><h4>Why it should matter</h4>{why}</div>
    <div class="part res"><h4>Result</h4>{result}</div>
    <div class="part"><h4>Lesson</h4>{lesson}</div>
    <div class="part pub"><h4>Against published results</h4>{published}</div>
  </div>
</section>"""


def ch1():
    names = ["t1_plain", "t5c_bs256", "c100_t1_plain", "bf3b_slvit_300ep_lr5e-4"]
    bench = hbars([("10 ep CIFAR-10, no graphs", 9.45, "mute", "s/epoch, batch 256, ViT-Tiny", None),
                   ("10 ep CIFAR-10, CUDA graphs", 3.48, "vit", "s/epoch, batch 256, ViT-Tiny", None),
                   ("bench_step b256, eager", 49.3, "mute", "ms/step", None),
                   ("bench_step b256, graphs", 17.5, "vit", "ms/step (16.6-17.5)", None),
                   ("bench_step b1024, eager", 50.6, "mute", "ms/step", None),
                   ("bench_step b1024, graphs", 32.1, "vit", "ms/step", None)], "Step time with and without CUDA graphs", xlo=0, xhi=55, nd=2, lw=200)
    notes_ok = all(s in SRC_TXT_NOTES for s in ("9.45 s/epoch", "3.48 s/epoch", "49.3", "50.6", "32.1", "226 W"))
    src = "REPORT_NOTES.md (measured)" + ("" if notes_ok else " <b>[warning: numbers not found in REPORT_NOTES.md]</b>")
    return chap(1, "ch1", "A fast pipeline for tiny models", chapter_date(names),
        "<p>A 5M-parameter ViT on 32×32 images finishes a forward and backward pass in milliseconds. Is the GPU doing the work, or is it waiting on the data loader and on Python?</p>",
        "<ul><li>The whole dataset lives on the GPU as a uint8 tensor, so no CPU data loader runs. Crop, flip, RandAugment, MixUp/CutMix and random erasing all run batched on the GPU.</li>"
        "<li>bf16 autocast and <code>torch.compile</code> (kernel fusion).</li><li><code>--cudagraphs</code>: compile with <code>mode=reduce-overhead</code>, which records the training step once and replays it.</li>"
        "<li>Up to four training queues share the one H100 at the same time.</li></ul>",
        "<p>Small models are <em>launch-bound</em>. Each CUDA kernel finishes faster than Python can issue the next one, so the GPU idles between kernels. "
        "A CUDA graph records the whole sequence of kernel launches once and replays it with a single launch, which removes that per-kernel CPU overhead. "
        "The win shrinks as batches grow, because big kernels hide the launch cost. For the same reason one small job leaves most of an 80 GB, 350 W GPU unused, so several jobs can run side by side.</p>",
        fig(bench, f"Lower is better. Grey = eager, blue = CUDA graphs. Source: {src}.") +
        "<p>On a real 10-epoch CIFAR-10 run (batch 256), CUDA graphs cut 9.45 to 3.48 s/epoch, a <b>2.7×</b> speedup, at the same accuracy (53.2% vs 53.0% val). "
        "At batch 1024 the gain is 1.6×. With three concurrent jobs, each job's epoch got 9 to 55% slower, but total throughput roughly doubled. The GPU was using 9 of 80 GB and 226 of 350 W.</p>"
        + (f"<p>Measured cost of skipping graphs: <code>t5c_bs256</code> ran ungraphed at {RUNS_D['t5c_bs256']['res']['img_per_sec_steady']:,.0f} img/s and took {RUNS_D['t5c_bs256']['res']['total_minutes']:.1f} min.</p>" if R("t5c_bs256") else ""),
        "<p>Profile before tuning the model. For small vision transformers, the fixed cost of launching kernels can dominate. Removing it bought 2.7× on every later experiment, and parallel queues bought another ~2×.</p>",
        f"<p>Our pipeline is built for many short ablations, not for records. {cite('airbench')} reaches 94% on CIFAR-10 in 3.29 s with a small CNN. Our best ViT needs about 40 to 80 minutes for 93 to 94.5%.</p>")


def ch2():
    lad = ["t1_plain", "t2_ls_wd", "t3_randaug", "t4_mix", "t5_droppath", "t5b_mix_half", "t5c_bs256", "t6_full_300ep"]
    labs = {"t1_plain": "t1 crop+flip only", "t2_ls_wd": "t2 + label smooth, wd", "t3_randaug": "t3 + RandAugment", "t4_mix": "t4 + MixUp/CutMix",
            "t5_droppath": "t5 + drop-path", "t5b_mix_half": "t5b mixing on 1/2 batches", "t5c_bs256": "t5c batch 256 (4× steps)", "t6_full_300ep": "t6 full recipe, 300 ep"}
    items = [(labs[n], acc(n), "vit", n, acc(n, "final_train_acc_clean")) for n in lad]
    ch = hbars(items, "CIFAR-10 ablation ladder, val and clean-train accuracy", xlo=40, xhi=100, lw=200)
    lg = legend([("vit", "val acc (solid)", "bar"), ("vit", "clean train acc (outline)", "ghost")])
    g = lambda n: (acc(n, "final_train_acc_clean") or 0) - (acc(n) or 0)
    return chap(2, "ch2", "The regularization ladder on CIFAR-10", chapter_date(lad),
        "<p>A plain ViT-Tiny (12 layers, width 192, 3 heads, patch 4) has no built-in notion of locality. With only 50k images, how badly does it overfit, and which regularizer fixes it?</p>",
        "<p>One change per rung, 100 epochs, batch 1024, lr 2e-3 (the defaults):</p>" + code_flags(*lad),
        "<ul><li><b>Memorization</b>: a model with 5M parameters can store 50k labels. Train accuracy hits 100% while validation stalls. The gap between the two is the overfitting signal.</li>"
        "<li><b>RandAugment</b> creates new plausible images every step, so memorizing gets harder.</li>"
        "<li><b>MixUp/CutMix</b> blends two images and their labels. It is a strong regularizer, which also makes the training task harder: the model needs many more steps to fit it.</li>"
        "<li><b>Step count</b>: at batch 1024, 100 epochs is only ~4.9k optimizer steps. Going to batch 256 gives 4× the updates for the same number of images seen.</li></ul>",
        fig(ch, "Bars: final val accuracy. Outline: clean train accuracy. A long outline beyond the bar means memorization; outline ≈ bar means the model underfits.", lg) +
        f"<p><code>t1_plain</code> reaches {A('t1_plain', 'final_train_acc_clean')}% train but {A('t1_plain')}% val, a {g('t1_plain'):.1f}-point gap. RandAugment adds {D('t3_randaug', 't2_ls_wd')} points and shrinks the gap to {g('t3_randaug'):.1f}. "
        f"Adding MixUp/CutMix <em>drops</em> val by {D('t4_mix', 't3_randaug')} points, but train accuracy falls to {A('t4_mix', 'final_train_acc_clean')}%: the model now underfits. "
        f"Mixing only half the batches recovers {D('t5b_mix_half', 't5_droppath')}; four times the steps (batch 256) recovers {D('t5c_bs256', 't5_droppath')}; 300 epochs reaches {A('t6_full_300ep')}%.</p>",
        "<p>Read the train curve and the val curve together. A falling val number can mean overfitting or underfitting, and the fix is opposite in each case. "
        "Strong augmentation needs a long schedule. At batch 1024 the model was starved of optimizer steps, not over-regularized.</p>",
        f"<p>The matched published baseline (same 192/12/3 p4 shape), {cite('mimetic_c10')}, gets {P('mimetic_c10')}% at 100 epochs. Our 300-epoch {A('t6_full_300ep')}% is within 0.5 points of it, "
        f"and 5.6 points below tuned small ViTs such as {cite('vitlite_c10')} ({P('vitlite_c10')}%).</p>")


def ch3():
    items = [("ViT-Small, lr 2e-3", acc("s_small_200ep"), "mute", "s_small_200ep", acc("s_small_200ep", "final_train_acc_clean")),
             ("ViT-Small, lr 1e-3", acc("s_small_200ep_lr1e-3"), "vit", "s_small_200ep_lr1e-3", acc("s_small_200ep_lr1e-3", "final_train_acc_clean")),
             ("ViT-Tiny t5, batch 1024", acc("t5_droppath"), "mute", "", acc("t5_droppath", "final_train_acc_clean")),
             ("ViT-Tiny t5c, batch 256", acc("t5c_bs256"), "vit", "", acc("t5c_bs256", "final_train_acc_clean")),
             ("patch 8 (16 tokens)", acc("p8_tiny"), "hyb", "", acc("p8_tiny", "final_train_acc_clean")),
             ("patch 4 (64 tokens)", acc("t5_droppath"), "hyb", "", acc("t5_droppath", "final_train_acc_clean")),
             ("patch 2 (256 tokens)", acc("p2_tiny"), "hyb", "", acc("p2_tiny", "final_train_acc_clean"))]
    ch = hbars(items, "Width, learning rate and patch size on CIFAR-10", xlo=40, xhi=90, lw=190)
    names = ["s_small_200ep", "s_small_200ep_lr1e-3", "p2_tiny", "p8_tiny"]
    p2 = R("p2_tiny")
    p4 = R("t5_droppath")
    return chap(3, "ch3", "Scale up: width, learning rate and tokens", chapter_date(names),
        "<p>If ViT-Tiny overfits, does a wider ViT-Small (21M) do better? And what does patch size, which sets the number of tokens, buy?</p>",
        code_flags(*names),
        "<p><b>Learning rate must shrink as width grows.</b> With Adam every weight moves by about the learning rate each step, so a wider layer, which sums more inputs, changes its output more per step. "
        "A rate that is safe at width 192 can be too large at width 384 and the loss plateaus. "
        "<b>Patch size sets sequence length</b>: a 32×32 image cut into p×p patches gives (32/p)² tokens. Patch 2 means 256 tokens and roughly 4× the attention compute of patch 4; patch 8 gives only 16 tokens and loses spatial detail.</p>",
        fig(ch, "Bars: val accuracy; outlines: clean train accuracy. All patch runs use the underfitting 100-epoch batch-1024 recipe.", legend([("vit", "val", "bar"), ("vit", "clean train", "ghost")])) +
        f"<p>ViT-Small at lr 2e-3 stalled near loss 2.1 and {A('s_small_200ep')}% val. Halving lr to 1e-3 gave {A('s_small_200ep_lr1e-3')}% ({D('s_small_200ep_lr1e-3', 's_small_200ep')}), so the stall was an optimization failure. "
        f"Even fixed, Small (200 ep) is below Tiny at batch 256 ({A('t5c_bs256')}%). Patch 2 is {D('p2_tiny', 't5_droppath')} over patch 4 at "
        + (f"{p2['img_per_sec_steady'] / p4['img_per_sec_steady']:.2f}× the throughput" if p2 and p4 else "lower throughput") + f"; patch 8 loses {D('p8_tiny', 't5_droppath')}.</p>",
        "<p>Scale the learning rate down when you scale width up, and treat a flat loss curve as a learning-rate suspect before calling it a capacity limit. More parameters did not help at 50k images.</p>",
        f"<p>No published ViT-Small CIFAR-10 number is in our verified notes. The CCT paper's {cite('vit1216_c10')} (86M, patch 16) gets {P('vit1216_c10')}%, which agrees with the trend: bigger and coarser is not better on small data.</p>")


def ch4():
    pairs = [("CIFAR-10", "r18_full", "t5_droppath"), ("CIFAR-100", "c100_r18_full", "c100_t5_droppath"),
             ("Tiny-ImageNet", "tin_r18_full", "tin_t5_droppath"), ("ImageNet-64", "in64_r18_full", "in64_t5_full")]
    items = []
    for ds, r18, vit in pairs:
        items.append((f"{ds}: ResNet-18", acc(r18), "cnn", r18, acc(r18, "final_train_acc_clean")))
        items.append((f"{ds}: ViT-Tiny", acc(vit), "vit", vit, acc(vit, "final_train_acc_clean")))
    ch = hbars(items, "ResNet-18 vs ViT-Tiny with the same recipe and budget", xlo=0, xhi=100, lw=200)
    return chap(4, "ch4", "The ResNet-18 control", chapter_date([p[1] for p in pairs]),
        "<p>Is the ViT's weak result a property of the data and budget, or of the architecture? Train a ResNet-18 with the identical heavy recipe and compare.</p>",
        code_flags("r18_full", "c100_r18_full", "tin_r18_full", "in64_r18_full"),
        "<p>A convolution looks only at a small neighbourhood and uses the same weights at every position. Those two assumptions (locality and translation equivariance) are true of natural images, "
        "so the network does not have to learn them from data. That is called <em>inductive bias</em>. A ViT starts with neither and must learn them from examples, which costs data and steps.</p>",
        fig(ch, "Same recipe (RandAugment, MixUp/CutMix, drop-path where it applies) and the same epochs per dataset.", legend([("cnn", "ResNet-18", "bar"), ("vit", "ViT-Tiny", "bar"), ("mute", "clean train (outline)", "ghost")])) +
        f"<p>CIFAR-10: ResNet-18 {A('r18_full')}% in {R('r18_full')['total_minutes']:.1f} min vs ViT-Tiny {A('t5_droppath')}% ({D('r18_full', 't5_droppath')}). "
        f"The gap is {D('c100_r18_full', 'c100_t5_droppath')} on CIFAR-100, {D('tin_r18_full', 'tin_t5_droppath')} on Tiny-ImageNet and {D('in64_r18_full', 'in64_t5_full')} on ImageNet-64 (30 epochs, 1.28M images).</p>",
        "<p>At short budgets the CNN wins everywhere. The margin shrinks as the dataset grows: with 1.28M images the ViT has more examples to learn locality from.</p>",
        f"<p>Our ResNet-18 ({A('r18_full')}%) matches {cite('r18kuangliu_c10')} ({P('r18kuangliu_c10')}%) and is {P('r18cutout_c10') - acc('r18_full'):.1f} points under the tuned {cite('r18cutout_c10', 'Cutout-paper ResNet-18')} ({P('r18cutout_c10')}%, 200 ep, batch 128).</p>")


def ch5():
    lr = [("lr 3e-3 (paper recipe)", acc("bf1_slvit"), "mute", "bf1_slvit", acc("bf1_slvit", "final_train_acc_clean")),
          ("lr 1e-3", acc("bf1_slvit_lr1e-3"), "vit", "bf1_slvit_lr1e-3", acc("bf1_slvit_lr1e-3", "final_train_acc_clean")),
          ("lr 5e-4", acc("bf1_slvit_lr5e-4"), "vit", "bf1_slvit_lr5e-4", acc("bf1_slvit_lr5e-4", "final_train_acc_clean")),
          ("Tiny shape, lr 3e-3", acc("bf1b_tiny_samerecipe"), "bad", "bf1b", acc("bf1b_tiny_samerecipe", "final_train_acc_clean")),
          ("impulse init, lr 3e-3", acc("bf2_slvit_init"), "bad", "bf2", acc("bf2_slvit_init", "final_train_acc_clean"))]
    chlr = hbars(lr, "SL-ViT 100-epoch screens: learning rate check", xlo=40, xhi=95, lw=180)
    names = ["bf3_slvit_300ep", "bf3b_slvit_300ep_lr5e-4", "bf4_slvit_300ep_mixhalf"]
    cv = curves(names, "300-epoch SL-ViT runs: val accuracy", ylo=40, yhi=95,
                labels={"bf3_slvit_300ep": "bf3 lr 3e-3", "bf3b_slvit_300ep_lr5e-4": "bf3b lr 5e-4", "bf4_slvit_300ep_mixhalf": "bf4 lighter mix"},
                cls_map={"bf3_slvit_300ep": "mute", "bf3b_slvit_300ep_lr5e-4": "vit", "bf4_slvit_300ep_mixhalf": "hyb"})
    return chap(5, "ch5", "Matching published pure ViTs", chapter_date(["bf1_slvit"] + names),
        f"<p>Published pure ViTs reach about 93.6 to 94.5% on CIFAR-10 at 3M parameters. Our ViT-Tiny was at {A('t6_full_300ep')}%. Can we reproduce their shape and recipe?</p>",
        "<p>The SL-ViT shape (9 layers, width 192, 12 heads of dim 16, MLP ratio 2, 2.7M params) and recipe (batch 128, warmup, random erasing). Then LSA, SeqPool and impulse init, and finally the learning rate:</p>"
        + code_flags("bf1_slvit", "bf1_slvit_lr5e-4", "bf3b_slvit_300ep_lr5e-4", "bf4_slvit_300ep_mixhalf"),
        "<ul><li><b>LSA</b> (locality self-attention) masks each token's attention to itself and learns the softmax temperature. Attention gets sharper and has to look at neighbours.</li>"
        "<li><b>SeqPool</b> classifies from an attention-weighted average of all tokens instead of a single CLS token.</li>"
        "<li><b>Impulse init</b> sets the query/key weights so each head starts out attending to a fixed nearby offset, like a convolution filter.</li>"
        "<li><b>Mix-off at epoch 175</b> turns MixUp/CutMix off for the last 40% so the model can fit clean images.</li>"
        "<li><b>Batch 128</b> means 8× the optimizer steps per epoch of batch 1024, which is what chapter 2 said we lacked.</li></ul>",
        fig(chlr, "100-epoch screens at batch 128. Red = runs that stalled (train ≈ val ≈ 49%).", legend([("vit", "val", "bar"), ("vit", "clean train", "ghost"), ("bad", "stalled", "bar")])) +
        fig(cv, "Val accuracy over 300 epochs. Evaluated every 5 epochs.") +
        f"<p>The paper's lr 3e-3 gave {A('bf1_slvit')}%, and with impulse init or the Tiny shape it stalled at ~49% train accuracy. Lowering lr fixed it: 1e-3 → {A('bf1_slvit_lr1e-3')}, 5e-4 → {A('bf1_slvit_lr5e-4')}. "
        f"At 300 epochs, lr 3e-3 reached {A('bf3_slvit_300ep')}% and lr 5e-4 (<code>bf3b</code>) reached <b>{A('bf3b_slvit_300ep_lr5e-4')}%</b>. Lighter mixing (<code>bf4</code>) gave {A('bf4_slvit_300ep_mixhalf')}%.</p>",
        "<p>A published recipe is tied to its own pipeline. Their lr 3e-3 assumed their sampler (3 augmented copies per image, ~117k steps). Without those repeats, the same lr was too high. Rerun a short learning-rate sweep whenever you port a recipe.</p>",
        f"<p><code>bf3b</code> {A('bf3b_slvit_300ep_lr5e-4')}% vs {cite('vitlite_c10')} {P('vitlite_c10')} {pill(acc('bf3b_slvit_300ep_lr5e-4'), P('vitlite_c10'))} and {cite('sptvit_c10')} {P('sptvit_c10')} {pill(acc('bf3b_slvit_300ep_lr5e-4'), P('sptvit_c10'))}. "
        f"It is still below {cite('slvit_c10')} ({P('slvit_c10')}), which also uses shifted patch tokenization (SPT). We never implemented SPT.</p>")


def ch6():
    base = "bf3b_slvit_300ep_lr5e-4"
    abl = [("bf3b (all parts)", acc(base), "vit", base, None), ("− LSA", acc("abl_no_lsa"), "bad", "abl_no_lsa", None),
           ("− SeqPool (CLS token)", acc("abl_cls"), "mute", "abl_cls", None), ("− impulse init", acc("abl_default_init"), "mute", "abl_default_init", None),
           ("ViT-Tiny shape instead", acc("abl_tiny_shape"), "hyb", "abl_tiny_shape", None)]
    m, s, n = mean_std([acc(base), acc("bf3b_seed1"), acc("bf3b_seed2")])
    refs = [(m, f"seed mean {m:.2f}")] if m else []
    ch = hbars(abl, "Remove one part of bf3b at a time (CIFAR-10, 300 ep, seed 0)", xlo=88, xhi=96, lw=190, refs=refs)
    names = ["abl_no_lsa", "abl_cls", "abl_default_init", "abl_tiny_shape"]
    return chap(6, "ch6", "One-change ablation: which part earned the credit?", chapter_date(names + ["bf3b_seed1"]),
        "<p><code>bf3b</code> changed five things at once versus our ViT-Tiny. Which ones mattered? Undo exactly one per run, and repeat seeds to know what a real difference is.</p>",
        code_flags(*names) + code_flags("bf3b_seed1", "bf3b_seed2"),
        "<p>A bundle of changes only tells you the sum. A one-change ablation tells you each part's share, and only if the differences beat seed noise. "
        f"Three seeds of <code>bf3b</code> give a standard deviation of {s:.2f} points, so differences under about 0.3 points are noise.</p>" if s else "<p>A one-change ablation isolates each part's share.</p>",
        fig(ch, "Dashed line: mean of three bf3b seeds. Red = the part whose removal hurt most.") +
        (f"<p>Removing LSA costs {D('abl_no_lsa', base)} points, by far the biggest drop. CLS instead of SeqPool costs {D('abl_cls', base)}, default init {D('abl_default_init', base)}. "
        f"Swapping the SL-ViT shape for the plain ViT-Tiny shape (12 layers, 3 heads, MLP 4) with the same recipe <em>gains</em> {D('abl_tiny_shape', base)} points to <b>{A('abl_tiny_shape')}%</b>, at {R('abl_tiny_shape')['params_M']:.1f}M vs {R(base)['params_M']:.1f}M params.</p>" if R("abl_tiny_shape") and R(base) else ""),
        "<p>We credited the jump from 88 to 93% to the SL-ViT shape. The controlled test says otherwise: the recipe (lr, batch 128, long schedule, mix-off) and LSA did the work, and the shape we abandoned is better when trained properly. "
        "Bundled comparisons in the literature (a different shape <em>and</em> recipe) can mislead in the same way.</p>",
        f"<p>ViT-Tiny shape + bf3b recipe: {A('abl_tiny_shape')}% vs {cite('slvit_c10')} {P('slvit_c10')} {pill(acc('abl_tiny_shape'), P('slvit_c10'))}, "
        f"close to the {cite('structinit_c10', 'structured-init ViT-Tiny')} ({P('structinit_c10')}%, 300 ep). Note the comparison is not size-matched: 5.4M vs 2.9M params.</p>")


def ch7():
    lad = ["c100_t1_plain", "c100_t2_ls_wd", "c100_t3_randaug", "c100_t4_mix", "c100_t5_droppath", "c100_t6_full_300ep", "c100_s_small_200ep",
           "c100_r18_full", "c100_bf3b_slvit_300ep_lr5e-4", "c100_tinyshape_bf3b"]
    lab = {"c100_t1_plain": "t1 plain", "c100_t2_ls_wd": "t2 + ls, wd", "c100_t3_randaug": "t3 + RandAug", "c100_t4_mix": "t4 + mix", "c100_t5_droppath": "t5 + drop-path",
           "c100_t6_full_300ep": "t6 300 ep", "c100_s_small_200ep": "ViT-Small 200 ep", "c100_r18_full": "ResNet-18 100 ep",
           "c100_bf3b_slvit_300ep_lr5e-4": "bf3b recipe (SL-ViT)", "c100_tinyshape_bf3b": "bf3b recipe (Tiny shape)"}
    items = [(lab[n], acc(n), "cnn" if "r18" in n else ("hyb" if "bf3b" in n else "vit"), n, acc(n, "final_train_acc_clean")) for n in lad]
    ch = hbars(items, "CIFAR-100 runs", xlo=40, xhi=100, lw=180, refs=[(P("slvit_c100"), "SL-ViT")])
    tin = ["tin_t1_plain", "tin_t3_randaug", "tin_t4_mix", "tin_t5_droppath", "tin_p4_tiny", "tin_s_small_200ep", "tin_r18_full"]
    tl = {"tin_t1_plain": "t1 plain p8", "tin_t3_randaug": "t3 + RandAug", "tin_t4_mix": "t4 + mix", "tin_t5_droppath": "t5 + drop-path", "tin_p4_tiny": "patch 4 (256 tok)",
          "tin_s_small_200ep": "ViT-Small 200 ep", "tin_r18_full": "ResNet-18"}
    i64 = ["in64_t1_plain", "in64_t5_full", "in64_s_small_full", "in64_r18_full"]
    il = {"in64_t1_plain": "IN-64 ViT-Tiny plain", "in64_t5_full": "IN-64 ViT-Tiny full", "in64_s_small_full": "IN-64 ViT-Small 40 ep", "in64_r18_full": "IN-64 ResNet-18"}
    items2 = [(tl[n], acc(n), "cnn" if "r18" in n else "vit", n, acc(n, "final_train_acc_clean")) for n in tin] + \
             [(il[n], acc(n), "cnn" if "r18" in n else "vit", n, acc(n, "final_train_acc_clean")) for n in i64]
    ch2_ = hbars(items2, "Tiny-ImageNet (64 px, 200 classes) and ImageNet-64 (1000 classes)", xlo=0, xhi=100, lw=180, refs=[(P("slvit_tin"), "SL-ViT TIN")])
    lg = legend([("vit", "ViT ladder", "bar"), ("hyb", "bf3b recipe", "bar"), ("cnn", "ResNet-18", "bar"), ("vit", "clean train", "ghost")])
    g = lambda n: (acc(n, "final_train_acc_clean") or 0) - (acc(n) or 0)
    return chap(7, "ch7", "Harder data: CIFAR-100, Tiny-ImageNet, ImageNet-64", chapter_date(lad + tin + i64),
        "<p>Do the CIFAR-10 lessons transfer when there are 100 classes (500 images each), 200 classes at 64 px, or 1.28M images?</p>",
        "<p>The same ladder at batch 256 with sqrt-scaled lr (1e-3; ViT-Small 5e-4); patch 8 at 64 px keeps 64 tokens. Then the bf3b recipe on CIFAR-100:</p>"
        + code_flags("c100_t3_randaug", "c100_bf3b_slvit_300ep_lr5e-4", "c100_tinyshape_bf3b", "tin_t5_droppath", "in64_s_small_full"),
        "<p>Fewer images per class make memorizing easier and learning general features harder, so the train-val gap grows. More total data has the opposite effect: with ImageNet's 1.28M images the ViT can learn locality from examples, so the CNN's built-in advantage matters less.</p>",
        fig(ch, "CIFAR-100. Dashed: SL-ViT published.", lg) + fig(ch2_, "Tiny-ImageNet and ImageNet-64 (30 to 40 epochs).", lg) +
        f"<p>CIFAR-100 plain ViT: {A('c100_t1_plain', 'final_train_acc_clean')}% train, {A('c100_t1_plain')}% val, a {g('c100_t1_plain'):.1f}-point gap. RandAugment again helps most ({D('c100_t3_randaug', 'c100_t2_ls_wd')}). "
        f"The bf3b recipe reaches <b>{A('c100_bf3b_slvit_300ep_lr5e-4')}%</b>; the Tiny-shape version is {A('c100_tinyshape_bf3b')}. ResNet-18 gets {A('c100_r18_full')}% in {R('c100_r18_full')['total_minutes']:.1f} min. "
        f"On Tiny-ImageNet, ResNet-18 leads the best ViT by {(acc('tin_r18_full') or 0) - max(acc(n) or 0 for n in tin if 'r18' not in n):.1f} points. On ImageNet-64, ViT-Small (40 ep) reaches {A('in64_s_small_full')}% and "
        f"<em>passes</em> ResNet-18 (30 ep, {A('in64_r18_full')}%).</p>",
        "<p>The ViT-vs-CNN gap depends on data size per inductive bias you leave out. On small datasets, build locality in (LSA, a conv stem) or bring pretrained weights. On ImageNet scale, the ViT catches up even at short budgets. The ImageNet-64 comparison is not epoch-matched (40 vs 30).</p>",
        f"<p>CIFAR-100: {A('c100_bf3b_slvit_300ep_lr5e-4')}% vs {cite('sptvit_c100')} {P('sptvit_c100')} {pill(acc('c100_bf3b_slvit_300ep_lr5e-4'), P('sptvit_c100'))}, "
        f"{cite('vitlite_c100')} {P('vitlite_c100')} {pill(acc('c100_bf3b_slvit_300ep_lr5e-4'), P('vitlite_c100'))}, {cite('slvit_c100')} {P('slvit_c100')} {pill(acc('c100_bf3b_slvit_300ep_lr5e-4'), P('slvit_c100'))}. "
        f"The hybrid {cite('cct_c100')} gets {P('cct_c100')}, and the from-scratch record is {cite('asam_c100')} at {P('asam_c100')} with 26M params and 900 epochs. "
        f"Tiny-ImageNet: {cite('slvit_tin')} reports {P('slvit_tin')}; our best ViT is {max(acc(n) or 0 for n in tin if 'r18' not in n):.2f}. We have no verified ImageNet-64 reference.</p>")


def rd_sweep_chart(runs, title, labels):
    ser = []
    for n in runs:
        r = RUNS_D.get(n, {}).get("rec")
        if r and r.get("depth_sweep"):
            ser.append((n, r))
    if not ser:
        return '<p class="pend">No recurrent evaluations yet.</p>'
    ys = [a for _, r in ser for a in r["depth_sweep"]["acc"]]
    ch = Chart(680, 300, title, mr=170)
    ch.setx(1, 32)
    ch.sety(math.floor(min(ys) / 5) * 5, math.ceil(max(ys) / 2) * 2)
    ch.axes("loop iterations r at test time", "val acc (%)", xticks=[1, 4, 8, 12, 16, 20, 24, 28, 32])
    ch.vline(8, "train mean r = 8")
    cls = ["vit", "hyb", "mute", "ours", "cnn", "bad"]
    ends = []
    for i, (n, r) in enumerate(ser):
        ds = r["depth_sweep"]
        ch.line(ds["r"], ds["acc"], cls[i % len(cls)], name=None, label_end=False, marker=False)
        ends.append([ch.Y(ds["acc"][-1]), f"{labels.get(n, n)} {ds['acc'][-1]:.1f}"])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
    for y, lab in ends:
        ch.parts.append(f'<text class="dl" x="{ch.w - ch.mr + 8}" y="{y + 4:.1f}" fill="{FALLBACK["ink"]}">{esc(lab)}</text>')
    lg = legend([(cls[i % len(cls)], labels.get(n, n), "line") for i, (n, _) in enumerate(ser)])
    return fig(ch.svg(), "Each line is one trained model, evaluated with r = 1 to 32 loop iterations. Right of the dashed line is depth the model never trained with on average.", lg)


def exit_chart(n):
    r = RUNS_D.get(n, {}).get("rec")
    if not r or not r.get("adaptive_exit"):
        return ""
    ds, ae = r["depth_sweep"], r["adaptive_exit"]
    kl = sorted(ae["kl"], key=lambda d: d["avg_iters"])
    xs = ds["r"] + [d["avg_iters"] for d in kl]
    ys = ds["acc"] + [d["acc"] for d in kl]
    keep = [i for i, rr in enumerate(ds["r"]) if rr <= 16]
    ds = {"r": [ds["r"][i] for i in keep], "acc": [ds["acc"][i] for i in keep]}
    ys = ds["acc"] + [d["acc"] for d in kl]
    ch = Chart(640, 280, f"Adaptive exit vs fixed depth ({n})", mr=110)
    ch.setx(1, 16)
    ch.sety(math.floor(min(ys) / 2) * 2, math.ceil(max(max(ys), ae["oracle"]["acc"]) + 1))
    ch.axes("average loop iterations per image", "val acc (%)", xticks=[1, 2, 4, 6, 8, 10, 12, 14, 16])
    ch.line(ds["r"], ds["acc"], "mute", name="fixed r", marker=False)
    ch.line([d["avg_iters"] for d in kl], [d["acc"] for d in kl], "vit", name="KL exit", marker=False)
    o = ae["oracle"]
    ch.dot(o["avg_iters"], o["acc"], "hyb", f"oracle: {o['acc']:.2f}% at {o['avg_iters']:.2f} iterations", shape="diamond")
    ch.text(o["avg_iters"], o["acc"], "oracle exit", dx=10, dy=4)
    return fig(ch.svg(), "Grey: every image runs r loops. Blue: each image stops when its prediction stops changing (KL between successive loops below τ; τ sweep). "
               "Orange: an oracle that stops at the first loop with a correct answer, an upper bound.",
               legend([("mute", "fixed r", "line"), ("vit", "KL exit rule", "line"), ("hyb", "oracle", "diamond")]))


def ch8():
    c10 = ["rd_pilot_a", "rd_pilot_b", "rd_pilot_noinject", "rd_flat6", "rd_deep_r8", "rd_deep_r8_add", "rd_deep_r8_ds", "rd_deep_r8_add_ds", "rd_deep_single", "rd_deep_single_add"]
    lab = {"rd_pilot_a": "pilot r4, concat", "rd_pilot_b": "pilot r4, add", "rd_pilot_noinject": "pilot r4, no inject", "rd_flat6": "flat 6-block control",
           "rd_deep_r8": "r8 concat (default)", "rd_deep_r8_add": "r8 add", "rd_deep_r8_ds": "r8 concat + DS", "rd_deep_r8_add_ds": "r8 add + DS",
           "rd_deep_single": "1-block core, concat", "rd_deep_single_add": "1-block core, add"}
    def inj(n):
        rec = RUNS_D.get(n, {}).get("rec")
        return rec.get("inject") if rec else None
    clsf = lambda n: "mute" if "flat" in n else ("vit" if inj(n) == "add" else ("bad" if inj(n) == "none" else "hyb"))
    items = [(lab[n], acc(n), clsf(n), n, None) for n in c10]
    ch = hbars(items, "Recurrent-depth ViT on CIFAR-10 (100 ep, 2.3-2.8M params)", xlo=76, xhi=88, lw=180)
    lg = legend([("vit", "add injection", "bar"), ("hyb", "concat injection", "bar"), ("bad", "no injection", "bar"), ("mute", "non-recurrent control", "bar")])
    sweep = rd_sweep_chart(["rd_deep_r8_add_ds", "rd_deep_r8_add", "rd_deep_r8", "rd_pilot_b", "rd_deep_single_add"], "CIFAR-10 depth extrapolation r = 1..32",
                           {"rd_deep_r8_add_ds": "r8 add+DS", "rd_deep_r8_add": "r8 add", "rd_deep_r8": "r8 concat", "rd_pilot_b": "r4 add (pilot)", "rd_deep_single_add": "1-block core, add"})
    sweep += rd_sweep_chart(["c100_rd_r8_add", "c100_rd_r8_add_ds"], "CIFAR-100 depth extrapolation r = 1..32", {"c100_rd_r8_add": "C100 r8 add", "c100_rd_r8_add_ds": "C100 r8 add+DS"})
    ext_rows = []
    for n in c10 + ["c100_rd_r8_add", "c100_rd_r8_add_ds", "c100_rd_flat6", "rd_deep_r8_add_ds_seed1", "rd_deep_r8_add_ds_seed2"]:
        rec = RUNS_D.get(n, {}).get("rec")
        if not rec or not rec.get("depth_sweep"):
            continue
        d = rec["depth_sweep"]
        ext_rows.append([f"<code>{n}</code>", str(rec.get("inject")), str(rec.get("r_mean")), f"{d['acc_at_r_mean']:.2f}", str(d["best_r"]), f"{d['best_acc']:.2f}", f"{d['acc_at_r_max']:.2f}",
                         f"{d['acc_at_r_max'] - d['acc_at_r_mean']:+.2f}"])
    ext = table(["run", "inject", "train r̄", "acc @ r̄", "best r", "best acc", "acc @ r=32", "32 vs r̄"], ext_rows, num_cols=(2, 3, 4, 5, 6, 7))
    best = RUNS_D.get("rd_deep_r8_add_ds", {}).get("rec") or {}
    dif = best.get("difficulty", {})
    c10h = best.get("cifar10h", {})
    kl = best.get("adaptive_exit", {}).get("kl", [])
    ex_txt = ""
    if kl and best.get("depth_sweep"):
        tgt = best["depth_sweep"]["acc_at_r_mean"]
        ok = [d for d in kl if d["acc"] >= tgt - 0.1]
        if ok:
            e = min(ok, key=lambda d: d["avg_iters"])
            ex_txt = (f"<p>Adaptive exit: the KL rule keeps {e['acc']:.2f}% (within 0.1 of fixed r = 8's {tgt:.2f}%) with {e['avg_iters']:.2f} loops per image on average. "
                      f"Images the model gets wrong exit later ({dif.get('mean_exit_wrong', 0):.1f} vs {dif.get('mean_exit_correct', 0):.1f} loops). Exit step correlates with the model's own loss "
                      f"(Spearman {dif.get('spearman_exit_vs_own_loss_at_r_mean', 0):.2f}) and with ResNet-18's loss on the same image ({dif.get('spearman_exit_vs_ref_loss', 0):.2f}), "
                      f"but only weakly with CIFAR-10H human label uncertainty ({c10h.get('spearman_exit_vs_entropy', 0):.2f}) and with CIFAR-10-C corruption severity "
                      f"({best.get('cifar10c', {}).get('pooled_sev_spearman', 0):.2f}).</p>")
    seeds = [acc("rd_deep_r8_add_ds"), acc("rd_deep_r8_add_ds_seed1"), acc("rd_deep_r8_add_ds_seed2")]
    m, s, k = mean_std(seeds)
    return chap(8, "ch8", "Research track: a recurrent-depth ViT", chapter_date(c10),
        "<p>Can one shared block, looped r times, match a deeper ViT at the same parameter count? Does it keep improving when looped past its training depth, and can each image stop when it is done?</p>",
        "<p><b>Prelude</b> (2 blocks) embeds the image; a <b>core</b> (2 blocks, or 1) is looped r times, with r sampled per batch around a mean of 4 or 8 and gradients through only the last 4 loops; "
        "a <b>coda</b> (2 blocks) and head classify. <b>Injection</b> feeds the prelude output back into every loop, by addition (<code>add</code>), by a concat+linear adapter (<code>concat</code>) or not at all. "
        "<b>Deep supervision</b> (<code>--rd-coda-every-step</code>) applies the loss after every loop.</p>" + code_flags("rd_pilot_b", "rd_deep_r8", "rd_deep_r8_add_ds", "c100_rd_r8_add"),
        "<p>Looping reuses parameters, so compute at test time becomes a dial you can turn without retraining. Re-injecting the input stops the state from drifting away from the image after many loops. "
        "Deep supervision teaches every loop to produce a usable answer, which is what an early exit needs.</p>",
        fig(ch, "Final val accuracy at the default test depth.", lg) + sweep + ext + exit_chart("rd_deep_r8_add_ds") +
        f"<p>Add injection beats concat by {D('rd_pilot_b', 'rd_pilot_a')} (pilot) and {D('rd_deep_r8_add', 'rd_deep_r8')} (r̄ = 8), and no injection is {D('rd_pilot_noinject', 'rd_pilot_b')} behind add. "
        f"The best recurrent model, r8 add + DS, gets {A('rd_deep_r8_add_ds')}% vs {A('rd_flat6')}% for the parameter-matched flat control ({D('rd_deep_r8_add_ds', 'rd_flat6')}); three seeds: {m:.2f} ± {s:.2f}. "
        f"On CIFAR-100, r8 add gets {A('c100_rd_r8_add')} vs {A('c100_rd_flat6')} for the flat control ({D('c100_rd_r8_add', 'c100_rd_flat6')}).</p>" + ex_txt
        + "<p><b>The injection-flag mistake.</b> The deep chain (<code>rd_deep_r8</code>, <code>_ds</code>, <code>_single</code>) left <code>--rd-inject</code> unset, so it silently used the <code>concat</code> default, although the pilot had shown <code>add</code> was better. "
        "We reran the same three configurations with <code>--rd-inject add</code>. The mistake turned into a replication: the add-over-concat gap appeared again at r̄ = 8 and with a single-block core.</p>",
        f"<p>With add injection, looping beats a flat model of equal size, and accuracy stays flat when the loop runs past its training depth; but all recurrent models are far below the {A('bf3b_slvit_300ep_lr5e-4')}% of the 300-epoch flat ViT (they had 100 epochs and a weaker recipe). "
        "Human-judged difficulty barely predicts exit step, so the exit rule tracks model uncertainty more than image difficulty. Always set every flag explicitly in queue scripts.</p>",
        f"<p>{cite('vitrm_c10')} reports {P('vitrm_c10')}% on CIFAR-10 and {cite('vitrm_c100', 'ViTRM')} {P('vitrm_c100')}% on CIFAR-100 at ~3.6M params; our recurrent models are at {A('rd_deep_r8_add_ds')} and {A('c100_rd_r8_add')} after 100 epochs. "
        "Related work without CIFAR numbers: reViT (<a href=\"https://arxiv.org/abs/2610.12448\" target=\"_blank\" rel=\"noopener\">2610.12448</a>) picks one global depth at inference; "
        "Geiping et al. (<a href=\"https://arxiv.org/abs/2502.05171\" target=\"_blank\" rel=\"noopener\">2502.05171</a>) is the recurrent-depth design we follow.</p>")


def ch9():
    pairs = []
    for e in sorted(STUDENTS, key=lambda e: e["disp"]):
        if e["px"] == 224 and not e["kd"] and not e["variant"]:
            twin = next((f for f in STUDENTS if f["stem"] == e["stem"] and f["px"] == 128 and not f["kd"] and not f["variant"]), None)
            if twin:
                pairs.append((e, twin))
    items = []
    for e, t in pairs:
        items.append((f"{e['disp']} @224", e["acc"], e["fam"], e["name"], None))
        items.append((f"{e['disp']} @128", t["acc"], e["fam"], t["name"], None))
    res_chart = hbars(items, "Input resolution: 224 vs 128 px (same model, no KD)", xlo=60, xhi=95, lw=170) if items else ""
    kd_rows, kd_rows_raw = [], []
    for e in STUDENTS:
        if e["kd"]:
            base = next((f for f in STUDENTS if f["stem"] == e["stem"] and f["px"] == e["px"] and not f["kd"] and not f["variant"]), None)
            if base is None:
                base_c = next((f for f in EDGE if f["stem"] == e["stem"] and f["px"] == e["px"] and not f["kd"] and not f["variant"] and f.get("acc") is not None), None)
                b = f"{base_c['acc']:.2f} (collapsed)" if base_c else "—"
                kd_rows.append([f"<code>{e['name']}</code>", e["disp"], str(e["px"]), e["kd"], b, f"{e['acc']:.2f}", "—"])
            else:
                kd_rows.append([f"<code>{e['name']}</code>", e["disp"], str(e["px"]), e["kd"], f"{base['acc']:.2f}", f"{e['acc']:.2f}", f"{e['acc'] - base['acc']:+.2f}"])
                kd_rows_raw.append(e["acc"] - base["acc"])
    kd_tab = table(["run", "student", "px", "KD", "no KD", "with KD", "gain"], kd_rows, num_cols=(2, 4, 5, 6)) if kd_rows else ""
    t = TEACH
    names = [e["name"] for e in EDGE]
    pre = ["in64pre_slvit_p8", "in64pre_tiny_p8", "ec100_in64pre_slvit_64", "ec100_in64pre_slvit_64_kd", "ec100_in64pre_tiny_64", "ec100_in64pre_tiny_64_kd"]
    pre_status = ", ".join(f"<code>{n}</code> {A(n) if R(n) else RUNS_D.get(n, {}).get('status', 'queued')}" for n in pre)
    scratch = {"slvit": "c100_bf3b_slvit_300ep_lr5e-4", "tiny": "c100_tinyshape_bf3b"}
    pre_rows = []
    for k, lab in (("slvit", "SL-ViT shape (2.7M)"), ("tiny", "ViT-Tiny shape (5.4M)")):
        for suf, kdl in (("", "no KD"), ("_kd", "soft KD from ViT-B")):
            n = f"ec100_in64pre_{k}_64{suf}"
            if R(n) and R(scratch[k]) and R(f"in64pre_{k}_p8"):
                pre_rows.append([lab, f"<code>in64pre_{k}_p8</code> {A(f'in64pre_{k}_p8')}%", f"<code>{n}</code>", kdl, f"{A(n)}",
                                 f"{A(scratch[k])} (<code>{scratch[k]}</code>)", f"{D(n, scratch[k])}"])
    pre_tab = table(["model", "ImageNet-64 pretrain, top-1", "CIFAR-100 fine-tune at 64 px", "KD", "fine-tuned acc", "same shape from scratch (300 ep)", "gain"], pre_rows, num_cols=(4, 6)) if pre_rows else ""
    mh = [(n, R(n)) for n in ("ec100_mnv4_hm_128", "ec100_mnv4_hm_224", "ec100_mnv4_hm_128_lr1e-4", "ec100_mnv4_hm_224_lr1e-4")]
    mnv4_txt = ("<p><b>MobileNetV4-Hybrid-M, three attempts.</b> The first 128 px run (lr 5e-4) blew up on its first optimizer step and ended at "
                + (f"{R('_failed_ec100_mnv4_hm_128')['final_val_acc']:.2f}%" if R("_failed_ec100_mnv4_hm_128") else "chance")
                + f". Zero-initializing the classifier head fixed that, but lr 5e-4 still diverged after warmup: {A('ec100_mnv4_hm_224')}% at 224 px and {A('ec100_mnv4_hm_128')}% at 128 px. "
                f"At lr 1e-4 both train cleanly from step 1: <b>{A('ec100_mnv4_hm_128_lr1e-4')}%</b> at 128 px and <b>{A('ec100_mnv4_hm_224_lr1e-4')}%</b> at 224 px "
                "(9.9M params, so outside the 6M budget). The lr 5e-4 runs are marked superseded and left out of the charts.</p>") if all(R(n) for n in ("ec100_mnv4_hm_128", "ec100_mnv4_hm_224", "ec100_mnv4_hm_128_lr1e-4", "ec100_mnv4_hm_224_lr1e-4")) else ""
    return chap(9, "ch9", "Edge track: pretrained students under 6M parameters", chapter_date(names),
        f"<p>If pretraining on external data is allowed, how close to the CIFAR-100 state of the art ({SOTA_PRE}%) can a model of at most ~6M parameters get, and what does it cost in compute and CPU latency?</p>",
        "<p>Fine-tune ImageNet-pretrained timm models for 30 epochs (ViT-B/16 teacher: 20). Images are upsampled on the GPU to 224 or 128 px. Distill from our fine-tuned ViT-B/16 teacher (soft KD, τ = 2, α = 0.5; or hard labels). "
        "Every finished model is exported to ONNX and timed on CPU with onnxruntime at 1 and 4 threads, fp32 and int8. We also queued our own ViTs pretrained on ImageNet-64 and fine-tuned at 64 px.</p>"
        + code_flags("ec100_T_vitb_ar2_224", "ec100_tinyvit_5m_224", "ec100_kd_tinyvit_5m_224", "ec100_in64pre_slvit_64_kd"),
        "<p>Pretraining supplies the visual features that 50k images cannot teach. Resolution matters because these networks were pretrained at 224 px and hierarchical ones downsample 32×; at 128 px each feature map is smaller and less like what they saw in pretraining. "
        "Distillation gives the student the teacher's full probability vector instead of a one-hot label, which carries information about which classes look alike.</p>",
        fig(pareto_chart("params", "params (M)", "Accuracy vs parameters"), "CIFAR-100 accuracy vs parameter count for every finished edge run. Hover a point for its run name.", PARETO_LEGEND)
        + (fig(res_chart, "Each pair is one pretrained model fine-tuned at two input sizes.") if res_chart else "") + kd_tab +
        (f"<p>Teacher: {t['disp']} at 224 px, {t['acc']:.2f}% ({t['params']:.0f}M params). Best ≤6M student: {BEST_SMALL['disp']}{' + KD' if BEST_SMALL['kd'] else ''} at {BEST_SMALL['px']} px, <b>{BEST_SMALL['acc']:.2f}%</b>, "
         f"which is {100 * BEST_SMALL['acc'] / t['acc']:.1f}% of our teacher and {100 * BEST_SMALL['acc'] / SOTA_PRE:.1f}% of published SOTA.</p>" if t and BEST_SMALL else "")
        + smallest_within() + f"<p>ImageNet-64 pretraining of our own ViTs (pretrain on 64 px ImageNet, then fine-tune at 64 px on CIFAR-100): {pre_status}.</p>" + pre_tab + mnv4_txt,
        f"<p>The 99%-of-SOTA bar is {0.99 * SOTA_PRE:.2f}%. In the verified notes the smallest model that clears it is EfficientNet-L2 with {P('effl2_params')}M params at 475 px. No published model under that size reaches it, so that bar is the wrong target for a 5M edge model. "
        "A useful edge target is a fraction of a teacher we measured ourselves under the same pipeline, plus the 90% and 95% of SOTA lines. "
        + kd_res_lesson(pairs, kd_rows_raw) + "</p>",
        f"<p>Small published fine-tunes: {cite('effb0_c100')} {P('effb0_c100')}% (4M), {cite('mnv3_c100')} {P('mnv3_c100')}% (4.2M), {cite('deitt_c100')} {P('deitt_c100')}% (5.5M). "
        + (f"Our {BEST_SMALL['disp']} at {BEST_SMALL['acc']:.2f}% {pill(BEST_SMALL['acc'], P('effb0_c100'))} vs EfficientNet-B0. " if BEST_SMALL else "")
        + f"Bigger: {cite('effv2s_c100')} {P('effv2s_c100')}% (24M), {cite('vith_c100')} {P('vith_c100')}% (632M), {cite('effl2sam_c100')} {P('effl2sam_c100')}% (480M). "
        f"No published CIFAR-100 fine-tune exists for TinyViT-5M in our notes.</p>")


def kd_res_lesson(pairs, kd_gains):
    res = sorted(e["acc"] - t["acc"] for e, t in pairs)
    out = ""
    if res:
        out += f" Going from 128 to 224 px changed accuracy by a median of {statistics.median(res):+.2f} points across {len(res)} models (range {res[0]:+.2f} to {res[-1]:+.2f})."
    if kd_gains:
        out += f" Distillation from the ViT-B teacher changed it by {min(kd_gains):+.2f} to {max(kd_gains):+.2f} points. At this budget, resolution was the bigger lever."
    return out


def ch10():
    failed = sorted(d.name for d in RUNS.iterdir() if d.is_dir() and d.name.startswith("_failed"))
    fl = []
    for f in failed:
        r = jload(RUNS / f / "results.json")
        if r:
            if "mnv4" not in f:
                fl.append(f"<code>{f[len('_failed_'):]}</code> ended at {r['final_val_acc']:.2f}% (best EMA {r.get('best_ema_val_acc') or 0:.2f}%)")
    reruns = []
    for f in failed:
        n = f[len("_failed_"):]
        if "mnv4" in n:
            continue
        reruns.append(f"<code>{n}</code> {A(n) + '%' if R(n) else RUNS_D.get(n, {}).get('status', 'queued')}")
    kdok = f" The KD run at 128 px (<code>ec100_kd_effformv2_s0_128</code>) trained normally to {A('ec100_kd_effformv2_s0_128')}%." if R("ec100_kd_effformv2_s0_128") else ""
    ef_sym = ("; ".join(fl) or "loss stayed near ln 100 = 4.61") + ". The 128 px log shows training loss at 4.5 to 4.6 throughout, close to ln 100 = 4.61 (chance)." + kdok
    ef_fix = ("Bisected in commit 41588c0: eager bf16, compiled fp32, and compiled bf16 with only the attention modules in fp32 all train, so <code>finetune.py</code> now runs EfficientFormerV2 attention in fp32. "
              "The collapsed runs were moved to <code>runs/_failed_*</code> and requeued (<code>run_rerun_effformv2.sh</code>): " + (", ".join(reruns) or "pending") + ".")
    hm_done = all(R(n) for n in ("ec100_mnv4_hm_128", "ec100_mnv4_hm_224", "ec100_mnv4_hm_128_lr1e-4", "ec100_mnv4_hm_224_lr1e-4"))
    hm_row = None
    if hm_done:
        hm_row = ["MobileNetV4-Hybrid-M: head init, then lr", "Ch. 9",
                  f"Step-1 blow-up (<code>_failed_ec100_mnv4_hm_128</code> {R('_failed_ec100_mnv4_hm_128')['final_val_acc']:.2f}%); after zero-init head, lr 5e-4 still diverged after warmup: 224 px {A('ec100_mnv4_hm_224')}%, 128 px {A('ec100_mnv4_hm_128')}%",
                  f"zero-init head + lr 1e-4: 128 px {A('ec100_mnv4_hm_128_lr1e-4')}%, 224 px {A('ec100_mnv4_hm_224_lr1e-4')}%",
                  "A first-step blow-up and a post-warmup divergence are different bugs; fix one, then re-check the curve"]
    rows = [
        ["Paper lr 3e-3 too high for our sampler", "Ch. 5", f"<code>bf1_slvit</code> {A('bf1_slvit')}%; <code>bf1b</code>, <code>bf2</code> stalled at ~49% train", "lr sweep: 5e-4 → " + A("bf1_slvit_lr5e-4") + "; bf3b " + A("bf3b_slvit_300ep_lr5e-4"), "Re-sweep lr when porting a recipe; the paper's sampler gave ~3× the steps"],
        ["lr not scaled down for ViT-Small", "Ch. 3", f"<code>s_small_200ep</code> loss plateau ~2.1, {A('s_small_200ep')}%", f"lr 1e-3 → {A('s_small_200ep_lr1e-3')}%", "Scale lr with width"],
        ["<code>--rd-inject</code> left at its default (concat)", "Ch. 8", f"<code>rd_deep_r8</code> {A('rd_deep_r8')}% used concat by accident", f"rerun with add: {A('rd_deep_r8_add')}%; the gap replicated", "Pass every flag explicitly; log the effective config"],
        ["CUDA graphs overwrote the loop output", "Ch. 1, 8", "A CUDA-graphed core reuses its output buffer on every replay, so looping it fed each step the overwritten tensor", "clone the core output each loop; grads match eager to 1.3e-6, ~1.9× faster (10.5 → 5.6 s/epoch)", "Graph replays share memory; copy outputs you keep"],
        ["EfficientFormerV2-S0 collapsed to chance under compile + bf16", "Ch. 9", ef_sym, ef_fix + (f" Reruns reached {A('ec100_effformv2_s0_224')}% at 224 px." if R("ec100_effformv2_s0_224") else ""), "Check for loss ≈ ln(classes) early and stop the run"],
        ["wandb run id helper removed in wandb 0.30", "Ch. 1", "<code>wandb.util.generate_id</code> no longer exists", "fixed in train_vit.py (commit of Oct 10 16:49)", "Pin or wrap fast-moving library APIs"],
    ]
    if hm_row:
        rows.append(hm_row)
    return chap(10, "ch10", "Mistakes and fixes", "throughout",
        "<p>What went wrong, how did we notice, and what did it cost?</p>",
        "<p>Each row is a real failure from the run logs, notes or commit history.</p>",
        "<p>Most failures here are silent: a run finishes with a plausible number. The checks that caught them were cheap: train accuracy next to val accuracy, the loss compared to ln(number of classes), and reading the effective config from results.json.</p>",
        table(["mistake", "where", "symptom", "fix / outcome", "rule"], rows),
        "<p>Silent misconfiguration (a default flag, a too-high lr) cost more GPU time than any crash. A crash stops; a bad config runs to the end.</p>",
        "<p>Not applicable.</p>")


def int8_rows():
    f = ROOT / "research" / "int8_audit_results.jsonl"
    out = {}
    if not f.exists():
        return []
    for l in f.read_text().splitlines():
        d = json.loads(l)
        v = d.get("variants") or {}
        if "fp32_acc" not in d or not v:
            continue
        st = {k: x for k, x in v.items() if k.startswith("static") and isinstance(x, dict) and x.get("acc") is not None}
        if "dynamic" not in v or not st:
            continue
        bk = max(st, key=lambda k: st[k]["acc"])
        out[d["model"]] = ([d["model"], f"{d['fp32_acc']:.2f}", f"{v['dynamic']['acc']:.2f}",
                    f"{st[bk]['acc']:.2f} ({bk.split('_', 1)[1]})", f"{d['fp32_MB']:.1f} / {v['dynamic']['MB']:.1f} / {st[bk]['MB']:.1f}"])  # last record per model wins
    return list(out.values())


def ch_wave1():
    def row(label, name, note=""):
        return [label, f"<code>{name}</code>", A(name), note]
    rows = [
        row("FlexiViT: IN-64 ViT-Tiny, patch 8 → 4 at 64 px", "ec100_in64pre_tiny_p4_64", "256 tokens, ~4x MACs"),
        row("  same + KD from ViT-B", "ec100_in64pre_tiny_p4_64_kd"),
        row("  patch 4 at 32 px (64 tokens, same cost as p8@64)", "ec100_in64pre_tiny_p4_32"),
        row("IN-21k-P pretrained ViT-Tiny (ours, 30 ep at 64 px), fine-tune 30 ep, patch 8", "ec100_in21kpre_tiny_64", f"IN-64 pretrain: {A('ec100_in64pre_tiny_64')}; with KD {A('ec100_in21kpre_tiny_64_kd')}"),
        row("  IN-21k-P, patch 4 at 64 px", "ec100_in21kpre_tiny_p4_64", f"IN-64 pretrain: {A('ec100_in64pre_tiny_p4_64')}; with KD {A('ec100_in21kpre_tiny_p4_64_kd')}"),
        row("  IN-21k-P, patch 4 at 64 px, KD, 60 ep fine-tune", "ec100_in21kpre_tiny_p4_64_kd_60ep", f"seed 1: {A('ec100_in21kpre_tiny_p4_64_kd_60ep_s1')}"),
        row("  IN-21k-P, patch 8 at 128 px, KD, 60 ep (same 256 tokens as patch 4 at 64 px)", "ec100_in21kpre_tiny_p8_128_kd_60ep", "smaller patches beat larger input at equal tokens"),
        row("  IN-21k-P SL-ViT + SPT (2.75M, ours, 30 ep pretrain), patch 8 at 128 px, KD, 60 ep", "ec100_in21kpre_slvit_spt_p8_128_kd_60ep", f"same setup with ViT-Tiny 5.4M: {A('ec100_in21kpre_tiny_p8_128_kd_60ep')}; from scratch 2.75M: 80.91"),
        row("  IN-21k-P, patch 4 at 64 px, KD, 120 ep fine-tune", "ec100_in21kpre_tiny_p4_64_kd_120ep", f"seed 1: {A('ec100_in21kpre_tiny_p4_64_kd_120ep_s1')}"),
        row("  IN-21k-P, patch 4 at 64 px, KD, 200 ep fine-tune", "ec100_in21kpre_tiny_p4_64_kd_200ep", "best pure ViT at 64 px input; +0.3 over 120 ep, near saturation"),
        row("  IN-21k-P pretraining itself, 60 ep (ViT-Tiny p8 at 64 px; val = IN-21k-P held-out)", "in21kpre_tiny_p8_e60", f"30 ep: {A('in21kpre_tiny_p8')}; ~36.7 GPU-hours at 4 jobs/GPU; its 200-ep KD fine-tune below"),
        row("  IN-21k-P 60-ep pretrain init, patch 4 at 64 px, KD, 200 ep fine-tune", "ec100_in21kpre_e60_tiny_p4_64_kd_200ep", f"negative: 30-ep init gives {A('ec100_in21kpre_tiny_p4_64_kd_200ep')}; 2x pretraining (+0.7 IN-21k val) costs 1.4 on CIFAR-100, behind at every eval (1 seed each)"),
        row("Dense IN-64 ViT-Tiny fine-tune, 30 ep, seed 1", "ec100_dense_in64pre_tiny_64_s1", f"seed 0: {A('ec100_in64pre_tiny_64')}"),
        row("Dense, 60 ep (equal-compute control)", "ec100_dense_in64pre_tiny_64_60ep"),
        row("MoE upcycled E4, 30 ep, seed 0", "ec100_upcyc_in64pre_tiny_e4_64", f"seed 1: {A('ec100_upcyc_in64pre_tiny_e4_64_s1')}"),
        row("MoE upcycled E4, 60 ep", "ec100_upcyc_in64pre_tiny_e4_64_60ep"),
        row("From scratch dense ViT, 150 ep", "c100_dense_vit_150ep"),
        row("  routed MLP-MoE E4 / E8, 150 ep", "c100_moe_vit_e4_150ep", f"E8: {A('c100_moe_vit_e8_150ep')}"),
        row("  routed MLP-MoE E4, 300 ep (dense 300 ep 75.16)", "c100_moe_vit_e4"),
        row("RepViT-M0.9 dense 128 px / CondConv K4 / K8", "ec100_moe_repvit_k4_128", f"dense 87.42, K8 {A('ec100_moe_repvit_k8_128')}"),
        row("RepViT-M0.9 CondConv K4 at 224 px (dense 88.33)", "ec100_moe_repvit_k4_224", "inside ~0.6 seed noise; seed-1 pair queued"),
        row("RepViT-M0.9 dense at 256 px / 288 px", "ec100_repvit_m0_9_256", f"288 px: {A('ec100_repvit_m0_9_288')}"),
        row("RepViT-M0.9 KD from ViT-B, 60 ep: soft / DIST / logit-std", "ec100_kd60_soft_repvit_224", f"DIST {A('ec100_kd60_dist_repvit_224')}, logit-std {A('ec100_kd60_lsd_repvit_224')}"),
        row("RepViT-M0.9 function-matching KD, 30 ep (control)", "ec100_fm_repvit_224_e30"),
        row("RepViT-M0.9 function-matching KD, 200 ep", "ec100_fm_repvit_224_e200", "4.80M params, 0.82 GMACs"),
        row("RepViT-M0.9 function-matching KD, 1000 ep at 128 px", "ec100_fm_repvit_128_e1000", "0.27 GMACs; 30-ep 128 px: 87.42"),
        row("RepViT-M0.9 no teacher, 200 ep (control)", "ec100_nokd_repvit_224_e200", "same recipe, plain labels"),
        row("TinyViT-5M function-matching KD, 200 ep", "ec100_fm_tinyvit5m_224_e200", "5.10M params, 1.26 GMACs: best ≤6M"),
        row("TinyViT-5M / RepViT-M0.9 function-matching KD, 400 ep", "ec100_fm_tinyvit5m_224_e400", f"RepViT: {A('ec100_fm_repvit_224_e400')}; both level with 200 ep (saturated)"),
    ]
    return chap(11, "ch11", "Wave 1: patch size, recurrence ablations, mixture-of-experts", chapter_date(["ec100_in64pre_tiny_p4_64", "c100_moe_vit_e4_150ep"]),
        "<p>After the core recipe, which architectural ideas buy accuracy at a fixed edge budget: smaller patches on a pretrained ViT, a better recurrent block, or more total parameters via mixture-of-experts?</p>",
        "<ul><li><b>FlexiViT patch switch:</b> PI-resize the ImageNet-64 patch-8 embedding to patch 4 and fine-tune (<code>--patch-size 4</code> with a pretrained checkpoint).</li>"
        "<li><b>Recurrent C2 ablations:</b> LoRA per loop step (r8, r32), sandwich norm, add vs concat injection, and an equal-MAC flat 20-layer control.</li>"
        "<li><b>MoE (<code>moe.py</code>, <code>--moe</code>):</b> CondConv in RepViT stage 3, routed top-1 MLP-MoE with a shared expert in the last 4 ViT blocks, and sparse upcycling of the pretrained ViT-Tiny. Active params and MACs matched to dense; stop rule set in advance: at least +1 point.</li></ul>",
        "<p>Smaller patches give more tokens and finer detail for the same weights. MoE adds parameters without adding active compute. LoRA gives each loop step its own small adapter.</p>",
        table(["experiment", "run", "val top-1 %", "note"], rows) +
        "<h4>INT8 export audit (ONNX Runtime CPU, 10k test images)</h4>" +
        table(["model", "FP32 %", "dynamic INT8 %", "static INT8 % (best calibration)", "MB FP32 / dyn / static"], int8_rows()) +
        "<p>Dynamic INT8 (MatMul/Gemm only) is lossless (within 0.31 points) for every model and shrinks the transformers about 3x. Static QDQ PTQ with ORT defaults collapses RepViT, FastViT and our ViT; only DeiT-T and MNv4 survive. For our ViT, quantizing only Conv/Gemm/MatMul gives 81.55 (FP32 81.46), so residual Add and Softmax are the culprits. These static numbers describe ORT's default x86 pipeline and still need a cross-check with other settings and the real target runtime.</p>" +
        "<p><b>Static INT8 fixed for our smallest model (2026-10-13).</b> On the 2.75M CIFAR-100 ViT (SPT + LSA), ORT's full static PTQ gave about 11%. The cause was LSA's diagonal mask: the export adds a -3e4 constant to the attention logits, so ORT quantizes the mask Add output and the softmax input over [-3e4, x] and every real logit lands in one int8 bin. Quantizing q·k<sup>T</sup> before the mask and applying the mask inside the softmax fixes it. A full static int8 graph (every tensor between ops int8, 135 activation quantizers; LayerNorm, softmax and GELU compute in float internally with int8 in/out) scored 80.50 with plain PTQ and <b>80.65 with 10 epochs of QAT</b> (3-seed means in ONNX Runtime, FP32 80.91), at 2.99 MB vs 10.55 MB. On x86 it is still not faster than FP32 (3.4 vs 2.7 ms). Details: <code>research/results_qat_int8.md</code>.</p>" +
        "" f"<p>Recurrent C2 (CIFAR-100, 100 ep, 1 seed): add injection {A('c100_rd_r8_add')} (rerun {A('c100_rd_c2_base_rerun')}), LoRA r8 {A('c100_rd_c2_lora8')}, r32 {A('c100_rd_c2_lora32')}, sandwich {A('c100_rd_c2_sandwich_add')}, concat {A('c100_rd_c2_concat')}; equal-MAC flat-20 control {A('c100_rd_c2_flat20')}. Looping wins about 1.8 to 2 points at equal compute; LoRA does not pay for its parameters.</p>",
        "<p><b>Patient distillation paid most.</b> 200 epochs of function-matching KD from the ViT-B teacher lifted RepViT-M0.9 to " + A("ec100_fm_repvit_224_e200") + " and TinyViT-5M to " + A("ec100_fm_tinyvit5m_224_e200") + " (from 88.33 / 88.31 at 30 epochs), while the same recipe at 30 epochs gave only " + A("ec100_fm_repvit_224_e30") + ". The 200-epoch no-teacher control (same recipe, plain labels) reached only " + A("ec100_nokd_repvit_224_e200") + ", below the 30-epoch fine-tune, so the gain comes from the teacher combined with the long schedule, not from length alone. TinyViT-5M seed 1: " + A("ec100_fm_tinyvit5m_224_e200_s1") + ". <b>Patch size paid; MoE did not.</b> Patch 4 at 64 px added +2.6 points over patch 8 (83.78 vs 81.17) at ~4x MACs. No MoE form beat dense outside seed noise (~0.6 points), and the equal-compute control (dense, 60 epochs, +1.9) beat every MoE variant: many apparent MoE gains are really longer training.</p>",
        "<p>FlexiViT reports patch-size flexibility via PI-resize at ImageNet scale; here it transfers to a 5.4M ViT at 64 px. MoE results in the literature are at far larger data and model scale; at 50k images and ~5M active parameters experts did not specialise (load entropy 0.99 when upcycled).</p>")


def seeds_line():
    import statistics as st
    def ms(names):
        v = [acc(n) for n in names if acc(n) is not None]
        if len(v) < 2:
            return None
        return f"{st.mean(v):.2f} ± {st.stdev(v):.2f} (n={len(v)}: {', '.join(f'{x:.2f}' for x in v)})"
    parts = []
    for label, names in [("CIFAR-100 SL-ViT shape, 1000 ep", ["c100_bf3b_slvit_1000ep", "c100_bf3b_slvit_1000ep_s1", "c100_bf3b_slvit_1000ep_s2"]),
                         ("CIFAR-100 SL-ViT + SPT, 1000 ep", ["c100_spt_slvit_1000ep", "c100_spt_slvit_1000ep_s1", "c100_spt_slvit_1000ep_s2"]),
                         ("CIFAR-10 ViT-Tiny shape, 1000 ep", ["c10_tinyshape_bf3b_1000ep", "c10_tinyshape_bf3b_1000ep_s1", "c10_tinyshape_bf3b_1000ep_s2"]),
                         ("CIFAR-10 ViT-Tiny + SPT, 1000 ep", ["c10_spt_tinyshape_1000ep", "c10_spt_tinyshape_1000ep_s1", "c10_spt_tinyshape_1000ep_s2"]),
                         ("CIFAR-10 SL-ViT + SPT (2.75M), 1000 ep", ["c10_spt_slvit_1000ep", "c10_spt_slvit_1000ep_s1", "c10_spt_slvit_1000ep_s2"]),
                         ("CIFAR-100 SL-ViT + SPT distilled from own 3-seed ensemble, 1000 ep", ["c100_kd_ens3_fm_1000ep", "c100_kd_ens3_fm_1000ep_s1", "c100_kd_ens3_fm_1000ep_s2"]),
                         ("CIFAR-100 2.75M distilled, 2000 ep", ["c100_kd_ens3_fm_2000ep", "c100_kd_ens3_fm_2000ep_s1"]),
                         ("CIFAR-100 1.25M width-128 student, distilled, 1000 ep", ["c100_kd_ens3_fm_d128_1000ep", "c100_kd_ens3_fm_d128_1000ep_s1", "c100_kd_ens3_fm_d128_1000ep_s2"]),
                         ("CIFAR-100 ResNet-18 width 32 control, no teacher, 1000 ep", ["c100_r18w32_sgd_1000ep", "c100_r18w32_sgd_1000ep_s1"]),
                         ("CIFAR-10 ResNet-18 width 32 control, no teacher, 1000 ep", ["c10_r18w32_sgd_1000ep", "c10_r18w32_sgd_1000ep_s1"]),
                         ("CIFAR-100 0.71M width-96 student, distilled, 1000 ep", ["c100_kd_ens3_fm_d96_1000ep", "c100_kd_ens3_fm_d96_1000ep_s1", "c100_kd_ens3_fm_d96_1000ep_s2"]),
                         ("Tiny-ImageNet SL-ViT + SPT (2.9M), 1000 ep", ["tin_spt_slvit_p8_1000ep", "tin_spt_slvit_p8_1000ep_s1"]),
                         ("Tiny-ImageNet 2.9M distilled from 2 copies of itself, 1000 ep", ["tin_kd_ens2_fm_1000ep", "tin_kd_ens2_fm_1000ep_s1"]),
                         ("Tiny-ImageNet 2.9M born-again distilled (teacher: seed-0 copy, 66.16), 1000 ep", ["tin_kd_ba_fm_1000ep", "tin_kd_ba_fm_1000ep_s1"])]:
        m = ms(names)
        if m:
            parts.append(f"{label}: <b>{m}</b>")
    return ("<p><b>Seeds (final-epoch val):</b> " + "; ".join(parts) + ".</p>") if parts else ""


def ch_long():
    def row(label, name, ref, pub):
        r = R(name) or {}
        return [label, f"<code>{name}</code>", A(ref), A(name), f"{r.get('final_train_acc_clean', 0):.2f}" if r else "—", f"{r.get('total_minutes', 0):.0f}" if r else "—", pub]
    rows = [
        row("CIFAR-100, SL-ViT shape, 2.7M, pure ViT", "c100_bf3b_slvit_1000ep", "c100_bf3b_slvit_300ep_lr5e-4", f"SL-ViT {P('slvit_c100')}; CCT hybrid {P('cct_c100')} (300 ep), {P('cct5k_c100')} (5000 ep)"),
        row("CIFAR-100, SL-ViT shape + SPT (full SL-ViT: SPT+LSA), 2.75M", "c100_spt_slvit_1000ep", "c100_spt_slvit_300ep", f"CCT hybrid {P('cct_c100')} (300 ep); SL-ViT {P('slvit_c100')}"),
        row("CIFAR-100, SL-ViT + SPT distilled from our own 3-seed ensemble (function matching, mix on), 2.75M, 1000 ep", "c100_kd_ens3_fm_1000ep", "c100_spt_slvit_1000ep", f"teacher = 3 x 2.75M pure ViTs, CIFAR-only; 300-ep screen {A('c100_kd_ens3_fm_300ep')} vs no-teacher mix-on control {A('c100_mixon_300ep')}; CCT 82.72 (1500 ep)"),
        row("CIFAR-100, multiscale pyramid tokens (64+16+4+1 = 85, shared projection), 300 ep", "c100_ms_shared_300ep", "c100_spt_slvit_300ep", f"seeds 1/2 {A('c100_ms_shared_300ep_s1')} / {A('c100_ms_shared_300ep_s2')}; baseline seed 2 {A('c100_spt_slvit_300ep_s2')}; per-scale proj {A('c100_ms_perscale_300ep')}; Laplacian {A('c100_ms_lap_300ep')}; 21-register control {A('c100_reg21_300ep')} / {A('c100_reg21_300ep_s1')}. No gain (RetinaViT, arXiv 2403.13677, is the same idea)"),
        row("CIFAR-100, 2.75M distilled from the same 3-seed ensemble, 2000 ep (vs 1000 ep)", "c100_kd_ens3_fm_2000ep", "c100_kd_ens3_fm_1000ep", f"seed 1 {A('c100_kd_ens3_fm_2000ep_s1')}; 2-seed mean 83.02, +0.25 over the 1000-ep mean 82.77; level with CCT {P('cct5k_c100')} at 5000 ep; plain training gained only +0.07 from 2000 ep"),
        row("CIFAR-100 CNN control: ResNet-18 width 32 (2.83M), same aug/reg stack, SGD (Tiny-ImageNet screen), 1000 ep", "c100_r18w32_sgd_1000ep", "c100_spt_slvit_1000ep", "same size, no teacher: the ViT (80.91, 3 seeds) is +2.3 ahead"),
        row("CIFAR-100 CNN student: ResNet-18 width 32 distilled from the same 3-seed ViT ensemble (83.27), 1000 ep", "c100_r18w32_kd_ens3_fm_1000ep", "c100_r18w32_sgd_1000ep", "+0.95 from distillation vs +1.86 for the ViT student; ViT student (82.77) stays 3.2 ahead"),
        row("CIFAR-100, width-96 student (6 heads, 0.71M) distilled from the same 3-seed ensemble, 1000 ep", "c100_kd_ens3_fm_d96_1000ep", "c100_kd_ens3_fm_d128_1000ep", f"a quarter of the 2.75M params; above SL-ViT {P('slvit_c100')} (2.9M)"),
        row("CIFAR-100, width-128 student (8 heads, 1.25M) distilled from the same 3-seed ensemble, 1000 ep", "c100_kd_ens3_fm_d128_1000ep", "c100_spt_slvit_1000ep", "less than half the params, +0.5 over the undistilled 2.75M 3-seed mean (80.91); Gani et al. pure ViT 79.15"),
        row("CIFAR-100, distilled from a 4-model teacher (3 seeds + 2000-ep run, ensemble 83.61), 2.75M, 1000 ep", "c100_kd_ens4_fm_1000ep", "c100_kd_ens3_fm_1000ep", "stronger teacher: +0.3 over the 3-model teacher's 3-seed mean 82.77 (1 seed, within ~2 std)"),
        row("CIFAR-100, SL-ViT + SPT, patch 2 (256 tokens, ~4x MACs), 2.75M, 1000 ep", "c100_spt_slvit_p2_1000ep", "c100_spt_slvit_1000ep", "token-count test: +1.1 over the patch-4 3-seed mean; CCT 82.72 at 1500 ep"),
        row("CIFAR-100, SL-ViT + SPT, 2000 ep (vs 1000 ep)", "c100_spt_slvit_2000ep", "c100_spt_slvit_1000ep", "saturated: +0.1 over the 1000-ep 3-seed mean"),
        row("CIFAR-100, ViT-Tiny shape, 5.4M", "c100_tinyshape_bf3b_1000ep", "c100_tinyshape_bf3b", ""),
        row("CIFAR-100, ViT-Tiny shape + SPT, 5.4M", "c100_spt_tinyshape_1000ep", "c100_tinyshape_bf3b", "SPT +1.9 here too; the 2.75M shape is still better"),
        row("CIFAR-10 CNN control: ResNet-18 width 32 (2.81M), same recipe, SGD lr 0.1 wd 5e-4, 1000 ep", "c10_r18w32_sgd_1000ep", "c10_spt_slvit_1000ep", "same size, no teacher: the ViT (96.75, 3 seeds) is +0.6 ahead"),
        row("CIFAR-10, SL-ViT + SPT distilled from own 3-seed ensemble (97.28), 2.75M, 1000 ep", "c10_kd_ens3_fm_1000ep", "c10_spt_slvit_1000ep", f"+0.5 over the plain 3-seed mean (96.75); CCT {P('cct_c10')} (300 ep), 97.48 (1500 ep)"),
        row("CIFAR-10, SL-ViT shape + SPT, 2.75M", "c10_spt_slvit_1000ep", "bf3b_slvit_300ep_lr5e-4", f"half the params of ViT-Tiny, level on seed mean (line above); CCT hybrid {P('cct_c10')} (300 ep), 97.48 (1500 ep)"),
        row("CIFAR-10, ViT-Tiny shape + SPT, 5.4M", "c10_spt_tinyshape_1000ep", "abl_tiny_shape", f"CCT hybrid {P('cct_c10')} (300 ep), 97.48 (1500 ep)"),
        row("CIFAR-10, ViT-Tiny shape, 5.4M", "c10_tinyshape_bf3b_1000ep", "abl_tiny_shape", f"SL-ViT {P('slvit_c10')}; impulse-init ViT-Tiny {P('structinit_c10')}; CCT hybrid {P('cct_c10')} (300 ep), {P('cct5k_c10')} (5000 ep)"),
        row("Tiny-ImageNet 64 px, SL-ViT shape + SPT (patch 8), 2.9M, 300 ep", "tin_spt_slvit_p8_300ep", "tin_p4_tiny", f"SL-ViT {P('slvit_tin')}; Gani et al. {P('gani_tin')} (two-stage); our ResNet-18 {A('tin_r18_full')}"),
        row("Tiny-ImageNet 64 px, SL-ViT shape + SPT (patch 8), 2.9M, 1000 ep", "tin_spt_slvit_p8_1000ep", "tin_spt_slvit_p8_300ep", f"SL-ViT {P('slvit_tin')}; Gani et al. {P('gani_tin')}"),
        row("Tiny-ImageNet, 2.9M distilled from one copy of itself (born-again, function matching), 1000 ep", "tin_kd_ba_fm_1000ep", "tin_spt_slvit_p8_1000ep", f"teacher 66.16; seed 1 {A('tin_kd_ba_fm_1000ep_s1')} (same teacher); 2-seed mean 67.99, +1.8 over its teacher"),
        row("Tiny-ImageNet, 2.9M distilled from a 2-model ensemble of itself (seeds 0+1, ensemble 68.98), 1000 ep", "tin_kd_ens2_fm_1000ep", "tin_kd_ba_fm_1000ep", f"seed 1 {A('tin_kd_ens2_fm_1000ep_s1')}, 2-seed mean 69.48, +0.5 over its teacher ensemble; above every pure transformer <= 10M we found (67.46) and the 2-block conv hybrid at 224 px (68.76); 70.47 hybrid at 224 px still leads"),
        row("Tiny-ImageNet CNN control: ResNet-18 width 32 (2.86M), same aug/reg stack, SGD lr 0.1 wd 5e-4 (held-out screen), 1000 ep", "tin_r18w32_sgd_1000ep", "tin_spt_slvit_p8_1000ep", "same size, no teacher: level with the 2.9M ViT (66.26, 2 seeds)"),
        row("Tiny-ImageNet CNN control: ResNet-18 width 56 (8.65M, widest under 10M), same recipe, 1000 ep", "tin_r18w56_sgd_1000ep", "tin_kd_ens2_fm_1000ep", "under 10M: level with our best distilled ViT at 3x the params"),
        row("Tiny-ImageNet CNN student: ResNet-18 width 32 distilled from the same 2-ViT ensemble (68.98), 1000 ep", "tin_r18w32_kd_ens2_fm_1000ep", "tin_r18w32_sgd_1000ep", "+0.66 from the teacher vs +3.2 for the ViT student (69.48); same-teacher ViT student is 2.3 ahead"),
        row("Tiny-ImageNet CNN control: ResNet-18 standard (11.27M), same recipe, 1000 ep", "tin_r18_sgd_1000ep", "tin_kd_ens2_fm_1000ep", "~4x params: edges our best distilled ViT; above ResNet-18 + AdAutoMix 69.19 (arXiv 2312.11954)"),
        row("Tiny-ImageNet 64 px, ViT-Tiny shape + SPT (patch 8), 5.4M, 300 ep", "tin_spt_tinyshape_p8_300ep", "tin_p4_tiny", ""),
        row("CIFAR-100, recurrent r8 add, 2.7M (500 ep vs 100 ep)", "c100_rd_r8_add_500ep", "c100_rd_r8_add", ""),
        row("CIFAR-100, flat 6-layer control (500 ep vs 100 ep)", "c100_rd_flat6_500ep", "c100_rd_flat6", ""),
    ]
    return chap(12, "ch12", "Long schedules: 500 to 1000 epochs", chapter_date(["c100_tinyshape_bf3b_1000ep"]),
        "<p>Our 300-epoch pure ViTs still improved at the end of training, and CCT reports large gains from 300 to 1500+ epochs. How far does the same recipe go with 3.3x more epochs, no extra data and no architecture change?</p>",
        "<p>Only <code>--epochs 1000 --warmup-epochs 30 --mix-off-epoch 583</code> on the bf3b recipe (mix-off kept at the same 58% of training). Recurrent and flat models: 500 epochs, warmup 25.</p>",
        "<p>Small ViTs on 50k images are optimisation-limited, not capacity-limited, at 300 epochs: strong augmentation means each epoch shows new views, so more epochs keep paying until train accuracy saturates.</p>",
        table(["model", "run", "300/100 ep %", "long run %", "clean train %", "GPU min (shared)", "published"], rows),
        seeds_line() + "<p><b>Schedule length was the biggest single lever left.</b> The 2.7M SL-ViT-shape pure ViT reached " + A("c100_bf3b_slvit_1000ep") + " on CIFAR-100 (+4.1 over 300 epochs) and the 5.4M ViT-Tiny shape reached " + A("c10_tinyshape_bf3b_1000ep") + " on CIFAR-10 (+1.9; seed means in the line above), above every single-stage, CIFAR-only pure ViT we found, and level with the two-stage self-supervised ViT of Gani et al. (" + str(P('gani_c100')) + " / " + str(P('gani_c10')) + "). On CIFAR-100 that is below the CCT conv-tokenizer hybrid at 300 epochs; on CIFAR-10 the 3-seed mean (96.54) matches it (96.53). On CIFAR-100 the smaller 2.7M shape beat the 5.4M one by 2.3 points. The recurrent model's 2-point lead over the flat model at 100 epochs disappeared at 500 epochs (" + A("c100_rd_r8_add_500ep") + " vs " + A("c100_rd_flat6_500ep") + "): looping mostly bought faster convergence, not a better optimum, and the looped model costs more MACs. Adding SL-ViT's shifted patch tokenization (SPT) gave " + A("c100_spt_slvit_300ep") + " at 300 epochs (+1.0) and " + A("c100_spt_slvit_1000ep") + " at 1000 epochs (3-seed mean in the line above, about +1.9 over the no-SPT mean), level with the CCT hybrid at 300 epochs (" + str(P('cct_c100')) + ") while staying a pure ViT. On CIFAR-10, SPT on the ViT-Tiny shape gave " + A("c10_spt_tinyshape_1000ep") + " (3-seed mean in the line above), above CCT at 300 epochs (" + str(P('cct_c10')) + "). Clean train accuracy is 99.9%+ for the 1000-epoch runs.</p>" + (f"<p><b>Held-out check:</b> the recipes were chosen while watching the test set, so the CIFAR-100 SPT recipe was rerun on 45k images with a fixed 5k held-out split (<code>--holdout 5000</code>), scoring the test set once at the end: held-out val {R('c100_spt_slvit_1000ep_holdout')['final_val_acc']:.2f} vs test {R('c100_spt_slvit_1000ep_holdout')['final_test_acc']:.2f}. Selection did not overfit the test set; the run is about 1 point lower because it trains on 10% less data." + (f" CIFAR-10 SPT recipe, same check: held-out {R('c10_spt_tinyshape_1000ep_holdout')['final_val_acc']:.2f} vs test {R('c10_spt_tinyshape_1000ep_holdout')['final_test_acc']:.2f}." if R('c10_spt_tinyshape_1000ep_holdout') else "") + "</p>" if R('c100_spt_slvit_1000ep_holdout') else ""),
        "<p>CCT trains 1500 to 5000 epochs for its best numbers, so comparing at matched epochs matters: at 1000 epochs our best pure ViT (with SPT) is " + A("c100_spt_slvit_1000ep") + " vs CCT's 82.72 at 1500 (hybrid, from the paper's long-schedule table, see research/long_runs.md).</p>")


def ch11():
    return chap(13, "ch13", "What I would do next", "—",
        "<p>Where is the next point of accuracy, or the next unit of cost, most cheaply found?</p>",
        "<ul>"
        f"<li><b>Shifted patch tokenization</b> on the ViT-Tiny-shape bf3b model. It is the one SL-ViT part we never built; the notes rank it first ({cite('spt_gain_c100', 'SPT alone')}: 73.81 → {P('spt_gain_c100')} on CIFAR-100).</li>"
        "<li><b>Seeds on the edge track.</b> Every edge number is one seed; the CIFAR-10 seeds showed ~0.1 to 0.2 points of noise, but fine-tunes at a new resolution may vary more.</li>"
        "<li><b>Seeds and lr sweeps for the edge track.</b> TinyViT-5M got an lr sweep at 128 px; the other students use one lr, and MobileNetV4-Hybrid-M needed 1e-4. A per-model lr check could move several rows.</li>"
        "<li><b>Latency on an idle machine.</b> The CPU timings were taken while training jobs shared the host; rerun <code>edge_bench.py</code> alone and add GPU latency.</li>"
        "<li><b>Recurrent depth with the bf3b recipe at 300 epochs</b>, so the recurrent model is compared to a strong flat baseline, and an exit rule calibrated per dataset.</li>"
        f"<li><b>KD from a strong CNN into the from-scratch ViT at 32 px</b>, the same-data setting where {cite('deith_c100', 'DeiT-Ti hard distillation')} moved {P('deitb_c100')} → {P('deith_c100')} on CIFAR-100.</li>"
        "</ul>",
        "<p>The ablation showed recipe and locality mattered more than shape; SPT and KD add locality and a teacher without adding inference cost.</p>",
        "<p>Pending.</p>", "<p>Pick one change at a time and run three seeds.</p>", "<p>See the Sources section for every number cited.</p>")


SRC_TXT_NOTES = (ROOT / "REPORT_NOTES.md").read_text() if (ROOT / "REPORT_NOTES.md").exists() else ""

CHAPTERS = [("ch1", "Fast pipeline"), ("ch2", "Regularization ladder"), ("ch3", "Width, lr, tokens"), ("ch4", "ResNet-18 control"),
            ("ch5", "Matching published ViTs"), ("ch6", "One-change ablation"), ("ch7", "Harder datasets"), ("ch8", "Recurrent depth"),
            ("ch9", "Edge track"), ("ch10", "Mistakes and fixes"), ("ch11", "Wave 1: patches, recurrence, MoE, INT8, long KD"), ("ch12", "Long schedules"), ("ch13", "Next steps")]


# =================================================================================================== page
CSS = r"""
/* Layout: lab notebook. Sticky section bar, chapter index on the left at wide widths, one column on phones. */
:root{
  --bg:#f3f6f8; --panel:#ffffff; --ink:#15202b; --ink2:#4f5d6b; --rule:#d5dde4; --grid:#e3e9ee;
  --accent:#0d5c7a; --accent-soft:#e1eef3; --code:#eaf0f4;
  --s-vit:#2a78d6; --s-hyb:#eb6834; --s-cnn:#1baf7a; --s-ours:#6250d6; --s-mute:#8a96a3; --s-ref:#9a6700; --s-bad:#c62828;
  --good:#1a7f37; --good-bg:#dcf2e3; --warn:#8a5a00; --warn-bg:#fbefd2; --crit:#b42318; --crit-bg:#fbe3e1;
  --f-display:"Bricolage Grotesque","Avenir Next","Segoe UI",system-ui,sans-serif;
  --f-body:"Source Serif 4","Iowan Old Style","Georgia",serif;
  --f-mono:"IBM Plex Mono","SFMono-Regular",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f161c; --panel:#162029; --ink:#e6edf2; --ink2:#a3b1bd; --rule:#2a3843; --grid:#22303a;
  --accent:#6cc3e0; --accent-soft:#17323e; --code:#1c2a34;
  --s-vit:#3987e5; --s-hyb:#d95926; --s-cnn:#199e70; --s-ours:#9085e9; --s-mute:#7d8a96; --s-ref:#d8a640; --s-bad:#e66767;
  --good:#6fd48f; --good-bg:#16352176; --warn:#e8bb5a; --warn-bg:#3a2e1270; --crit:#f2938b; --crit-bg:#3d1c1a80;
  color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f161c; --panel:#162029; --ink:#e6edf2; --ink2:#a3b1bd; --rule:#2a3843; --grid:#22303a;
  --accent:#6cc3e0; --accent-soft:#17323e; --code:#1c2a34;
  --s-vit:#3987e5; --s-hyb:#d95926; --s-cnn:#199e70; --s-ours:#9085e9; --s-mute:#7d8a96; --s-ref:#d8a640; --s-bad:#e66767;
  --good:#6fd48f; --good-bg:#16352176; --warn:#e8bb5a; --warn-bg:#3a2e1270; --crit:#f2938b; --crit-bg:#3d1c1a80;
  color-scheme:dark}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:4.5rem}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 var(--f-body);-webkit-text-size-adjust:100%}
a{color:var(--accent);text-underline-offset:2px}
a:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:2px;border-radius:3px}
h1,h2,h3,h4{font-family:var(--f-display);line-height:1.2;text-wrap:balance;margin:0}
h1{font-size:clamp(1.9rem,5vw,2.8rem);font-weight:700;letter-spacing:-.01em}
h2{font-size:1.65rem;font-weight:700;margin-block:2.2rem .8rem}
h3{font-size:1.3rem;font-weight:650}
h4{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;color:var(--ink2);font-weight:600;margin-bottom:.35rem}
p{margin:.4rem 0}
code,pre,.n,.tick,.mono{font-family:var(--f-mono)}
code{font-size:.84em;background:var(--code);padding:.05em .3em;border-radius:3px;overflow-wrap:anywhere}
.bar-top{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);border-bottom:1px solid var(--rule)}
.bar-top nav{max-width:84rem;margin:0 auto;padding-inline:16px;display:flex;gap:.25rem;overflow-x:auto;scrollbar-width:thin}
.bar-top a{font:600 .9rem/1 var(--f-display);text-decoration:none;color:var(--ink);padding:.85rem .7rem;white-space:nowrap;border-bottom:3px solid transparent}
.bar-top a:hover{border-bottom-color:var(--accent)}
.bar-top .brand{color:var(--accent);margin-right:auto;padding-left:0}
@media (max-width:520px){.bar-top .brand{display:none}.bar-top nav{justify-content:space-between}}
.wrap{max-width:84rem;margin:0 auto;padding-inline:16px;padding-block:1.5rem 4rem;display:grid;gap:2rem}
@media (min-width:1100px){.wrap{grid-template-columns:14rem minmax(0,1fr)}.toc{position:sticky;top:4rem;align-self:start;max-height:calc(100vh - 5rem);overflow:auto}}
main{min-width:0}
.toc{font:.9rem/1.4 var(--f-display);min-width:0}
.toc h4{margin-top:.2rem}
.toc ol,.toc ul{list-style:none;margin:0 0 1rem;padding:0;display:grid;gap:.1rem}
.toc a{display:flex;gap:.5rem;text-decoration:none;color:var(--ink);padding:.25rem .4rem;border-radius:4px}
.toc a:hover{background:var(--accent-soft)}
.toc .k{font-family:var(--f-mono);color:var(--ink2);font-size:.8rem;min-width:1.4rem}
@media (max-width:1099px){.toc{border:1px solid var(--rule);border-radius:6px;padding:.8rem 1rem;background:var(--panel)}.toc ol{grid-template-columns:repeat(auto-fill,minmax(12rem,1fr))}}
.hero{display:grid;gap:.6rem;padding-block:.5rem 1rem;border-bottom:1px solid var(--rule)}
.hero .lede{font-size:1.1rem;max-width:62ch;color:var(--ink)}
.meta{font:.82rem/1.5 var(--f-mono);color:var(--ink2);display:flex;flex-wrap:wrap;gap:.3rem 1.2rem}
.prose{max-width:68ch}
.tw{overflow-x:auto;margin-block:.8rem;border:1px solid var(--rule);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.86rem;line-height:1.4}
caption{text-align:left;padding:.5rem .7rem;color:var(--ink2);font-size:.8rem}
th,td{padding:.42rem .6rem;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
th{font:600 .74rem/1.3 var(--f-display);text-transform:uppercase;letter-spacing:.05em;color:var(--ink2);background:var(--panel);position:sticky;top:0}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.n{font-size:.82rem}
tbody tr:hover{background:var(--accent-soft)}
.sub{color:var(--ink2);font-size:.8em}
.pill{display:inline-block;font:600 .74rem/1 var(--f-mono);padding:.28rem .5rem;border-radius:999px;white-space:nowrap;border:1px solid currentColor}
.pill.beat{color:var(--good);background:var(--good-bg)}
.pill.match{color:var(--warn);background:var(--warn-bg)}
.pill.below{color:var(--crit);background:var(--crit-bg)}
.pill.pend-pill,.pend{color:var(--ink2);font-style:italic}
.pill.beat::before{content:"▲ "}.pill.below::before{content:"▼ "}.pill.match::before{content:"= "}
.grid2{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(min(100%,30rem),1fr))}
.fig{margin:1rem 0;min-width:0}
.svgw{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:.4rem}
.svgw svg{display:block;width:100%;max-width:860px;min-width:30rem;height:auto;margin-inline:auto}
.chart .dl,.chart .hlt,.chart .bv{paint-order:stroke;stroke:var(--panel);stroke-width:3.5px;stroke-linejoin:round}
figcaption{font-size:.85rem;color:var(--ink2);margin-top:.35rem;max-width:72ch}
.chart text{font-family:var(--f-mono);font-size:11px}
.chart .tick,.chart .axl,.chart .hlt,.chart .pendt{fill:var(--ink2)}
.chart .axl{font-size:11.5px}
.chart .dl,.chart .bl,.chart .bv{fill:var(--ink)}
.chart .bl{font-size:11.5px}
.chart .grid{stroke:var(--grid)}
.chart .axis{stroke:var(--ink2)}
.chart .hl{stroke:var(--s-ref);stroke-width:1.3}
.chart .ln{stroke-width:2;fill:none;stroke-linejoin:round}
.chart .pt{stroke:var(--panel);stroke-width:1.5}
.chart .pt.hollow{fill:var(--panel)!important;stroke-width:2.2}
.chart .ghost{fill:none;stroke-width:1.4}
.vit{--c:var(--s-vit)}.hyb{--c:var(--s-hyb)}.cnn{--c:var(--s-cnn)}.ours{--c:var(--s-ours)}.mute{--c:var(--s-mute)}.ref{--c:var(--s-ref)}.bad{--c:var(--s-bad)}
.chart .bar,.chart circle.pt,.chart rect.pt,.chart path.pt{fill:var(--c)}
.chart .pt.hollow{stroke:var(--c)}
.chart .ln,.chart .ghost{stroke:var(--c)}
.chart .ln.mute{stroke-dasharray:none;opacity:.9}
.legend{display:flex;flex-wrap:wrap;gap:.3rem 1rem;font:.8rem/1.3 var(--f-mono);color:var(--ink2);margin-bottom:.4rem}
.legend span{display:inline-flex;align-items:center;gap:.4rem}
.sw{display:inline-block;width:.8rem;height:.8rem;background:var(--c)}
.sw.circle{border-radius:50%}.sw.square{border-radius:2px}.sw.diamond{transform:rotate(45deg) scale(.8)}
.sw.ring{background:transparent;border:2px solid var(--ink2);border-radius:50%}
.sw.line{height:3px;width:1.2rem}.sw.dash{height:0;width:1.2rem;border-top:2px dashed var(--s-ref);background:none}
.sw.bar{height:.55rem;width:1.2rem;border-radius:2px}.sw.ghost{height:.8rem;width:1.2rem;background:none;border:1.5px solid var(--c);border-radius:3px}
.cli{background:var(--code);border-radius:6px;padding:.6rem .8rem;overflow-x:auto;font-size:.76rem;line-height:1.5;margin:.5rem 0;white-space:pre}
.cli code{background:none;padding:0;overflow-wrap:normal}
.chapter{border-top:2px solid var(--ink);padding-top:1rem;margin-top:2.2rem}
.ch-head{display:flex;gap:1rem;align-items:baseline}
.ch-num{font:700 2.4rem/1 var(--f-display);color:var(--accent);font-variant-numeric:tabular-nums}
.ch-date{font:.78rem var(--f-mono);color:var(--ink2);margin:.2rem 0 0}
.ch-grid{display:grid;gap:1rem 2rem;margin-top:1rem;grid-template-columns:minmax(0,1fr)}
@media (min-width:900px){.ch-grid{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.ch-grid .res{grid-column:1/-1}}
.part{min-width:0}
.part ul{padding-left:1.1rem;margin:.3rem 0}
.part li{margin:.2rem 0}
.part.q p:first-child{font-size:1.06rem}
.part.res{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:1rem}
.part.pub{border-left:3px solid var(--s-ref);padding-left:.9rem}
.plist{columns:2 20rem;font-size:.88rem}
.imgs{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(min(100%,22rem),1fr))}
.imgs img{width:100%;height:auto;border-radius:6px;border:1px solid var(--rule);background:#fff}
.note{font-size:.88rem;color:var(--ink2)}
footer{color:var(--ink2);font-size:.82rem;border-top:1px solid var(--rule);padding-top:1rem;margin-top:3rem}
"""

FONTS = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,650;12..96,700&family=IBM+Plex+Mono:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">'
TITLE = "Small ViT Field Notes"


def sources_section():
    rows = []
    for k in USED_PUB:
        lab, val, setting, url, f, needle = PUB[k]
        ok = "" if k not in PUB_MISSING else ' <span class="pill below">not found in source file</span>'
        rows.append([esc(lab), f"{val:g}", esc(setting), f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(url.replace("https://", ""))}</a>', f"<code>{SRC_FILES[f].relative_to(ROOT)}</code>{ok}"])
    return table(["result", "value", "setting", "link", "verified in"], rows, num_cols=(1,))


def images_block():
    picks = [("report_assets/c100_cls_attention.png", "CLS attention maps of the CIFAR-100 ViT-Tiny (c100_t6_full_300ep): attention lands on the object."),
             ("report_assets/pos_embed_similarity.png", "Learned position embeddings (CIFAR-10 ViT-Tiny): each patch's embedding is most similar to its spatial neighbours. The model learned a 2-D grid from data.")]
    out = []
    for p, cap in picks:
        uri = png_uri(ROOT / p, 900)
        if uri:
            out.append(f'<figure class="fig"><img src="{uri}" alt="{esc(cap)}" loading="lazy" width="900"><figcaption>{esc(cap)}</figcaption></figure>')
    return '<div class="imgs">' + "".join(out) + "</div>" if out else ""


def build_body():
    b3 = acc("bf3b_slvit_300ep_lr5e-4")
    ts = acc("abl_tiny_shape")
    c100 = max([v for v in (acc("c100_bf3b_slvit_300ep_lr5e-4"), acc("c100_tinyshape_bf3b")) if v is not None], default=None)
    n_done = COST["n_done"]
    stories = [ch1(), ch2(), ch3(), ch4(), ch5(), ch6(), ch7(), ch8(), ch9(), ch10(), ch_wave1(), ch_long(), ch11()]
    dash = f"""
<section id="dashboard" aria-labelledby="dash-h">
  <h2 id="dash-h">Dashboard</h2>
  <h3>Scoreboard against published results</h3>
  <p class="prose">Our best numbers per setting next to the published result for that setting. Pills: <span class="pill beat">beat</span> more than 0.5 points above, <span class="pill match">matched</span> within 0.5, <span class="pill below">below</span> more than 0.5 under. "% of it" is our accuracy divided by the published one.</p>
  {scoreboard()}
  <h3 style="margin-top:2rem">CIFAR-100 edge track: accuracy vs cost</h3>
  <p class="prose">Every finished fine-tune of a pretrained model (ec100_*). Teachers are the two runs at the top right. The step line is the Pareto frontier: no other run is both cheaper and more accurate.</p>
  {PARETO_LEGEND}
  {fig(pareto_chart("params", "params (M)", "Accuracy vs parameters"), "Parameters (M, 100-class head).")}
  {fig(pareto_chart("gmacs", "GMACs", "Accuracy vs GMACs"), "Multiply-accumulates per image at the run's input size.")}
  {fig(pareto_chart("int8_t1", "CPU int8 latency, 1 thread (ms)", "Accuracy vs CPU int8 latency"), "onnxruntime int8 (dynamic quantization), batch 1, one thread. <b>These latencies were measured while the GPU and CPU were shared with running training jobs, so they are noisy</b>; use them to rank models roughly, not as device numbers.")}
  <h3>Smallest model within X%</h3>
  {smallest_within()}
  <h3>Seed variance</h3>
  <p class="prose">Most of our comparisons are single runs. These repeats show how much a number moves with only the seed changed.</p>
  {seed_table()}
  <h3>Compute and cost</h3>
  {cost_block()}
  <h3>Still running or queued</h3>
  {pending_block()}
</section>"""
    toc_ch = "".join(f'<li><a href="#{cid}"><span class="k">{i + 1:02d}</span>{esc(t)}</a></li>' for i, (cid, t) in enumerate(CHAPTERS))
    toc = f"""<aside class="toc" aria-label="Contents"><h4>Sections</h4><ul><li><a href="#dashboard"><span class="k">A</span>Dashboard</a></li><li><a href="#story"><span class="k">B</span>The story</a></li><li><a href="#data"><span class="k">C</span>All runs</a></li><li><a href="#sources"><span class="k">D</span>Sources</a></li></ul><h4>Chapters, in the order they happened</h4><ol>{toc_ch}</ol></aside>"""
    data = f"""
<section id="data" aria-labelledby="data-h">
  <h2 id="data-h">All runs</h2>
  <p class="prose">Final val accuracy (best in parentheses when different), clean train accuracy (no augmentation), wall minutes and steady-state images per second. Minutes and img/s of overlapping runs include slowdown from sharing the GPU.</p>
  <h3>CIFAR-10</h3>{ds_table("cifar10")}
  <h3>CIFAR-100, from scratch</h3>{ds_table("cifar100")}
  <h3>CIFAR-100, pretrained fine-tunes (edge track)</h3>{edge_table()}
  <h3>Tiny-ImageNet</h3>{ds_table("tiny_imagenet")}
  <h3>ImageNet-64</h3>{ds_table("imagenet64")}
  <h3>What the models learned</h3>{images_block()}
</section>"""
    src = f"""
<section id="sources" aria-labelledby="src-h">
  <h2 id="src-h">Sources</h2>
  <p class="prose">Every published number on this page, its setting and its link. Each value was copied from <code>REFERENCES.md</code> or <code>research/*.md</code>, which quote the source page; the generator checks that each value appears in that file.</p>
  {sources_section()}
</section>"""
    hero = f"""
<header class="hero">
  <h1>{TITLE}</h1>
  <p class="lede">Training small vision transformers on CIFAR-10, CIFAR-100, Tiny-ImageNet and ImageNet-64 on one H100. A from-scratch ViT reached {num(b3)}% on CIFAR-10 with the SL-ViT shape and {num(ts)}% with the ViT-Tiny shape, against 93.6 to 94.5% published. On CIFAR-100 it reached {num(c100)}% from scratch, and a pretrained 5M-parameter student reached {num(BEST_SMALL['acc'] if BEST_SMALL else None)}%.</p>
  <div class="meta"><span>{n_done} finished runs</span><span>{COST['n_live']} running</span><span>{(COST['run_min'] + COST['live_min']) / 60:.1f} GPU-h of run time</span><span>git {GIT_FIRST} → {GIT_LAST} ({GIT_N} commits)</span><span>generated {time.strftime('%Y-%m-%d %H:%M')}</span></div>
</header>"""
    story = '<section id="story" aria-labelledby="story-h"><h2 id="story-h">The story, step by step</h2><p class="prose">Thirteen chapters in the order the work happened. Each one states the question, the exact command-line change, the mechanism, the result, the lesson, and how it compares with published work.</p>' + "".join(stories) + "</section>"
    nav = '<div class="bar-top"><nav aria-label="Sections"><a class="brand" href="#top">Small ViT Field Notes</a><a href="#dashboard">Dashboard</a><a href="#story">Story</a><a href="#data">Data</a><a href="#sources">Sources</a></nav></div>'
    foot = f'<footer>Generated by <code>ViT/make_dashboard.py</code> from <code>ViT/runs/*</code>. Rerun it to refresh. Repository: <a href="https://github.com/RNA8/vision_models" target="_blank" rel="noopener">github.com/RNA8/vision_models</a>.</footer>'
    return nav + f'<div class="wrap" id="top">{toc}<main>{hero}{dash}{story}{data}{src}{foot}</main></div>'


def atomic_write(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main():
    body = build_body()
    style = f"<style>{CSS}</style>"
    full = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"<title>{TITLE}</title>\n{style}\n{FONTS}\n"
            "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}</style>\n"
            f"</head>\n<body>\n{body}\n</body>\n</html>\n")
    frag = f"<title>{TITLE}</title>\n{style}\n{FONTS}\n{body}\n"
    atomic_write(ROOT / "DASHBOARD.html", full)
    atomic_write(ROOT / "dashboard_artifact.html", frag)
    print(f"DASHBOARD.html {len(full) / 1e6:.2f} MB, dashboard_artifact.html {len(frag) / 1e6:.2f} MB")
    print(f"runs: {COST['n_done']} done, {COST['n_live']} running/stalled; published numbers used: {len(USED_PUB)}")
    if PUB_MISSING:
        print("WARNING: published values not found in their source files:", PUB_MISSING)


if __name__ == "__main__":
    main()
