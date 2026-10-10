#!/usr/bin/env python3
"""
eval_recurrent.py: test-time depth analysis of a recurrent-depth ViT run (train_vit.py --model rd_*).

Given a run dir with model.pt + results.json, on the test set:
  (a) depth sweep: accuracy (and NLL) vs r = 1..R (default 32), with analytic FLOPs per r. Does accuracy
      keep improving past the training r_mean (extrapolation), or does it get worse (overthinking, 2502.10954)?
  (b) per-image adaptive exit: stop at the first t >= 2 with KL(p_t || p_{t-1}) < tau (Geiping et al.
      2502.05171 use tau = 5e-4) or |max p_t - max p_{t-1}| < tau. A tau sweep gives accuracy vs average
      iterations, against the fixed-r curve. An oracle exit (first step that is correct, if any) is an upper bound.
  (c) difficulty: Spearman between exit step and (i) per-image loss under a reference model
      (--ref-run-dir, e.g. a ResNet-18 run), (ii) CIFAR-10-C corruption severity (--cifar10c-dir;
      0 = clean, 1-5; a seeded subset of --c10c-per-severity test indices, the same indices at every
      severity, on --c10c-corruptions) and (iii) human uncertainty from CIFAR-10H (--cifar10h; Peterson
      et al. 2019, ~51 annotators per test image): entropy of the human label distribution and
      1 - p_human(true label). (ii) and (iii) apply to cifar10 runs only and are skipped if the files are
      absent. tau for (c) is --tau, or automatically the largest KL tau whose accuracy is at least the
      fixed r = r_mean accuracy (fewest iterations at no accuracy loss).

Data (all optional, under data/): CIFAR-10-C/ (Zenodo 2535967: <corruption>.npy (50000, 32, 32, 3) =
5 severities x the 10k test images in test-set order, labels.npy), cifar10h-probs.npy (10000, 10), rows
in the same order as the CIFAR-10 test set (the uoft-cs/cifar10 HF test split matches the original
test_batch order; argmax agrees with our labels on 99.21%).
  (d) state convergence: ||s_t - s_{t-1}|| / ||s_t|| vs t.

The exit is simulated offline from the full R-step rollout, which gives exactly what an online
early-exit loop would give per image (each image's exit depends only on its own predictions).
Effective FLOPs for an exit rule count the coda + head at every step (the rule needs p_t each step).

Writes <run_dir>/recurrent_eval.json and recurrent_{depth_sweep,exit_pareto,difficulty,state_convergence}.png.

  python eval_recurrent.py runs/rd_pilot_a --ref-run-dir runs/r18_full
  python eval_recurrent.py runs/x --fake-data --n-test 512 --r-max 8      # CPU smoke test
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_vit  # noqa: E402

GEIPING_TAU = 5e-4
# one noise, one blur, one weather and two digital corruptions
C10C_DEFAULT = ("gaussian_noise", "motion_blur", "snow", "contrast", "jpeg_compression")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("run_dir")
    p.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    p.add_argument("--fake-data", action="store_true")
    p.add_argument("--n-test", type=int, default=0, help="use the first N test images; 0 = all")
    p.add_argument("--r-max", type=int, default=32, help="depth sweep r = 1..r_max")
    p.add_argument("--bs", type=int, default=1000)
    p.add_argument("--tau", type=float, default=None, help="KL exit threshold for (c); default: auto")
    p.add_argument("--ref-run-dir", default=None, help="reference model run dir for per-image loss (c.i)")
    p.add_argument("--cifar10c-dir", default=os.path.join(HERE, "data", "CIFAR-10-C"),
                   help="dir with <corruption>.npy + labels.npy (standard CIFAR-10-C layout)")
    p.add_argument("--c10c-per-severity", "--c10c-n", dest="c10c_n", type=int, default=1000,
                   help="test images per corruption-severity (a seeded subset, shared by all severities)")
    p.add_argument("--c10c-corruptions", default=",".join(C10C_DEFAULT),
                   help="comma list of corruption names, or 'all' for every .npy found")
    p.add_argument("--cifar10h", default=os.path.join(HERE, "data", "cifar10h-probs.npy"),
                   help="CIFAR-10H human label distributions (10000, 10), CIFAR-10 test order")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="suffix for output files")
    return p.parse_args()


# --------------------------------------------------------------------------- loading
def load_run_args(run_dir):
    """Saved args merged over train_vit's current defaults (older runs may lack newer flags)."""
    argv, sys.argv = sys.argv, ["train_vit.py"]
    try:
        ns = train_vit.parse_args()
    finally:
        sys.argv = argv
    with open(os.path.join(run_dir, "results.json")) as f:
        saved = json.load(f)["args"]
    vars(ns).update(saved)
    return ns


def load_model(run_dir, num_classes, img_size, device):
    a = load_run_args(run_dir)
    m = train_vit.build_model(a, num_classes, img_size)
    sd = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.to(device).eval(), a


# --------------------------------------------------------------------------- stats
def rankdata(a):
    """Average ranks (ties share the mean rank), as scipy.stats.rankdata."""
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size)
    a = a[sorter]
    obs = np.r_[True, a[1:] != a[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------- core computations
@torch.no_grad()
def rollout_all(model, x_u8, mean, std, r_max, bs, device):
    """Log-probs (R, N, C) and state deltas (R, N) on the CPU, float32."""
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    logps, deltas = [], []
    for k in range(0, x_u8.size(0), bs):
        with amp:
            out = model.rollout(train_vit.normalize(x_u8[k:k + bs].to(device), mean, std), r_max)
        logps.append(F.log_softmax(out["logits"].float(), -1).cpu())
        deltas.append(out["delta"].float().cpu())
    return torch.cat(logps, 1), torch.cat(deltas, 1)


def exit_metrics(logp):
    """Per-step change metrics, (R, N); entry t compares step t with t-1 (entry 0 = +inf)."""
    p = logp.exp()
    kl = (p[1:] * (logp[1:] - logp[:-1])).sum(-1).clamp_min(0)
    mp = (p[1:].amax(-1) - p[:-1].amax(-1)).abs()
    inf = torch.full_like(kl[:1], float("inf"))
    return {"kl": torch.cat([inf, kl]), "maxprob": torch.cat([inf, mp])}


def exit_steps(metric, tau):
    """1-based exit step per image: the first t >= 2 with metric < tau, else R."""
    R = metric.size(0)
    hit = metric < tau
    first = torch.where(hit.any(0), hit.float().argmax(0), torch.full_like(hit[0], R - 1, dtype=torch.long))
    return first + 1


def evaluate_exit(logp, y, steps, model):
    pred = logp.argmax(-1).gather(0, (steps - 1)[None]).squeeze(0)
    st = steps.double()
    flops = np.mean([model.flops(int(t), coda_evals=int(t)) for t in steps.tolist()])
    return {"acc": 100.0 * (pred == y).double().mean().item(), "avg_iters": st.mean().item(),
            "median_iters": st.median().item(), "gflops": flops / 1e9}


# --------------------------------------------------------------------------- CIFAR-10-C
def cifar10c(model, args, mean, std, device, tau, r_max, x_clean, y_clean):
    """Mean exit step and accuracy vs severity (0 = clean). Every severity uses the same seeded subset
    of test indices, so severity is the only thing that changes between groups."""
    d = args.cifar10c_dir
    if not d or not os.path.isfile(os.path.join(d, "labels.npy")):
        print(f"(c.ii) CIFAR-10-C not found at {d}; skipping")
        return None
    labels = np.load(os.path.join(d, "labels.npy"))
    m = len(labels) // 5  # images per severity (10000 in the official release)
    avail = min(m, x_clean.size(0))
    n = min(args.c10c_n, avail)
    idx = np.sort(np.random.default_rng(args.seed).choice(avail, n, replace=False))
    if not np.array_equal(labels[idx], y_clean.numpy()[idx]):
        print("(c.ii) WARNING: CIFAR-10-C labels.npy does not match the test labels at the sampled indices")
    if args.c10c_corruptions.strip().lower() == "all":
        names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(d, "*.npy"))
                       if not f.endswith("labels.npy"))
    else:
        names = [c.strip() for c in args.c10c_corruptions.split(",") if c.strip()]
    missing = [c for c in names if not os.path.isfile(os.path.join(d, f"{c}.npy"))]
    if missing:
        print(f"(c.ii) missing corruption files, skipped: {missing}")
    names = [c for c in names if c not in missing]
    if not names:
        print("(c.ii) no CIFAR-10-C corruptions to evaluate; skipping")
        return None

    def run(x, yy):
        lp, _ = rollout_all(model, x, mean, std, r_max, args.bs, device)
        st = exit_steps(exit_metrics(lp)["kl"], tau)
        pred = lp.argmax(-1).gather(0, (st - 1)[None]).squeeze(0)
        r_m = min(model.r_mean, r_max) - 1
        return st, 100.0 * (pred == yy).double().mean().item(), 100.0 * (lp[r_m].argmax(-1) == yy).double().mean().item()

    t0 = time.time()
    st0, acc0, accr0 = run(x_clean[torch.from_numpy(idx)], y_clean[torch.from_numpy(idx)])
    out = {"tau": tau, "n_per_severity": int(n), "seed": args.seed, "corruptions": {},
           "severity0": {"mean_exit": st0.double().mean().item(), "acc_at_exit": acc0, "acc_at_r_mean": accr0}}
    all_sev, all_st = [np.zeros(n)], [st0.numpy()]
    for name in names:
        arr = np.load(os.path.join(d, f"{name}.npy"), mmap_mode="r")
        rec = {"mean_exit": [out["severity0"]["mean_exit"]], "acc_at_exit": [acc0], "acc_at_r_mean": [accr0]}
        sevs, sts = [np.zeros(n)], [st0.numpy()]
        for s in range(1, 6):
            rows = (s - 1) * m + idx
            x = torch.from_numpy(np.ascontiguousarray(arr[rows])).permute(0, 3, 1, 2).contiguous()
            st, acc, accr = run(x, torch.from_numpy(labels[rows]).long())
            rec["mean_exit"].append(st.double().mean().item())
            rec["acc_at_exit"].append(acc)
            rec["acc_at_r_mean"].append(accr)
            sevs.append(np.full(n, s))
            sts.append(st.numpy())
        rec["sev_spearman"] = spearman(np.concatenate(sts), np.concatenate(sevs))
        out["corruptions"][name] = rec
        all_sev += sevs[1:]
        all_st += sts[1:]
        print(f"  C10-C {name:18s} mean exit sev 0..5: " + " ".join(f"{v:5.2f}" for v in rec["mean_exit"])
              + f" | acc@exit " + " ".join(f"{v:5.1f}" for v in rec["acc_at_exit"])
              + f" | rho {rec['sev_spearman']:+.3f}")
    out["pooled_sev_spearman"] = spearman(np.concatenate(all_st), np.concatenate(all_sev))
    for k in ("mean_exit", "acc_at_exit", "acc_at_r_mean"):
        out[f"{k}_by_severity"] = [float(np.mean([c[k][s] for c in out["corruptions"].values()])) for s in range(6)]
    out["seconds"] = time.time() - t0
    return out


# --------------------------------------------------------------------------- CIFAR-10H
def cifar10h(args, st, pred, y):
    """Spearman of exit step vs human uncertainty; st/pred/y are for test images 0..N-1."""
    path = args.cifar10h
    if not path or not os.path.isfile(path):
        print(f"(c.iii) CIFAR-10H not found at {path}; skipping")
        return None
    probs = np.load(path).astype(np.float64)
    n = st.size(0)
    if probs.ndim != 2 or probs.shape[0] < n:
        print(f"(c.iii) CIFAR-10H shape {probs.shape} does not cover {n} test images; skipping")
        return None
    probs, yy, st_np = probs[:n], y.numpy(), st.numpy().astype(np.float64)
    agree = float((probs.argmax(1) == yy).mean())
    if agree < 0.9:
        print(f"(c.iii) WARNING: CIFAR-10H argmax agrees with the test labels on only {100 * agree:.1f}% "
              "(wrong order or fake data?)")
    ent = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(1)
    unc = 1.0 - probs[np.arange(n), yy]
    # equal-count entropy deciles; ~44% of CIFAR-10H images have entropy exactly 0, so ties are broken
    # at random (seeded) and the lowest deciles all hold zero-entropy images
    tie = np.random.default_rng(args.seed).random(n)
    order = np.lexsort((tie, ent))
    bins = np.array_split(order, min(10, n))
    correct = (pred == y).numpy()
    out = {"path": path, "n": int(n), "argmax_agreement": agree,
           "spearman_exit_vs_entropy": spearman(st_np, ent),
           "spearman_exit_vs_1_minus_p_true": spearman(st_np, unc),
           "frac_zero_entropy": float((ent < 1e-12).mean()), "mean_entropy": float(ent.mean()),
           "decile_mean_entropy": [float(ent[b].mean()) for b in bins],
           "decile_mean_exit": [float(st_np[b].mean()) for b in bins],
           "decile_acc_at_exit": [float(100.0 * correct[b].mean()) for b in bins]}
    nz = ent > 1e-12
    out["spearman_exit_vs_entropy_nonzero_only"] = spearman(st_np[nz], ent[nz]) if nz.sum() > 2 else None
    return out


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    t0 = time.time()

    run_args = load_run_args(args.run_dir)
    assert run_args.model.startswith("rd_"), f"{args.run_dir} is not a recurrent-depth run ({run_args.model})"
    data_ns = argparse.Namespace(dataset=run_args.dataset, data_dir=args.data_dir, fake_data=args.fake_data)
    _, _, xte, yte, ds = train_vit.load_data(data_ns, "cpu")
    if args.n_test:
        xte, yte = xte[:args.n_test], yte[:args.n_test]
    mean = torch.tensor(ds.MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(ds.STD, device=device).view(1, 3, 1, 1)
    img_size = xte.size(-1)
    model, _ = load_model(args.run_dir, ds.NUM_CLASSES, img_size, device)
    R, r_mean = args.r_max, model.r_mean
    y = yte.long()
    print(f"{args.run_dir}: {run_args.model} r_mean={r_mean} k={model.k_backprop} inject={model.inject} "
          f"| {xte.size(0)} test images | r = 1..{R} on {device}")

    logp, delta = rollout_all(model, xte, mean, std, R, args.bs, device)

    # (a) depth sweep
    correct = logp.argmax(-1) == y[None]
    acc_r = (100.0 * correct.double().mean(1)).tolist()
    nll_r = (-logp.gather(-1, y[None, :, None].expand(R, -1, 1)).squeeze(-1)).mean(1).tolist()
    gflops_r = [model.flops(r) / 1e9 for r in range(1, R + 1)]
    best_r = int(np.argmax(acc_r)) + 1
    sweep = {"r": list(range(1, R + 1)), "acc": acc_r, "nll": nll_r, "gflops": gflops_r, "best_r": best_r,
             "acc_at_r_mean": acc_r[min(r_mean, R) - 1], "best_acc": acc_r[best_r - 1],
             "acc_at_2r_mean": acc_r[min(2 * r_mean, R) - 1], "acc_at_r_max": acc_r[-1]}
    print(f"(a) acc@r_mean({r_mean}) {sweep['acc_at_r_mean']:.2f} | best {sweep['best_acc']:.2f} at r={best_r} "
          f"| acc@2r_mean {sweep['acc_at_2r_mean']:.2f} | acc@{R} {acc_r[-1]:.2f}")

    # (b) adaptive exit
    mets = exit_metrics(logp)
    taus = {"kl": sorted(set(np.logspace(-6, 0, 25).tolist() + [GEIPING_TAU])),
            "maxprob": np.logspace(-5, -0.3, 20).tolist()}
    exits = {}
    for name, ts in taus.items():
        exits[name] = []
        for tau in ts:
            st = exit_steps(mets[name], tau)
            exits[name].append({"tau": tau, **evaluate_exit(logp, y, st, model)})
    oracle_hit = correct.any(0)
    oracle_st = torch.where(oracle_hit, correct.float().argmax(0) + 1, torch.full_like(y, R))
    oracle = {"acc": 100.0 * oracle_hit.double().mean().item(), "avg_iters": oracle_st.double().mean().item()}
    g = next(e for e in exits["kl"] if e["tau"] == GEIPING_TAU)
    print(f"(b) KL exit tau=5e-4: acc {g['acc']:.2f} at {g['avg_iters']:.2f} avg iters "
          f"(fixed r_mean: {sweep['acc_at_r_mean']:.2f} at {r_mean}) | oracle {oracle['acc']:.2f} "
          f"at {oracle['avg_iters']:.2f}")

    # tau for (c): fewest iterations with no accuracy loss vs fixed r_mean
    if args.tau is not None:
        tau_c, tau_rule = args.tau, "user"
    else:
        ok = [e for e in exits["kl"] if e["acc"] >= sweep["acc_at_r_mean"]]
        tau_c = max(ok, key=lambda e: e["tau"])["tau"] if ok else min(taus["kl"])
        tau_rule = "largest KL tau with acc >= fixed r_mean acc" if ok else "smallest tau (none matched)"
    st_c = exit_steps(mets["kl"], tau_c)
    exit_c = evaluate_exit(logp, y, st_c, model)
    hist = np.bincount(st_c.numpy(), minlength=R + 1)[1:].tolist()
    print(f"(c) tau = {tau_c:.2e} ({tau_rule}): acc {exit_c['acc']:.2f}, avg iters {exit_c['avg_iters']:.2f}")

    # (c.i) difficulty under a reference model, plus the model's own loss as a sanity check
    pred_c = logp.argmax(-1).gather(0, (st_c - 1)[None]).squeeze(0)
    own_loss = -logp[min(r_mean, R) - 1].gather(-1, y[:, None]).squeeze(-1)
    diff = {"tau": tau_c, "tau_rule": tau_rule, "exit_hist": hist, **exit_c,
            "spearman_exit_vs_own_loss_at_r_mean": spearman(st_c.numpy(), own_loss.numpy()),
            "mean_exit_correct": st_c[pred_c == y].double().mean().item() if (pred_c == y).any() else None,
            "mean_exit_wrong": st_c[pred_c != y].double().mean().item() if (pred_c != y).any() else None}
    ref_loss = None
    if args.ref_run_dir:
        ref, ref_args = load_model(args.ref_run_dir, ds.NUM_CLASSES, img_size, device)
        losses = []
        with torch.no_grad():
            for k in range(0, xte.size(0), args.bs):
                xb = train_vit.normalize(xte[k:k + args.bs].to(device), mean, std)
                losses.append(F.cross_entropy(ref(xb).float(), y[k:k + args.bs].to(device), reduction="none").cpu())
        ref_loss = torch.cat(losses)
        diff.update(ref_run=args.ref_run_dir, ref_model=ref_args.model,
                    spearman_exit_vs_ref_loss=spearman(st_c.numpy(), ref_loss.numpy()),
                    ref_mean_loss=ref_loss.mean().item())
        print(f"(c.i) Spearman(exit step, {ref_args.model} loss) = {diff['spearman_exit_vs_ref_loss']:+.3f}")
    print(f"(c.i) Spearman(exit step, own loss at r_mean) = {diff['spearman_exit_vs_own_loss_at_r_mean']:+.3f}")
    c10c = c10h = None
    if run_args.dataset == "cifar10":
        c10c = cifar10c(model, args, mean, std, device, tau_c, R, xte, y)
        c10h = cifar10h(args, st_c, pred_c, y)
    else:
        print("(c.ii/iii) CIFAR-10-C and CIFAR-10H only apply to cifar10 runs; skipping")
    if c10c:
        print(f"(c.ii) mean exit by severity 0..5: " + " ".join(f"{v:.2f}" for v in c10c["mean_exit_by_severity"])
              + f" | pooled Spearman(exit step, severity) = {c10c['pooled_sev_spearman']:+.3f}"
              + f" ({c10c['n_per_severity']}/severity, {len(c10c['corruptions'])} corruptions, {c10c['seconds']:.0f}s)")
    if c10h:
        print(f"(c.iii) CIFAR-10H: Spearman(exit, human entropy) = {c10h['spearman_exit_vs_entropy']:+.3f}, "
              f"Spearman(exit, 1 - p_human(true)) = {c10h['spearman_exit_vs_1_minus_p_true']:+.3f} "
              f"(argmax agreement {100 * c10h['argmax_agreement']:.1f}%)")
        print("(c.iii) mean exit by human-entropy decile: " + " ".join(f"{v:.2f}" for v in c10h["decile_mean_exit"]))

    # (d) state convergence
    dn = delta.numpy()
    conv = {"t": list(range(1, R + 1)), "mean": dn.mean(1).tolist(), "median": np.median(dn, 1).tolist(),
            "q10": np.quantile(dn, 0.1, 1).tolist(), "q90": np.quantile(dn, 0.9, 1).tolist()}
    print(f"(d) ||s_t - s_t-1|| / ||s_t||: t=1 {conv['mean'][0]:.3f}, t={r_mean} {conv['mean'][min(r_mean, R) - 1]:.4f}, "
          f"t={R} {conv['mean'][-1]:.4f}")

    out = {"run_dir": os.path.abspath(args.run_dir), "model": run_args.model, "r_mean": r_mean,
           "k_backprop": model.k_backprop, "inject": model.inject, "n_test": int(xte.size(0)), "r_max": R,
           "device": device, "seed": args.seed, "depth_sweep": sweep,
           "adaptive_exit": {"kl": exits["kl"], "maxprob": exits["maxprob"], "oracle": oracle},
           "difficulty": diff, "cifar10c": c10c, "cifar10h": c10h, "state_convergence": conv,
           "seconds": time.time() - t0}
    tag = f"_{args.tag}" if args.tag else ""
    path = os.path.join(args.run_dir, f"recurrent_eval{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    make_plots(out, args.run_dir, tag, st_c, ref_loss)
    print(f"Saved {path} and plots ({out['seconds']:.0f}s)")


def make_plots(out, run_dir, tag, st_c, ref_loss):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sw, r_mean, name = out["depth_sweep"], out["r_mean"], out["model"]
    title = f"{os.path.basename(out['run_dir'])} ({name}, r_mean={r_mean}, inject={out['inject']})"

    # (a) depth sweep
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(sw["r"], sw["acc"], "o-", ms=3)
    ax[0].axvline(r_mean, color="gray", ls="--", label=f"training r_mean={r_mean}")
    ax[0].set(xlabel="recurrence r", ylabel="test accuracy (%)", title="accuracy vs test-time depth")
    ax[0].legend()
    ax2 = ax[0].twinx()
    ax2.plot(sw["r"], sw["nll"], color="C3", alpha=0.5)
    ax2.set_ylabel("test NLL", color="C3")
    ax[1].plot(sw["gflops"], sw["acc"], "o-", ms=3)
    ax[1].set(xlabel="GFLOPs / image", ylabel="test accuracy (%)", title="accuracy vs compute")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, f"recurrent_depth_sweep{tag}.png"), dpi=110)
    plt.close(fig)

    # (b) Pareto
    ae = out["adaptive_exit"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (xk, xl) in enumerate([("avg_iters", "average iterations"), ("gflops", "GFLOPs / image")]):
        fx = sw["r"] if xk == "avg_iters" else sw["gflops"]
        ax[i].plot(fx, sw["acc"], "o-", ms=3, label="fixed r")
        for m, lab in (("kl", "exit: KL(p_t||p_t-1) < tau"), ("maxprob", "exit: |dmax p| < tau")):
            ax[i].plot([e[xk] for e in ae[m]], [e["acc"] for e in ae[m]], ".-", label=lab)
        g = next(e for e in ae["kl"] if e["tau"] == GEIPING_TAU)
        ax[i].plot(g[xk], g["acc"], "k*", ms=10, label="KL tau=5e-4 (Geiping)")
        if xk == "avg_iters":
            ax[i].plot(ae["oracle"]["avg_iters"], ae["oracle"]["acc"], "r^", label="oracle exit")
        ax[i].set(xlabel=xl, ylabel="test accuracy (%)", xscale="log")
        ax[i].legend(fontsize=7)
    ax[0].set_title("adaptive exit vs fixed depth")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, f"recurrent_exit_pareto{tag}.png"), dpi=110)
    plt.close(fig)

    # (c) difficulty
    diff, c10c, c10h = out["difficulty"], out["cifar10c"], out["cifar10h"]
    n = 1 + (ref_loss is not None) + (c10c is not None) + (c10h is not None)
    fig, ax = plt.subplots(1, n, figsize=(4.5 * n, 3.8), squeeze=False)
    ax = ax[0]
    ax[0].bar(range(1, len(diff["exit_hist"]) + 1), diff["exit_hist"])
    ax[0].set(xlabel="exit step", ylabel="# images", title=f"exit steps (KL tau={diff['tau']:.1e})")
    j = 1
    if ref_loss is not None:
        st, rl = st_c.numpy(), ref_loss.numpy()
        ts = sorted(set(st.tolist()))
        ax[j].plot(ts, [rl[st == t].mean() for t in ts], "o-")
        ax[j].set(xlabel="exit step", ylabel=f"mean {diff['ref_model']} loss",
                  title=f"Spearman rho = {diff['spearman_exit_vs_ref_loss']:+.3f}")
        j += 1
    if c10c is not None:
        for cname, rec in c10c["corruptions"].items():
            ax[j].plot(range(6), rec["mean_exit"], color="gray", alpha=0.3)
        ax[j].plot(range(6), c10c["mean_exit_by_severity"], "ko-", label="mean over corruptions")
        ax[j].set(xlabel="CIFAR-10-C severity (0 = clean)", ylabel="mean exit step",
                  title=f"pooled Spearman rho = {c10c['pooled_sev_spearman']:+.3f}")
        ax[j].legend(fontsize=7)
        j += 1
    if c10h is not None:
        dec = range(1, len(c10h["decile_mean_exit"]) + 1)
        ax[j].plot(dec, c10h["decile_mean_exit"], "o-")
        ax[j].set(xlabel="CIFAR-10H human-entropy decile (1 = most agreed)", ylabel="mean exit step",
                  title=f"rho(exit, H) = {c10h['spearman_exit_vs_entropy']:+.3f}, "
                        f"rho(exit, 1-p) = {c10h['spearman_exit_vs_1_minus_p_true']:+.3f}")
        ax[j].title.set_fontsize(8)
        a2 = ax[j].twinx()
        a2.plot(dec, c10h["decile_mean_entropy"], color="C3", alpha=0.5)
        a2.set_ylabel("mean human entropy (nats)", color="C3")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, f"recurrent_difficulty{tag}.png"), dpi=110)
    plt.close(fig)

    # (d) convergence
    cv = out["state_convergence"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(cv["t"], cv["mean"], label="mean")
    ax.plot(cv["t"], cv["median"], label="median")
    ax.fill_between(cv["t"], cv["q10"], cv["q90"], alpha=0.2, label="10-90%")
    ax.axvline(r_mean, color="gray", ls="--")
    ax.set(xlabel="iteration t", ylabel="||s_t - s_{t-1}|| / ||s_t||", yscale="log", title="state convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, f"recurrent_state_convergence{tag}.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
