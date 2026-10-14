# Multiscale input tokens for a small ViT ("pyramid tokens")

**Idea.** Patchify the image into PxP patches, downscale it 2x and patchify again with the same P, and repeat until
one patch is left. All tokens from all scales are concatenated into **one flat transformer sequence**. For 32 px CIFAR
and P=4 that is 64 + 16 + 4 + 1 = 85 tokens instead of 64, about 1.33x the tokens and 1.32x the MACs.

Status: implemented (`--multiscale`, commit cce93bf). CIFAR-100 300-epoch runs are queued in `run_wave9_multiscale.sh`;
results are at the bottom.

## A. Prior work (sources fetched 2026-10-13)

### Verdict
The core idea is **already published as RetinaViT** (Shu & Bain, arXiv 2403.13677: v1 March 2024, v2 September 2025).
- It builds an image pyramid, patchifies every level with the same 16 px patch and the **same** projection, goes down
  to a single whole-image token, and flattens everything into one ViT sequence.
- Our 64+16+4+1 layout is the 32 px version of the same design.

Things we did **not** find published:
- the Laplacian/residual-level variant;
- a shared vs per-scale projection ablation;
- combination with small-data ViT tricks (SPT/LSA);
- **any CIFAR or Tiny-ImageNet result for pyramid tokens**.

### Closest prior work, ranked

**1. RetinaViT** — https://arxiv.org/abs/2403.13677 (html v2: https://arxiv.org/html/2403.13677v2)
- *"The original image with a resolution of (224×224) is scaled down into four lower resolutions (128×128), (64×64),
  (32×32), and (16×16)"*
- *"then the whole hierarchy of five is flattened into one sequence of patches"*
- *"patch embeddings are extracted with the same convolution kernel therefrom"*
- Position embedding of a scaled patch: it *"calculates the weighted average embedding vector of all pixels covered by
  the particular patch in the original image"* and *"scales its norm to the square root of the relative edge length"*.
  Our `--ms-pos learned` initializes the coarse position embeddings the same way (avg-pooled fine pos-embed, no norm
  rescaling). `--ms-pos pooled` keeps them tied to the fine pos-embed and adds a learned per-scale vector.
- Results: ImageNet-1K, 90 epochs, one run per configuration (Table 1, v2).

  | layers | ViT | RetinaViT | diff |
  |---:|---:|---:|---:|
  | 12 | 76.5 | 76.7 | +0.2 |
  | 4 | 64.5 | 65.1 | +0.6 |
  | 2 | 46.4 | 47.2 | +0.8 |

  The authors say it *"did not provide a huge improvement in absolute prediction accuracy at the full configuration"*,
  but *"did improve the robustness of the model when the number of layers is reduced."*
- How it differs from ours: ImageNet only, no small-data results, no token-count control, no Laplacian variant.

**2. MSViT, dynamic mixed-scale tokenization** (Havtorn et al. 2023) — https://arxiv.org/abs/2307.02321
- Two scales: *"we extract square patches at both scales"*. A gate *"outputs a binary decision"* per region, so each
  region appears at **one** scale (no redundancy).
- Coarse position encodings are *"obtained by linearly interpolating the fine scale position encodings"*. This is
  analogous to our `--ms-pos pooled`.

**3. Charm** (image aesthetics) — https://arxiv.org/abs/2504.02522
- Uses two scales with the same patch size in one sequence: *"we arrange all high-resolution patches at the beginning,
  followed by low-resolution ones"*.
- Scale embeddings: *"These scale embeddings are randomly initialized and added element-wise to all patch embeddings."*
- Each region appears at one scale only. Three scales were not better than two.

**4. Quadformer / mixed-resolution tokenization** (Ronen et al. 2023) — https://arxiv.org/abs/2304.00287
- *"replacing the standard uniform grid with a mixed-resolution sequence of tokens"*, built with a quadtree.
- Non-overlapping, so there are no redundant scales.

**5. VAR** — https://arxiv.org/html/2404.02905
- Multi-scale token maps go into one standard transformer with a per-scale *"position embedding map"*.
- Same "all scales in one sequence" pattern, but for autoregressive generation with block-causal masking, not
  classification.

**6. CrossViT** — https://arxiv.org/abs/2103.14899
- *"process small-patch and large-patch tokens with two separate branches"*, fused through CLS tokens.
- Two branches, not one joint sequence.

**7. MPViT** — https://arxiv.org/abs/2112.11010
- *"Tokens of different scales are then independently fed into the Transformer encoders via multiple paths."*
- Parallel paths, not concatenation.

**8. Coarse-to-fine cascades**
- CF-ViT (https://arxiv.org/abs/2203.03821) and DVT (https://arxiv.org/abs/2105.15075) process scales one after
  another, not jointly.

**9. Multi-resolution training (not multi-scale tokens)**
- ResFormer (https://arxiv.org/abs/2212.00776) and FlexiViT (https://arxiv.org/abs/2212.08013) train over several
  resolutions or patch sizes, but each forward pass sees a single scale.

**10. Registers** (Darcet et al.) — https://arxiv.org/abs/2309.16588
- Extra learnable tokens change ViT behaviour on their own.
- We use 21 registers as the **token-matched control** (85 tokens, the same attention cost, no multiscale input).

**Not fetched:** MViT/MViTv2 (pooling attention), PVT, Twins and Swin (hierarchical backbones). These are
multi-scale *inside* the network, not at the input, so they are a different idea.

### What the literature predicts
- At full depth on ImageNet the gain was small (+0.2). The benefit grew for shallow models (+0.8 at 2 layers), from
  single runs.
- Our model is 9 layers × 192 wide, so expect anywhere from no gain to a few tenths of a point. That is inside our
  seed noise of about 0.3 to 0.6 pt unless the small-data regime changes the picture.
- Small data is untested in the literature. Coarse tokens are a strong low-frequency, global-context prior, which is
  exactly what ViTs lack on CIFAR, so a gain there is plausible.

## B. Implementation (`ViT/train_vit.py`)

### Flags
| flag | values | default |
|---|---|---|
| `--multiscale` | on/off | off |
| `--ms-proj` | `shared`, `per-scale` | `shared` |
| `--ms-pos` | `learned`, `pooled` | `learned` |
| `--ms-down` | `avg`, `bilinear` (antialiased) | `avg` |
| `--ms-pyramid` | `gauss`, `laplacian` | `gauss` |
| `--reg-tokens N` | control | 0 |

### Main pieces
- **`MultiscaleEmbed`** replaces timm's `patch_embed`.
  - Pyramid: x_{s+1} = down(x_s) on the normalized image.
  - Each level goes through the patch projection: `proj0` is timm's Conv2d or our SPTProj, either shared or one per
    scale.
  - Tokens are concatenated fine-to-coarse.
  - Laplacian mode: every level except the coarsest is x_s − up(x_{s+1}). With `avg`, `up` is nearest, so the residual
    is exactly zero-mean per 2x2 block.
- **`MultiscaleViT`** is a subclass of the timm ViT (set via `__class__`, like `apply_lsa`).
  - It overrides `_pos_embed`: CLS + tokens + [timm `pos_embed` (CLS + fine grid) | coarse position embeddings].
  - Coarse position embeddings are excluded from weight decay, like `pos_embed`.

### Composition with the rest of the recipe
- **SPT:** applied per scale. Every level is shifted by half a patch (2 px) at its own resolution. Shared mode uses
  one SPTProj for all levels.
- **LSA:** unchanged. The diagonal mask is built for N = 86.
- **seqpool:** pools all 85 tokens (CLS excluded). With `--reg-tokens`, registers are excluded from pooling, as in
  Darcet et al.
- **impulse init:**
  - Computed from the fine 8x8 grid only, before the multiscale conversion.
  - Q/K are weights, not per-token, so they apply unchanged.
  - Coarse position embeddings are initialized as avg-pooled copies of the (×18-scaled) fine pos-embed. A fine
    query's local impulse kernel therefore also matches the coarse token covering its neighbourhood.

### Safety checks
- **Default path bit-identical:** with the flag off, `state_dict` and outputs are `torch.equal` to the old code for
  seed 0.
- **Smoke tests:** 2 epochs ran on GPU with cudagraphs for the default path, multiscale per-scale + Laplacian, and 21
  registers.
- **`finetune.py`:** `build_from_spec` + `adapt_state_dict` still work on an old run's args, since the new args are
  read with `getattr` defaults.

### Cost
CIFAR-100 model; MACs from `torch.utils.flop_counter`, per 32 px image.

| model | params | tokens (+CLS) | MACs |
|---|---:|---:|---:|
| baseline (SL-ViT + SPT/LSA/seqpool) | 2.753M | 64 | 175.5M |
| multiscale, shared proj, learned pos | 2.757M | 85 | 232.2M |
| multiscale, per-scale proj | 2.897M | 85 | 232.2M |
| multiscale, pooled pos + scale embed | 2.753M | 85 | 232.2M |
| baseline + 21 registers (control) | 2.761M | 85 | 231.2M |

## C. Results (CIFAR-100, 300 ep, bf3b recipe + SPT)

All runs use the exact args of `c100_spt_slvit_300ep`: SL-ViT 9×192, SPT + LSA + seqpool + impulse,
`--mix-off-epoch 175`. The number reported is final top-1 at epoch 300.

### Runs
- Queues: `run_wave9_multiscale.sh` (first seeds and variants), `run_wave9b_seeds.sh` (seed 1) and
  `run_wave9c_seeds.sh` (seed 2). Seed 2 was added because the seed-1 reruns showed about 1 pt of seed spread.
- Status as of 2026-10-14 ~10:45 UTC: 7 runs finished.
  - `c100_ms_shared_300ep_s2` has since finished (76.53).
  - The remaining runs finished on 2026-10-14 (filled in by the main session): pooled pos 76.18, baseline seed 1 77.57, register control seed 2 77.43. All 11 runs are done.

| variant | tokens (+CLS) | params | MACs | s0 | s1 | s2 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline `c100_spt_slvit_300ep` | 64 | 2.753M | 175.5M | 76.70 | 77.57 | 77.82 | 77.36 (n=3) |
| multiscale, shared proj (`--multiscale`) | 85 | 2.757M | 232.2M | 76.42 | 77.48 | 76.53 | 76.81 (n=3) |
| multiscale, per-scale proj | 85 | 2.897M | 232.2M | 75.17 | | | 75.17 (n=1) |
| multiscale, Laplacian levels | 85 | 2.757M | 232.2M | 73.95 | | | 73.95 (n=1) |
| multiscale, pooled pos + scale emb. | 85 | 2.753M | 232.2M | 76.18 | | | 76.18 (n=1) |
| control: baseline + 21 registers | 85 | 2.761M | 231.2M | 77.19 | 77.07 | 77.43 | 77.23 (n=3) |

### Throughput
- The img/s figures inside results.json depend on how many other jobs shared the GPU, so they are not comparable
  between runs.
- An interleaved micro-benchmark measured all variants back to back in one process under the same contention
  (batch 128, compiled with CUDA graphs, fwd+bwd+AdamW).

| variant | train img/s | relative to baseline |
|---|---:|---:|
| baseline | 4124 | 1.00 |
| multiscale shared | 4043 | 0.98 |
| multiscale per-scale | 4045 | 0.98 |
| multiscale Laplacian | 4056 | 0.98 |
| 21 registers | 4164 | 1.01 |

- At this size the H100 is bound by kernel launches and contention, so the extra 32% of MACs barely shows.
- On an edge device that is bound by compute, expect roughly 1.3x the latency for both the 85-token multiscale
  model and the 85-token register control.

### Verdict
**No gain from multiscale input tokens on CIFAR-100 with our recipe.**

1. **Shared projection (the RetinaViT design)** is at parity with the baseline:
   - multiscale mean 76.81 over three seeds vs baseline mean 77.26 over two seeds;
   - the gap is −0.45 pt, inside a seed spread of about 1 pt (baseline 76.70 to 77.82, multiscale 76.42 to 77.48).
   - It is 32% more MACs for nothing measurable. This matches RetinaViT's ImageNet result at full depth (+0.2).
2. **Per-scale projection (−1.5 to −2 pt) and Laplacian levels (−2.7 to −3.3 pt) are clearly worse.** These are
   single seeds, but the gaps are larger than the seed spread.
   - Laplacian levels remove the low-pass content from the fine tokens. The SPT/impulse-initialized fine pathway
     seems to rely on it.
   - Per-scale projections add 0.14M parameters and split the data across four tokenizers.
3. **The token-matched control (21 registers) is also at parity** (77.13 vs 77.26). Neither "more tokens" nor
   "multiscale tokens" helps here.
4. **Seed noise at 300 epochs is about ±0.5 pt (a 1 pt range), larger than the 0.26 std at 1000 epochs.** Any claim of
   a sub-1-pt difference at 300 epochs needs at least 3 seeds.

**Recommendation:** do not adopt `--multiscale` for the CIFAR edge model. The flag stays in `train_vit.py`
(default off, bit-identical when off). The one regime where the literature suggests a benefit is very shallow
models: RetinaViT reports +0.8 at 2 layers and +0.6 at 4. That would be a cheap follow-up (e.g. `--depth 3`) if we
want a smaller edge model.
