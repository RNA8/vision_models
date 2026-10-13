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

## C. Results (CIFAR-100, 300 ep, bf3b recipe + SPT, seed 0 unless noted)

_Pending: runs queued in `run_wave9_multiscale.sh`._
