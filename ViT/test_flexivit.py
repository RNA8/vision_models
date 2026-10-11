"""Sanity tests for the FlexiViT patch switch in finetune.py (python test_flexivit.py)."""
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timm.layers.patch_embed import _compute_resize_matrix
from finetune import load_model, HERE

MAIN = os.environ.get('VIT_MAIN', HERE)
CK = os.path.join(MAIN, 'runs', 'in64pre_tiny_p8')
m8, _ = load_model(CK, 100, 64, transfer=True, drop_path=0.0)
m4, spec4 = load_model(CK, 100, 64, transfer=True, drop_path=0.0, patch_size=4)
m4s, _ = load_model(CK, 100, 32, transfer=True, drop_path=0.0, patch_size=4)
m8.eval(); m4.eval(); m4s.eval()
assert spec4["train_args"]["patch_size"] == 4
t = lambda m, s: m.patch_embed.grid_size
assert m8.patch_embed.grid_size == (8, 8) and m4.patch_embed.grid_size == (16, 16) and m4s.patch_embed.grid_size == (8, 8)
x = torch.randn(2, 3, 64, 64)
assert m8.forward_features(x).shape[1] - m8.num_prefix_tokens == 64 and m4.forward_features(x).shape[1] - m4.num_prefix_tokens == 256
print("token counts ok: p8@64 = 64, p4@64 = 256, p4@32 = 64")

# 1) exact PI identity: for a 4x4 patch xi, w8 . (pinv(B) xi) == w4 . xi (B = bilinear 8x8 -> 4x4 resize)
w8, w4 = m8.patch_embed.proj.weight.detach(), m4.patch_embed.proj.weight.detach()
B = _compute_resize_matrix((8, 8), (4, 4), "bilinear", True, torch.device("cpu"), torch.float64)
xi = torch.randn(16, dtype=torch.float64)
up = (torch.linalg.pinv(B) @ xi).reshape(8, 8)
lhs = (w8.double() * up).sum((1, 2, 3)) if False else torch.einsum("ochw,hw->oc", w8.double(), up).sum(1)
rhs = torch.einsum("ochw,hw->oc", w4.double(), xi.reshape(4, 4)).sum(1)
print("PI identity max err:", (lhs - rhs).abs().max().item(), "scale", rhs.abs().mean().item())
assert torch.allclose(lhs, rhs, atol=1e-4 * rhs.abs().max().item())

# 2) model level on a *fine-tuned* patch-8 CIFAR-100 net (no further training): p8 on 64px bilinear-upsampled
#    images vs the PI-resized p4 net on the 32px originals and on the same 64px images.
import numpy as np
z = np.load(os.path.join(MAIN, "data", "cifar100_uint8.npz"))
xk = [k for k in z if "test" in k and z[k].ndim == 4][0]; yk = [k for k in z if "test" in k and z[k].ndim == 1][0]
xs = torch.from_numpy(z[xk]).float() / 255
xs = xs if xs.shape[1] == 3 else xs.permute(0, 3, 1, 2)
ys = torch.from_numpy(z[yk]).long()
FT = os.path.join(MAIN, "runs", "ec100_in64pre_tiny_64")
f8, sp = load_model(FT, transfer=False)
f4, _ = load_model(FT, 100, 64, transfer=True, patch_size=4)
f4s, _ = load_model(FT, 100, 32, transfer=True, patch_size=4)
# transfer=True re-inits the head; copy the trained head back
for m in (f4, f4s):
    m.head.load_state_dict(f8.head.state_dict())
mean = torch.tensor(sp["mean"]).view(1, 3, 1, 1); std = torch.tensor(sp["std"]).view(1, 3, 1, 1)
x32 = (xs - mean) / std
x64 = F.interpolate(x32, size=64, mode="bilinear", align_corners=False)
def acc(m, x):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), 500):
            out.append(m(x[i:i + 500]))
    o = torch.cat(out); return (o.argmax(1) == ys).float().mean().item() * 100, o
a8, o8 = acc(f8, x64); a4s, o4s = acc(f4s, x32); a4, o4 = acc(f4, x64)
print(f"fine-tuned p8 @64: {a8:.2f}   PI p4 @32 (no training): {a4s:.2f}   PI p4 @64: {a4:.2f}")
print(f"argmax agreement p8@64 vs p4@32: {(o8.argmax(1) == o4s.argmax(1)).float().mean().item() * 100:.1f}%")
