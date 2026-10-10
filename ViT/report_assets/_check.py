import os; os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch, torch.nn.functional as F
import train_vit as T
from vit_data import cifar10
xtr,ytr,xte,yte = cifar10.load('data')
f = lambda u: u.permute(0,3,1,2).float()/255
def save(rows, path, ncol):
    import torchvision.utils as vu
    g = vu.make_grid(torch.cat(rows), nrow=ncol, padding=1, pad_value=1.0)
    g = F.interpolate(g[None], scale_factor=4, mode='nearest')[0]
    vu.save_image(g, path)
x = f(xte[:16]); torch.manual_seed(0)
rows=[x]+[T.rand_augment(T.random_crop_flip(x,4),2,0.9) for _ in range(4)]
save(rows,'report_assets/randaug_check.png',16)
orig=torch.randint
def force(k):
    def r(low,high=None,size=None,**kw):
        if high==len(T.RA_OPS): return torch.full(size,k,dtype=torch.long)
        return orig(low,high,size,**kw)
    return r
x8=f(xte[:8]); rows=[x8]
big=f(xte[:1024]); print("n ops",len(T.RA_OPS))
for k,name in enumerate(T.RA_OPS):
    T.torch.randint=force(k)
    torch.manual_seed(1)
    rows.append(T.rand_augment(x8,1,0.9))
    o=T.rand_augment(big,1,0.9)
    print(f"{name:13s} mean {o.mean():.3f} std {o.std():.3f} per-img-std {o.std((1,2,3)).mean():.3f} frac0.5 {(o==0.5).float().mean():.4f} min {o.min():.2f} max {o.max():.2f}")
T.torch.randint=orig
save(rows,'report_assets/randaug_per_op.png',8)
torch.manual_seed(2)
o=T.rand_augment(T.random_crop_flip(big,4),2,0.9)
print("finite",torch.isfinite(o).all().item(),"min",o.min().item(),"max",o.max().item())
print("in mean/std",big.mean().item(),big.std().item(),"out",o.mean().item(),o.std().item())
print("frac==0.5 in",(big==0.5).float().mean().item(),"out",(o==0.5).float().mean().item())
