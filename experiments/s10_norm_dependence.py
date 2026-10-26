"""S10 — Figure 1 under the revised protocol: CPOP versus the L2 norm of PGD-L2 perturbations (ResNet-50).
The same first N eligible images of P_eval (ResNet-50 labels them correctly) are attacked at every budget with PGD-L2
(alpha = eps/10, 20 steps, ground-truth labels) and scored with the main ADM probe (t=1, sigma, K=8; probe noise shared
with the clean source). Inputs are NOT quantized here: at the smallest budgets the per-pixel change is below one 8-bit
level. Every perturbed input is kept, successful or not (success is recorded).
out: OUTD/norm_dependence.csv  (eps, image_id, l2_norm, cpop, success; eps = 0 rows are the clean sources)
usage: python s10_norm_dependence.py [--n 50]"""
import argparse, numpy as np, pandas as pd, torch, torchattacks
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=50); ap.add_argument('--bs', type=int, default=25)
A = ap.parse_args()
EPS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
dev = DEVICE; u8, y = load_cache(); clf = build_clf('resnet50', dev); unet = load_unet(dev)
ids = split_ids('eval')
elig = []
for _, b in parts(ids, 256):
    with torch.no_grad(): p = clf(to_tensor(u8[b], dev)).argmax(1).cpu().numpy()
    elig += list(b[p == y[b]])
    if len(elig) >= A.n: break
elig = np.array(elig[:A.n], dtype=np.int64); log(f'{len(elig)} eligible images')
rows = []
for _, b in parts(elig, A.bs):
    x = to_tensor(u8[b], dev).contiguous(); yt = torch.from_numpy(y[b]).to(dev)
    c = cpop_all(unet, x, b)['cpop']
    rows += [dict(eps=0.0, image_id=int(i), l2_norm=0.0, cpop=float(v), success=False) for i, v in zip(b, c)]
    for e in EPS:
        xa = torchattacks.PGDL2(clf, eps=e, alpha=e / 10, steps=20)(x, yt).detach()
        with torch.no_grad(): ok = (clf(xa).argmax(1) != yt).cpu().numpy()
        l2 = (xa - x).flatten(1).norm(dim=1).cpu().numpy(); ca = cpop_all(unet, xa, b)['cpop']
        rows += [dict(eps=e, image_id=int(i), l2_norm=float(l), cpop=float(v), success=bool(s)) for i, l, v, s in zip(b, l2, ca, ok)]
        log(f'eps={e} mean l2={l2.mean():.3f} mean cpop={ca.mean():.4f} succ={ok.mean():.2f}')
df = pd.DataFrame(rows); df.to_csv(f'{OUTD}/norm_dependence.csv', index=False)
a = df[df.eps > 0]; rb = a[a.l2_norm > 1.0]
sl = np.polyfit(np.log10(rb.l2_norm), np.log10(rb.cpop), 1)[0]
log(f'clean mean cpop {df[df.eps == 0].cpop.mean():.4f}; plateau (l2<0.5) {a[a.l2_norm < 0.5].cpop.mean():.4f}; '
    f'log-log slope (l2>1, n={len(rb)}) {sl:.3f}; success by eps {a.groupby("eps").success.mean().round(2).to_dict()}')
