"""S9 — mechanism diagnostic for Section 4.2: does the denoiser read the perturbation as noise?
For a perturbed image x' = x + d (d in the [-1,1] input scale), measure with the ADM probe at t=1:
  cos_dn   = cos(eps_hat(x'), d / sqrt(1 - abar_1))      alignment of the noise estimate with the perturbation
  cos_base = cos(eps_hat(x'), eps_hat(x))                how much of the clean noise estimate survives
  frac     = ||P_d eps_hat(x')||^2 / ||eps_hat(x')||^2   share of the output energy along d
  norm_ratio = ||eps_hat(x')|| / ||eps_hat(x)||
sets: first 300 eligible images of resnet50 PGD_e4 (successful or not) and the same images under FGSM, TIFGSM,
      Square (as available), CTRL_sign/gauss/phase/shuffle/jpeg75/blur.
out: OUTD/scores/align/<kind>.npz
usage: python s9_alignment.py [--n 300]"""
import os, argparse, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=300); ap.add_argument('--bs', type=int, default=16); A = ap.parse_args()
X, _ = load_cache(); unet = load_unet(DEVICE, torch.float16)
KINDS = ['PGD_e4', 'FGSM_e4', 'TIFGSM_e4', 'Square_e4', 'CW_e4'] + [f'CTRL_{k}_e4' for k in ['sign', 'gauss', 'phase', 'shuffle', 'jpeg75', 'blur']]
os.makedirs(f'{OUTD}/scores/align', exist_ok=True)
for kind in KINDS:
    out = f'{OUTD}/scores/align/{kind}.npz'
    if os.path.exists(out): continue
    d = np.load(f'{OUTD}/adv/resnet50/{kind}.npz'); ids = d['image_id'][:A.n]; adv = d['adv'][:A.n]
    acc = {k: [] for k in ['cos_dn', 'cos_base', 'frac', 'norm_ratio']}
    for b in range(0, len(ids), A.bs):
        x = to_tensor(X[ids[b:b + A.bs]], DEVICE) * 2 - 1; xa = to_tensor(adv[b:b + A.bs], DEVICE) * 2 - 1
        dd = (xa - x).reshape(len(x), -1).float()
        u = eps_hat(unet, x.half()); ua = eps_hat(unet, xa.half())
        acc['cos_dn'].append(F.cosine_similarity(ua, dd, dim=1).cpu().numpy())
        acc['cos_base'].append(F.cosine_similarity(ua, u, dim=1).cpu().numpy())
        proj = (ua * dd).sum(1) ** 2 / (dd * dd).sum(1).clamp_min(1e-12)
        acc['frac'].append((proj / (ua * ua).sum(1)).cpu().numpy())
        acc['norm_ratio'].append((ua.norm(dim=1) / u.norm(dim=1)).cpu().numpy())
    R = {k: np.concatenate(v) for k, v in acc.items()}
    atomic_savez(out, image_id=ids, success=d['success'][:A.n], **R)
    log(kind, ' '.join(f'{k}={np.median(v):.3f}' for k, v in R.items()))
