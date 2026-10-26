"""S8 — probe transfer (Table 9) and Monte Carlo count K (AE #7 cost/accuracy) on the R1 splits.
sets (same as s3 tgrid): 1000 cal + 1000 test clean, first 500 of resnet50 / vit_b_16 PGD e4, first 500 resnet50 CTRL_gauss
  probe   ADM (main scores reused) vs google/ddpm-celebahq-256, google/ddpm-church-256 at t=1, sigma, K=8
          (both are 3-channel epsilon-prediction UNets with the same linear beta schedule as ADM; x = 2*img - 1)
  k       ADM at K in {1, 2, 4, 16}; probe noise is drawn per K (seed = f(image_id, 'main')), so each K is compared
          with the K=8 main scores of the same images
out: OUTD/scores/probe/<probe>/<set>.npz, OUTD/scores/kgrid/<set>_k<K>.npz
usage: python s8_probe_k.py probe k [--bs 16]"""
import os, argparse, numpy as np, torch
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('what', nargs='+'); ap.add_argument('--bs', type=int, default=16); A = ap.parse_args()
X, Y = load_cache()
n = 1000 if not SMOKE else 8; m = 500 if not SMOKE else 8
SETS = {'cal': (split_ids('cal')[:n], None), 'test': (split_ids('test')[:n], None)}
for tag in ['resnet50/PGD_e4', 'vit_b_16/PGD_e4', 'resnet50/CTRL_gauss_e4']:
    d = np.load(f'{OUTD}/adv/{tag}.npz'); SETS[tag.replace('/', '_')] = (d['image_id'][:m], (d['adv'][:m], d['success'][:m]))

def score(unet, out, ids, adv, k):
    if os.path.exists(out): log('exists', out); return
    os.makedirs(os.path.dirname(out), exist_ok=True); acc = {}
    for b in range(0, len(ids), A.bs):
        j = np.arange(b, min(b + A.bs, len(ids)))
        imgs = to_tensor(X[ids[j]], DEVICE) if adv is None else to_tensor(adv[0][j], DEVICE)
        for kk, v in cpop_all(unet, imgs, ids[j], k=k).items(): acc.setdefault(kk, []).append(v)
    kw = {} if adv is None else {'success': adv[1]}
    atomic_savez(out, image_id=ids, **kw, **{kk: np.concatenate(v) for kk, v in acc.items()}); log('saved', out)

for w in A.what:
    if w == 'probe':
        from diffusers import UNet2DModel
        for name, repo in [('celebahq', 'google/ddpm-celebahq-256'), ('church', 'google/ddpm-church-256')]:
            u = UNet2DModel.from_pretrained(repo, torch_dtype=torch.float16).to(DEVICE).eval()
            for p in u.parameters(): p.requires_grad_(False)
            for nm, (ids, adv) in SETS.items(): score(u, f'{OUTD}/scores/probe/{name}/{nm}.npz', ids, adv, K)
            del u; torch.cuda.empty_cache()
    elif w == 'k':
        u = load_unet(DEVICE, torch.float16)
        for k in [1, 2, 4, 16]:
            for nm, (ids, adv) in SETS.items(): score(u, f'{OUTD}/scores/kgrid/{nm}_k{k}.npz', ids, adv, k)
