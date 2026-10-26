"""S3 — ADM probe scores (cpop, raw_var, score_norm, min_pnorm) for clean splits and adversarial/control sets.
Probe noise is a deterministic function of (image_id, variant) -> an adversarial image and its clean progenitor
share probe directions (common random numbers); 'seeds' re-scores test images under other variants.

sources:  clean          cal+test+eval clean, variant 'main'          -> scores/clean.npz
          seeds          first 1000 test ids x variants s1..s4         -> scores/seeds.npz (key variant)
          sigma          eval clean + resnet50 PGD e4, sigma grid       -> scores/sigma.npz (AE #8 local-regime check)
          tgrid          integer timesteps 0/2/4 (t=1 = main): 1000 cal + 1000 test clean, 500 R50 + 500 ViT PGD e4
                                                                        -> scores/tgrid/<set>_t<t>.npz (t=0 is a trained level)
          ALL_ADV        every OUTD/adv/<clf>/<kind>.npz without a score file -> scores/adv/<clf>/<kind>.npz
          adv:<clf>/<kind>_e<eps>
usage: python s3_score.py clean seeds ALL_ADV [--bs 16]"""
import os, glob, argparse, numpy as np, torch
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('sources', nargs='+'); ap.add_argument('--bs', type=int, default=16); A = ap.parse_args()
dt = torch.float16 if DEVICE == 'cuda' else torch.float32
unet = load_unet(DEVICE, dt); X, Y = load_cache()

def run(tag, ids, get_imgs, variant='main', sigma=SIGMA, extra=None, t_step=T_STEP):
    out = f'{OUTD}/scores/{tag}.npz'
    if os.path.exists(out): log('exists', out); return
    pdir = out[:-4] + '_parts'; os.makedirs(pdir, exist_ok=True)
    for pi, pid in parts(np.arange(len(ids)), 512):
        pf = f'{pdir}/part_{pi:04d}.npz'
        if os.path.exists(pf): continue
        acc = {}
        for b in range(0, len(pid), A.bs):
            j = pid[b:b + A.bs]; r = cpop_all(unet, get_imgs(j), ids[j], variant=variant, sigma=sigma, t_step=t_step)
            for k, v in r.items(): acc.setdefault(k, []).append(v)
        acc = {k: np.concatenate(v) for k, v in acc.items()}
        kw = {k: np.asarray(v)[pid] for k, v in (extra or {}).items()}
        atomic_savez(pf, image_id=ids[pid], **kw, **acc)
        log(f'  {tag} part {pi}: {min(pid[-1] + 1, len(ids))}/{len(ids)}')
    merge_parts(pdir, out); log('saved', out)

def clean_src(ids): return lambda j: to_tensor(X[ids[j]], DEVICE)

for src in A.sources:
    if src == 'clean':
        ids = np.sort(np.concatenate([split_ids(s) for s in ['cal', 'test', 'eval']])); run('clean', ids, clean_src(ids))
    elif src == 'seeds':
        base = split_ids('test')[:1000 if not SMOKE else 16]
        for v in ['s1', 's2', 's3', 's4']: run(f'seeds_{v}', base, clean_src(base), variant=v)
    elif src == 'sigma':
        ref = np.load(f'{OUTD}/adv/resnet50/PGD_e4.npz'); n = 300 if not SMOKE else 8
        ids = ref['image_id'][:n]; adv = ref['adv'][:n]
        for sg in [0.00125, 0.0025, 0.005, 0.01, 0.02]:
            run(f'sigma/clean_s{sg}', ids, clean_src(ids), sigma=sg)
            run(f'sigma/adv_s{sg}', ids, lambda j: to_tensor(adv[j], DEVICE), sigma=sg)
    elif src == 'tgrid':
        n = 1000 if not SMOKE else 8; m = 500 if not SMOKE else 8
        sets = {'cal': (split_ids('cal')[:n], None), 'test': (split_ids('test')[:n], None)}
        for c in ['resnet50', 'vit_b_16']:
            d = np.load(f'{OUTD}/adv/{c}/PGD_e4.npz'); sets[f'{c}_PGD_e4'] = (d['image_id'][:m], (d['adv'][:m], d['success'][:m]))
        for t in [0, 2, 4]:
            for nm, (ids, adv) in sets.items():
                if adv is None: run(f'tgrid/{nm}_t{t}', ids, clean_src(ids), t_step=t)
                else: run(f'tgrid/{nm}_t{t}', ids, lambda j, a=adv[0]: to_tensor(a[j], DEVICE), extra={'success': adv[1]}, t_step=t)
    else:
        files = sorted(glob.glob(f'{OUTD}/adv/*/*.npz')) if src == 'ALL_ADV' else [f'{OUTD}/adv/{src[4:]}.npz']
        files = [f for f in files if not f.endswith('.tmp.npz')]
        for f in files:
            tag = 'adv/' + os.path.relpath(f, f'{OUTD}/adv')[:-4]
            if os.path.exists(f'{OUTD}/scores/{tag}.npz'): continue
            d = np.load(f); ids = d['image_id']; adv = d['adv']
            run(tag, ids, lambda j, adv=adv: to_tensor(adv[j], DEVICE), extra={'success': d['success']})
