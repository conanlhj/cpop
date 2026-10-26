"""S5 — feature-space baselines on penultimate features, fitted on the ref split only (20/class, GT labels).
  maha1   single-Gaussian Mahalanobis (as submitted; shrinkage 1e-2)
  maha_cc class-conditional means + tied covariance, score = min_c distance (Lee et al. 2018, no input
          preprocessing / layer ensemble)
  kd1     global Gaussian KDE, median-heuristic bandwidth (as submitted)
  kde_cc  KDE over ref features of the PREDICTED class (Feinman et al. 2017), bandwidth = median within-class distance
  lid     MLE LID (Ma et al. 2018 estimator), k=20 neighbours in ref (scalar; not the supervised multi-layer detector)
All scores oriented higher = more suspicious.
out: OUTD/feat_scores/<clf>/clean.npz and /<kind>_e<eps>.npz
usage: python s5_feature_baselines.py --clf resnet50   |   --dataset cifar --clf resnet20"""
import os, glob, argparse, numpy as np, torch
from common import *

import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--clf', required=True); ap.add_argument('--dataset', default='imagenet'); A = ap.parse_args()
dev = DEVICE if torch.cuda.is_available() else 'cpu'
if A.dataset == 'cifar':
    NC = 10; c = np.load(f'{OUTD}/cifar/{A.clf}/clean.npz'); SPL = pd.read_csv(f'{OUTD}/cifar/splits.csv')
    ADV = [f for f in glob.glob(f'{OUTD}/cifar/{A.clf}/*.npz') if not f.endswith('clean.npz')]; od = f'{OUTD}/cifar/feat_scores/{A.clf}'
else:
    NC = 1000; c = np.load(f'{OUTD}/clf/{A.clf}_clean.npz'); SPL = read_splits()
    ADV = glob.glob(f'{OUTD}/adv/{A.clf}/*.npz'); od = f'{OUTD}/feat_scores/{A.clf}'
sp = SPL.set_index('image_id').loc[c['image_id']].split.values
R = torch.from_numpy(c['feat'][sp == 'ref'].astype(np.float32)).to(dev); RY = torch.from_numpy(c['label'][sp == 'ref']).to(dev)
log(f'{A.clf}: ref {tuple(R.shape)}')

def shrink_prec(Z):
    cov = Z.t() @ Z / Z.shape[0]; cov += cov.diagonal().mean() * 1e-2 * torch.eye(cov.shape[0], device=dev)
    return torch.linalg.inv(cov.double()).float()
mu = R.mean(0, keepdim=True); P1 = shrink_prec(R - mu)
M = torch.stack([R[RY == k].mean(0) for k in range(NC)])
Pc = shrink_prec(R - M[RY])
h_glob = 2 * torch.cdist(R[:500], R[:500]).median().item() ** 2
wd = torch.cat([torch.pdist(R[RY == k]) for k in range(0, NC, max(1, NC // 100))]); h_cc = 2 * wd.median().item() ** 2
Rcls = [R[RY == k] for k in range(NC)]

@torch.no_grad()
def score(feat, pred):
    out = {k: [] for k in ['maha1', 'maha_cc', 'kd1', 'kde_cc', 'lid']}
    for i in range(0, len(feat), 1024):
        q = torch.from_numpy(feat[i:i + 1024].astype(np.float32)).to(dev); pr = torch.from_numpy(pred[i:i + 1024].astype(np.int64)).to(dev)
        d = q - mu; out['maha1'].append((d @ P1 * d).sum(1).clamp_min(0).sqrt())
        qP = q @ Pc; mP = M @ Pc
        dist = (qP * q).sum(1, keepdim=True) - 2 * qP @ M.t() + (mP * M).sum(1)[None]   # (b,1000) Mahalanobis^2
        out['maha_cc'].append(dist.min(1)[0].clamp_min(0).sqrt())
        d2 = torch.cdist(q, R) ** 2
        out['kd1'].append(-(torch.logsumexp(-d2 / h_glob, 1) - np.log(R.shape[0])))
        rr = d2.clamp_min(0).sqrt().topk(min(20, R.shape[0]), dim=1, largest=False)[0].clamp_min(1e-12)
        out['lid'].append(-1.0 / torch.log(rr / rr[:, -1:]).mean(1))
        kc = torch.stack([-(torch.logsumexp(-torch.cdist(q[j:j + 1], Rcls[int(pr[j])]) ** 2 / h_cc, 1)
                            - np.log(len(Rcls[int(pr[j])])))[0] for j in range(len(q))])
        out['kde_cc'].append(kc)
    return {k: torch.cat(v).cpu().numpy().astype(np.float32) for k, v in out.items()}

os.makedirs(od, exist_ok=True)
if not os.path.exists(f'{od}/clean.npz'):
    atomic_savez(f'{od}/clean.npz', image_id=c['image_id'], split=sp, **score(c['feat'], c['pred'])); log('saved clean')
for f in sorted(ADV):
    o = f'{od}/{os.path.basename(f)}'
    if f.endswith('.tmp.npz') or os.path.exists(o): continue
    d = np.load(f); atomic_savez(o, image_id=d['image_id'], success=d['success'], **score(d['feat'], d['pred'])); log('saved', o)
