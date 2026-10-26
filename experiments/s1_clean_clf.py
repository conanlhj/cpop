"""S1 — clean pass of one classifier over ref/cal/test/eval: prediction, max-prob, FS (all squeezers), penultimate features.
out: OUTD/clf/<clf>_clean.npz  (keys: image_id, label, pred, maxprob, feat[f16], fs_*)
usage: python s1_clean_clf.py --clf resnet50 [--bs 64]"""
import os, argparse, numpy as np, torch
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--clf', required=True); ap.add_argument('--bs', type=int, default=64); A = ap.parse_args()
out = f'{OUTD}/clf/{A.clf}_clean.npz'
if os.path.exists(out): log('exists', out); raise SystemExit
pdir = f'{OUTD}/clf/{A.clf}_clean_parts'; os.makedirs(pdir, exist_ok=True)
X, Y = load_cache(); ids = np.concatenate([split_ids(s) for s in ['ref', 'cal', 'test', 'eval']]); ids.sort()
clf = build_clf(A.clf, DEVICE); hook = FeatHook(clf)
PART = 2048
for pi, pid in parts(ids, PART):
    pf = f'{pdir}/part_{pi:04d}.npz'
    if os.path.exists(pf): continue
    acc = {}
    for b in range(0, len(pid), A.bs):
        bid = pid[b:b + A.bs]; r = clf_pass(clf, hook, to_tensor(X[bid], DEVICE))
        for k, v in r.items(): acc.setdefault(k, []).append(v)
    acc = {k: np.concatenate(v) for k, v in acc.items()}
    atomic_savez(pf, image_id=pid, label=Y[pid], **acc)
    log(f'{A.clf} part {pi} ({(pi + 1) * PART}/{len(ids)}) acc={np.mean(acc["pred"] == Y[pid]):.3f}')
merge_parts(pdir, out)
d = np.load(out); s = read_splits().set_index('image_id').loc[d['image_id']].split.values
log(f'{A.clf} top-1 (GT) ' + ' '.join(f'{k}={np.mean(d["pred"][s == k] == d["label"][s == k]):.4f}' for k in ['ref', 'cal', 'test', 'eval']))
log('saved', out)
