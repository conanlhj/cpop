"""P1 Exp E — operating-point (threshold) transfer across classifiers.

The honest differentiator: AUROC of FS_tuned is competitive, BUT a deployed detector
needs a fixed threshold. CPOP's score is computed from the diffusion model + image only
(classifier-independent), so a threshold calibrated on one classifier transfers to any
other. FS's score is the classifier's own prediction-disagreement, so its scale is
classifier-dependent and a transferred threshold mis-fires.

Pure analysis of Exp A per-sample scores (no GPU). For each detector:
  - calibrate threshold tau on SOURCE classifier clean scores at target FPR=5%
  - apply tau to every TARGET classifier: realized FPR (clean>tau) and TPR (adv>tau)
  - report FPR drift |realized-0.05| and TPR under transfer.
Also the model-zoo scenario: one global threshold (calibrated on a held-out classifier)
applied to all classifiers -> spread of realized FPR.
"""
import pandas as pd, numpy as np
IN  = '/workspace/outputs/tables/p1_expA_cross_arch.csv'
OUT = '/workspace/outputs/tables/p1_expE_threshold_transfer.csv'
TARGET_FPR = 0.05
EPS = 4
# fs_bit2 = FS with a fixed strong squeezer (gives FS a fair shot on AUROC, so the test
# isolates threshold-scale transfer, not squeezer choice).
DETS = {'cpop':'cpop', 'fs_common':'fs', 'fs_bit2':'fs_bit2', 'lid':'lid'}

df = pd.read_csv(IN); df = df[df.eps255 == EPS]
archs = sorted(df.classifier.unique())

def tau_at_fpr(clean_scores, fpr):
    return np.quantile(clean_scores, 1 - fpr)            # higher score -> adv

rows = []
for dname, col in DETS.items():
    for S in archs:
        cs = df[(df.classifier==S)&(df.label==0)][col].dropna().values
        if len(cs) < 5: continue
        tau = tau_at_fpr(cs, TARGET_FPR)
        for T in archs:
            ct = df[(df.classifier==T)&(df.label==0)][col].dropna().values
            at = df[(df.classifier==T)&(df.label==1)][col].dropna().values
            if len(ct) < 5 or len(at) < 5: continue
            rows.append({'detector':dname,'source':S,'target':T,
                         'realized_fpr':float((ct>tau).mean()),
                         'tpr':float((at>tau).mean()),
                         'self': S==T})
R = pd.DataFrame(rows); R.to_csv(OUT, index=False); print('saved', OUT, len(R), flush=True)

print(f'\n=== threshold TRANSFER (calibrate 5% FPR on source, apply to OTHER classifiers) eps={EPS}/255 ===', flush=True)
print(f'{"detector":10s} {"FPRdrift_mean":>13s} {"realFPR_mean":>12s} {"realFPR_std":>11s} {"TPR_mean":>9s} {"TPR_min":>8s}', flush=True)
summ=[]
for dname in DETS:
    tr = R[(R.detector==dname)&(~R.self)]               # cross-classifier only
    if len(tr)==0: continue
    drift = (tr.realized_fpr - TARGET_FPR).abs().mean()
    row={'detector':dname,'fpr_drift':round(drift,3),'realfpr_mean':round(tr.realized_fpr.mean(),3),
         'realfpr_std':round(tr.realized_fpr.std(),3),'tpr_mean':round(tr.tpr.mean(),3),'tpr_min':round(tr.tpr.min(),3)}
    summ.append(row)
    print(f'{dname:10s} {row["fpr_drift"]:13.3f} {row["realfpr_mean"]:12.3f} {row["realfpr_std"]:11.3f} {row["tpr_mean"]:9.3f} {row["tpr_min"]:8.3f}', flush=True)
pd.DataFrame(summ).to_csv('/workspace/outputs/tables/p1_expE_summary.csv', index=False)

print('\n=== self (oracle, threshold calibrated on the SAME classifier) for reference ===', flush=True)
for dname in DETS:
    sr = R[(R.detector==dname)&(R.self)]
    if len(sr)==0: continue
    print(f'  {dname:10s} realFPR={sr.realized_fpr.mean():.3f}  TPR={sr.tpr.mean():.3f}', flush=True)

print('\nReading: low FPR-drift + stable realized-FPR + high TPR under transfer = threshold is portable.', flush=True)
print('CPOP (classifier-independent score) should transfer cleanly; FS (classifier-dependent scale) should drift.', flush=True)
