"""A3 — benign-control table (ImageNet, Section 5.4): per input kind, mean over resnet50 and vit_b_16 deltas.
Flag rate / AUROC over ALL perturbed inputs (controls) or successful AEs (attacks); clean = P_test, threshold = P_cal @15%.
cpop ratio = median over images of cpop(perturbed)/cpop(clean source).  out: OUTD/analysis/a3_controls.md"""
import numpy as np
from scipy.stats import rankdata
import a1_analysis as A1   # reuses loaders (runs a1 once; cheap)
from common import OUTD
K = ['PGD_e4', 'FGSM_e4', 'TIFGSM_e4', 'Square_e4'] + [f'CTRL_{k}_e4' for k in A1.CTRL]
rows = ['| input | L2 | Linf/255 (mean) | label flip | CPOP flag@15 | CPOP AUROC | cpop ratio | Maha(class) flag / AUROC | KDE(class) flag / AUROC | FS common flag / AUROC |', '|' + '---|' * 10]
for k in K:
    vals = {}
    for c in ['resnet50', 'vit_b_16']:
        A = A1.adv_scores(c, k); ad = np.load(f'{OUTD}/adv/{c}/{k}.npz'); S = A1.CS[c][0]
        m = A['success'] if not k.startswith('CTRL') else np.ones_like(A['success'])
        src = -A1.cl['cpop'][[A1.cl_pos[i] for i in A['image_id']]]
        v = dict(l2=ad['l2'][m].mean(), li=ad['linf'][m].mean() * 255, flip=A['success'].mean(), ratio=np.median(A['cpop'][m] / src[m]))
        for d in ['cpop', 'maha_cc', 'kde_cc', 'fs_common']:
            v[d + '_f'] = A1.rate(A[d][m], A1.thr(S[d]['cal'], 0.15)); v[d + '_a'] = A1.auroc(S[d]['test'], A[d][m])
        for kk, x in v.items(): vals.setdefault(kk, []).append(x)
    g = {kk: np.mean(x) for kk, x in vals.items()}
    rows.append(f"| {k} | {g['l2']:.2f} | {g['li']:.1f} | {g['flip']:.3f} | {g['cpop_f']:.3f} | {g['cpop_a']:.3f} | {g['ratio']:.2f} | "
                f"{g['maha_cc_f']:.3f} / {g['maha_cc_a']:.3f} | {g['kde_cc_f']:.3f} / {g['kde_cc_a']:.3f} | {g['fs_common_f']:.3f} / {g['fs_common_a']:.3f} |")
open(f'{OUTD}/analysis/a3_controls.md', 'w').write('# A3 controls (resnet50 + vit_b_16 mean; attacks over successful, controls over all)\n\n' + '\n'.join(rows) + '\n')
print('\n'.join(rows))
