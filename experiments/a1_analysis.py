"""A1 — CPU analysis of the R1 re-experiment scores -> markdown tables (AE #1 2 5 6 7 8, + decomposition).
Light: numpy only, reads score files (never the adv image arrays). Run with --cpus 1.
All detector scores are oriented HIGHER = more suspicious (cpop, raw_var negated). Flag if s > q_{1-f}(clean P_cal).
TPR is over successful adversarial examples (pred(adv) != GT on correctly classified P_eval images).
Regimes: R2 = threshold from the target classifier's own clean P_cal; R1 = threshold from a different (source)
classifier's clean P_cal applied to the target. Input-only scores (cpop/raw_var/score_norm) have R1 == R2 by construction.
out: OUTD/analysis/a1_tables.md (+ .json)
usage: python a1_analysis.py"""
import os, glob, json, functools, numpy as np
from scipy.stats import rankdata, spearmanr
from common import OUTD, read_splits, SIGMA, log

CLFS = ['resnet50', 'vgg16', 'densenet121', 'mobilenet_v3_large', 'convnext_tiny', 'vit_b_16', 'swin_t', 'maxvit_t']
CLF4 = ['resnet50', 'densenet121', 'vit_b_16', 'swin_t']
ATK12 = ['FGSM', 'BIM', 'RFGSM', 'PGD', 'MIFGSM', 'DIFGSM', 'TIFGSM', 'NIFGSM', 'APGD', 'AutoAttack', 'CW', 'Square']
CTRL = ['sign', 'gauss', 'phase', 'shuffle', 'jpeg75', 'blur']
INPUT = ['cpop', 'raw_var', 'score_norm']
FEAT = ['maha1', 'maha_cc', 'kd1', 'kde_cc', 'lid']
FS = ['fs_common', 'fs_bit2']
DETS = INPUT + FEAT + FS
SIGN = {'cpop': -1, 'raw_var': -1}                      # low cpop / low raw_var = adversarial-like
FPRS = [0.01, 0.05, 0.10, 0.15]
rng = np.random.default_rng(0)
OUT = f'{OUTD}/analysis'; os.makedirs(OUT, exist_ok=True)
md, js = [], {}
def H(s): md.append(f'\n## {s}\n')
def T(head, rows):
    md.append('| ' + ' | '.join(head) + ' |'); md.append('|' + '---|' * len(head))
    for r in rows: md.append('| ' + ' | '.join(r) + ' |')
f3 = lambda v: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.3f}'

def auroc(neg, pos):
    r = rankdata(np.concatenate([neg, pos])); n0, n1 = len(neg), len(pos)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
def thr(cal, f): return np.quantile(cal, 1 - f)
def rate(x, t): return float((x > t).mean()) if len(x) else np.nan

# ---------------------------------------------------------------- load
spl = read_splits().set_index('image_id').split
cl = np.load(f'{OUTD}/scores/clean.npz'); cl_sp = spl.loc[cl['image_id']].values
CLEAN_IN = {d: SIGN.get(d, 1) * cl[d].astype(np.float64) for d in INPUT}
cl_pos = {i: j for j, i in enumerate(cl['image_id'])}

def clean_scores(clf):
    """per detector: dict split -> scores, plus id arrays; classifier-specific detectors from feat/fs files."""
    S = {d: {s: v[cl_sp == s] for s in ['cal', 'test', 'eval']} for d, v in CLEAN_IN.items()}
    ids = {s: cl['image_id'][cl_sp == s] for s in ['cal', 'test', 'eval']}
    fe = np.load(f'{OUTD}/feat_scores/{clf}/clean.npz', allow_pickle=True); fsp = fe['split'].astype(str)
    c = np.load(f'{OUTD}/clf/{clf}_clean.npz'); csp = spl.loc[c['image_id']].values
    for d in FEAT: S[d] = {s: fe[d][fsp == s].astype(np.float64) for s in ['cal', 'test', 'eval']}
    for d in FS: S[d] = {s: c[d][csp == s].astype(np.float64) for s in ['cal', 'test', 'eval']}
    # sanity: same id order across sources within each split
    for s in ['cal', 'test', 'eval']:
        assert (fe['image_id'][fsp == s] == ids[s]).all() and (c['image_id'][csp == s] == ids[s]).all(), (clf, s)
    acc = {s: (c['pred'][csp == s] == c['label'][csp == s]) for s in ['cal', 'test', 'eval']}
    return S, ids, acc

@functools.lru_cache(maxsize=None)
def adv_scores(clf, tag):
    sc = f'{OUTD}/scores/adv/{clf}/{tag}.npz'
    if not os.path.exists(sc): return None
    a = np.load(sc); out = {'image_id': a['image_id'], 'success': a['success']}
    for d in INPUT: out[d] = SIGN.get(d, 1) * a[d].astype(np.float64)
    fe = np.load(f'{OUTD}/feat_scores/{clf}/{tag}.npz'); ad = np.load(f'{OUTD}/adv/{clf}/{tag}.npz')
    assert (fe['image_id'] == a['image_id']).all() and (ad['image_id'] == a['image_id']).all()
    for d in FEAT: out[d] = fe[d].astype(np.float64)
    for d in FS: out[d] = ad[d].astype(np.float64)
    return out

CS = {c: clean_scores(c) for c in CLFS}; log('loaded clean')

# ---------------------------------------------------------------- 1. R1 / R2 / AUROC, PGD e4, 8 classifiers (AE #1 #6)
H('1. PGD ε=4/255, 8 classifiers — R2 (own clean P_cal), R1 (other classifier\'s P_cal), AUROC')
md.append('TPR over successful AEs; FPR realised on P_test (10k, disjoint from P_cal). R1 = mean over the 7 other '
          'source classifiers (min–max of realised FPR in brackets). Input-only scores: R1 ≡ R2.')
res1 = {}
for f in [0.05, 0.15]:
    rows = []
    for d in DETS:
        r2t, r2f, r1t, r1f, r1lo, r1hi, au = [], [], [], [], [], [], []
        for c in CLFS:
            S = CS[c][0]; A = adv_scores(c, 'PGD_e4'); pos = A[d][A['success']]
            t = thr(S[d]['cal'], f); r2t.append(rate(pos, t)); r2f.append(rate(S[d]['test'], t))
            au.append(auroc(S[d]['test'], pos))
            ft, ff = [], []
            for src in CLFS:
                if src == c: continue
                ts = thr(CS[src][0][d]['cal'], f); ft.append(rate(pos, ts)); ff.append(rate(S[d]['test'], ts))
            r1t.append(np.mean(ft)); r1f.append(np.mean(ff)); r1lo.append(min(ff)); r1hi.append(max(ff))
        res1[(d, f)] = dict(r2_tpr=r2t, r2_fpr=r2f, r1_tpr=r1t, r1_fpr=r1f, auroc=au)
        rows.append([d, f3(np.mean(r2f)), f3(np.mean(r2t)), f'{np.mean(r1f):.3f} [{min(r1lo):.3f}–{max(r1hi):.3f}]',
                     f3(np.mean(r1t)), f3(np.mean(au))])
    md.append(f'\n**target FPR {int(f*100)}%** (mean over 8 classifiers)\n')
    T(['detector', 'R2 FPR', 'R2 TPR', 'R1 FPR [range]', 'R1 TPR', 'AUROC'], rows)
js['t1'] = {f'{d}@{f}': v for (d, f), v in res1.items()}

md.append('\n**Per-classifier R2 TPR @15% (PGD ε=4)**\n')
rows = []
for d in DETS: rows.append([d] + [f3(v) for v in res1[(d, 0.15)]['r2_tpr']])
T(['detector'] + CLFS, rows)
md.append('\n**Per-classifier R1 FPR @15% for feature-space detectors, source = resnet50 (threshold transferred raw)**\n')
rows = []
for d in FEAT + FS:
    ts = thr(CS['resnet50'][0][d]['cal'], 0.15); rows.append([d] + [f3(rate(CS[c][0][d]['test'], ts)) for c in CLFS])
T(['detector'] + CLFS, rows)

# ---------------------------------------------------------------- 2. n_cal resampling (AE #2)
H('2. Calibration-set size — realised P_test FPR of the CPOP threshold (B=1000 resamples of P_cal)')
cal, test = CS['resnet50'][0]['cpop']['cal'], CS['resnet50'][0]['cpop']['test']
posR = {c: (lambda A: A['cpop'][A['success']])(adv_scores(c, 'PGD_e4')) for c in ['resnet50', 'vit_b_16']}
rows = []; js['t2'] = {}
for f in FPRS:
    for n in [100, 300, 1000, 3000, len(cal)]:
        B = 1 if n == len(cal) else 1000
        ts = np.array([thr(cal if B == 1 else rng.choice(cal, n, replace=False), f) for _ in range(B)])
        fp = (test[None] > ts[:, None]).mean(1)
        tp = {c: (p[None] > ts[:, None]).mean(1) for c, p in posR.items()}
        q = lambda v: f'{v.mean():.3f} [{np.quantile(v, .025):.3f}, {np.quantile(v, .975):.3f}]' if B > 1 else f'{v.mean():.3f}'
        rows.append([f'{int(f*100)}%', str(n), q(fp), f'{fp.std():.4f}' if B > 1 else '—',
                     f'{np.sqrt(f*(1-f)/n):.4f}', q(tp['resnet50']), q(tp['vit_b_16'])])
        js['t2'][f'{f}_{n}'] = dict(fpr_mean=float(fp.mean()), fpr_sd=float(fp.std()))
T(['target', 'n_cal', 'P_test FPR mean [95%]', 'sd', '√(f(1−f)/n)', 'TPR R50 PGD4', 'TPR ViT PGD4'], rows)

# ---------------------------------------------------------------- 3. selection effect (AE #2 #5)
H('3. Selection effect — clean FPR of the same CPOP threshold on different clean populations (PGD ε=4)')
md.append('P_test = all; E_c = P_eval images classifier c gets right; S_c = clean originals of successful AEs.')
cv = CLEAN_IN['cpop']; rows = []
for c in CLFS:
    S, ids, acc = CS[c]; A = adv_scores(c, 'PGD_e4')
    e_ids = A['image_id']; s_ids = A['image_id'][A['success']]
    ce = cv[[cl_pos[i] for i in e_ids]]; cs = cv[[cl_pos[i] for i in s_ids]]
    r = [c, str(len(e_ids)), f3(A['success'].mean())]
    for f in [0.05, 0.15]:
        t = thr(S['cpop']['cal'], f); r += [f3(rate(S['cpop']['test'], t)), f3(rate(S['cpop']['eval'], t)), f3(rate(ce, t)), f3(rate(cs, t))]
    rows.append(r)
T(['clf', '|E_c|', 'succ', 'P_test @5', 'P_eval @5', 'E_c @5', 'S_c @5', 'P_test @15', 'P_eval @15', 'E_c @15', 'S_c @15'], rows)
# misclassified vs correct clean (does the filter prefer hard images?)
md.append('\n**CPOP flag rate on P_test split by classifier correctness (@15%)**\n')
rows = []
for c in CLFS:
    S, ids, acc = CS[c]; t = thr(S['cpop']['cal'], 0.15)
    rows.append([c, f3(acc['test'].mean()), f3(rate(S['cpop']['test'][acc['test']], t)), f3(rate(S['cpop']['test'][~acc['test']], t))])
T(['clf', 'clean acc (P_test)', 'flag | correct', 'flag | wrong'], rows)

# ---------------------------------------------------------------- 4. system level (AE #5 #7)
H('4. System level — clean rerouting and end-to-end residual error (CPOP, PGD ε=4)')
md.append('Clean side on P_test: rerouted = flagged fraction; acc(pass) / acc(rerouted) = classifier accuracy on each part. '
          'Attack side over E_c (all attacked inputs, successful or not): no-filter error = success rate; '
          'residual = fooled AND not flagged.')
rows = []
for c in CLFS:
    S, ids, acc = CS[c]; A = adv_scores(c, 'PGD_e4')
    for f in [0.05, 0.10, 0.15]:
        t = thr(S['cpop']['cal'], f); fl = S['cpop']['test'] > t; fa = A['cpop'] > t
        rows.append([c, f'{int(f*100)}%', f3(fl.mean()), f3(acc['test'][~fl].mean()), f3(acc['test'][fl].mean()),
                     f3(A['success'].mean()), f3((A['success'] & ~fa).mean())])
T(['clf', 'target FPR', 'clean rerouted', 'acc(pass)', 'acc(rerouted)', 'no-filter attack err', 'residual err'], rows)

# ---------------------------------------------------------------- 5. epsilon cohort (AE #5)
H('5. ε sweep on the ε-controlled cohort (images whose attack succeeds at all three ε)')
rows = []; agg = {e: [] for e in [2, 4, 8]}
for c in CLFS:
    S = CS[c][0]; As = {e: adv_scores(c, f'PGD_e{e}') for e in [2, 4, 8]}
    for e in [2, 4, 8]: assert (As[e]['image_id'] == As[4]['image_id']).all()
    co = As[2]['success'] & As[4]['success'] & As[8]['success']
    r = [c, str(len(co)), ' / '.join(f3(As[e]['success'].mean()) for e in [2, 4, 8]), str(int(co.sum()))]
    for e in [2, 4, 8]:
        p = As[e]['cpop'][co]; t = thr(S['cpop']['cal'], 0.15); a = auroc(S['cpop']['test'], p)
        r += [f'{a:.3f} / {rate(p, t):.3f}']; agg[e].append((a, rate(p, t)))
    rows.append(r)
rows.append(['**mean**', '', '', ''] + [f'{np.mean([x[0] for x in agg[e]]):.3f} / {np.mean([x[1] for x in agg[e]]):.3f}' for e in [2, 4, 8]])
T(['clf', '|E_c|', 'succ ε2/4/8', 'cohort', 'ε=2 AUROC / TPR@15', 'ε=4', 'ε=8'], rows)

# ---------------------------------------------------------------- 6. attack families + controls, all detectors (AE #4 #6)
H('6. Attack families (4 classifiers) and benign controls — R2 TPR@15% / AUROC per detector')
def fam_row(kinds, clfs, tagf):
    rows = []
    for k in kinds:
        r = [k]; ns, sr = [], []
        for d in ['cpop', 'raw_var', 'score_norm', 'maha_cc', 'kde_cc', 'lid', 'fs_common']:
            tp, au = [], []
            for c in clfs:
                A = adv_scores(c, tagf(k))
                if A is None: continue
                S = CS[c][0]; pos = A[d][A['success']]
                tp.append(rate(pos, thr(S[d]['cal'], 0.15))); au.append(auroc(S[d]['test'], pos))
                if d == 'cpop': ns.append(len(A['success'])); sr.append(A['success'].mean())
            r.append(f'{np.mean(tp):.3f} / {np.mean(au):.3f}')
        rows.append([r[0], f'{int(np.mean(ns))}', f3(np.mean(sr))] + r[1:])
    return rows
head = ['input', 'n/clf', 'succ', 'cpop', 'raw_var', 'score_norm', 'maha_cc', 'kde_cc', 'lid', 'fs_common']
T(head, fam_row(ATK12, CLF4, lambda k: f'{k}_e4'))
md.append('\n**Benign controls (ResNet-50 + ViT-B/16 deltas; "succ" = label flipped, TPR over flipped only)**\n')
md.append('Controls flip few labels, so TPR here is also given over ALL perturbed inputs below.\n')
T(head, fam_row(['PGD'] + [f'CTRL_{k}' for k in CTRL], ['resnet50', 'vit_b_16'], lambda k: f'{k}_e4'))
rows = []
for k in ['PGD'] + [f'CTRL_{k}' for k in CTRL]:
    r = [k]
    for d in ['cpop', 'maha_cc', 'kde_cc', 'fs_common']:
        tp = [rate(adv_scores(c, f'{k}_e4')[d], thr(CS[c][0][d]['cal'], 0.15)) for c in ['resnet50', 'vit_b_16']]
        r.append(f3(np.mean(tp)))
    rows.append(r)
T(['input', 'cpop flag rate (all)', 'maha_cc', 'kde_cc', 'fs_common'], rows)

# ---------------------------------------------------------------- 7. probe seed variance (AE #2 #8)
H('7. Probe randomness — 5 probe-noise seeds on the first 1000 P_test images')
ids0 = CS['resnet50'][1]['test'][:1000]; base = cv[[cl_pos[i] for i in ids0]]
V = [base] + [-np.load(f'{OUTD}/scores/seeds_s{j}.npz')['cpop'].astype(np.float64) for j in range(1, 5)]
for j in range(1, 5): assert (np.load(f'{OUTD}/scores/seeds_s{j}.npz')['image_id'] == ids0).all()
V = np.stack(V); sd = V.std(0, ddof=1); spread = (-V).mean(0)
rho = [spearmanr(V[0], V[j])[0] for j in range(1, 5)]
rows = [['per-image sd / mean cpop (median)', f3(np.median(sd / np.abs(V.mean(0))))],
        ['Spearman(main, seed j), min', f3(min(rho))]]
for f in [0.05, 0.15]:
    t = thr(CS['resnet50'][0]['cpop']['cal'], f); fr = (V > t).mean(1)
    rows.append([f'FPR across seeds @{int(f*100)}% (min–max)', f'{fr.min():.3f}–{fr.max():.3f}'])
    fl = V > t; rows.append([f'images whose flag changes across seeds @{int(f*100)}%', f3((fl.any(0) & ~fl.all(0)).mean())])
T(['quantity', 'value'], rows)

# ---------------------------------------------------------------- 8. sigma grid + decomposition (AE #3 #8)
H('8. σ grid (300 R50 PGD ε=4 images, successful only) and the CPOP ≈ ½σ²·raw_var/‖ε̂‖² identity')
ref = adv_scores('resnet50', 'PGD_e4'); ok = ref['success'][:300]
rows = []
for sg in [0.00125, 0.0025, 0.005, 0.01, 0.02]:
    c_ = np.load(f'{OUTD}/scores/sigma/clean_s{sg}.npz'); a_ = np.load(f'{OUTD}/scores/sigma/adv_s{sg}.npz')
    assert (a_['image_id'] == ref['image_id'][:300]).all()
    cc, aa = c_['cpop'].astype(np.float64), a_['cpop'][ok].astype(np.float64)
    pred_c = 0.5 * sg ** 2 * c_['raw_var'] / c_['score_norm'] ** 2
    rows.append([str(sg), f'{np.median(cc):.2e}', f'{np.median(cc) / sg**2:.1f}', f3(np.median(aa) / np.median(cc)),
                 f3(auroc(-cc, -aa)), f3(spearmanr(c_['cpop'], pred_c)[0]), f3(np.median(c_['cpop'] / pred_c))])
T(['σ', 'median clean cpop', 'cpop/σ²', 'adv/clean median', 'AUROC (paired 300)', 'Spearman(cpop, ½σ²rv/‖ε̂‖²)', 'median ratio'], rows)
pc = 0.5 * SIGMA ** 2 * cl['raw_var'] / cl['score_norm'] ** 2
A = adv_scores('resnet50', 'PGD_e4'); pa = 0.5 * SIGMA ** 2 * (-A['raw_var']) / A['score_norm'] ** 2
md.append(f'\nFull data σ=0.005: Spearman(cpop, prediction) clean {spearmanr(cl["cpop"], pc)[0]:.4f} (n={len(pc)}), '
          f'R50 PGD4 adv {spearmanr(-A["cpop"], pa)[0]:.4f}; median ratio clean {np.median(cl["cpop"] / pc):.3f}, adv {np.median(-A["cpop"] / pa):.3f}.')
ce = np.array([cl_pos[i] for i in A['image_id'][A['success']]]); s = A['success']
md.append(f'Paired clean→adv (R50 PGD4, successful): raw_var ×{np.median(-A["raw_var"][s] / cl["raw_var"][ce]):.3f}, '
          f'‖ε̂‖ ×{np.median(A["score_norm"][s] / cl["score_norm"][ce]):.3f}, cpop ×{np.median(-A["cpop"][s] / cl["cpop"][ce]):.3f}; '
          f'raw_var decreases in {np.mean(-A["raw_var"][s] < cl["raw_var"][ce]):.3f}, ‖ε̂‖ increases in {np.mean(A["score_norm"][s] > cl["score_norm"][ce]):.3f}.')

# ---------------------------------------------------------------- 9. attack-free combination of the two factors
H('9. Attack-free two-factor score: clean-fitted 2-D Gaussian on (log raw_var, log ‖ε̂‖), two-sided')
md.append('Fitted on clean P_cal only (no attacks). Compared with cpop at R2 TPR@15% / AUROC.')
Z = np.stack([np.log(cl['raw_var']), np.log(cl['score_norm'])], 1).astype(np.float64)
mu_ = Z[cl_sp == 'cal'].mean(0); P_ = np.linalg.inv(np.cov(Z[cl_sp == 'cal'].T))
m2 = lambda z: np.einsum('ij,jk,ik->i', z - mu_, P_, z - mu_)
c2 = {s: m2(Z[cl_sp == s]) for s in ['cal', 'test']}; t2 = thr(c2['cal'], 0.15)
rows = []
for k, clfs in [('PGD_e2', CLFS), ('PGD_e4', CLFS), ('PGD_e8', CLFS), ('TIFGSM_e4', CLF4), ('CW_e4', CLF4), ('Square_e4', CLF4),
                ('CTRL_gauss_e4', ['resnet50', 'vit_b_16']), ('CTRL_blur_e4', ['resnet50', 'vit_b_16']), ('CTRL_jpeg75_e4', ['resnet50', 'vit_b_16'])]:
    a1, a2, b1, b2 = [], [], [], []
    for c in clfs:
        A = adv_scores(c, k); s = A['success'] if not k.startswith('CTRL') else np.ones_like(A['success'])
        za = np.stack([np.log(-A['raw_var'][s]), np.log(A['score_norm'][s])], 1); sa = m2(za)
        S = CS[c][0]; pc_ = A['cpop'][s]
        a1.append(rate(pc_, thr(S['cpop']['cal'], 0.15))); a2.append(auroc(S['cpop']['test'], pc_))
        b1.append(rate(sa, t2)); b2.append(auroc(c2['test'], sa))
    rows.append([k, f'{np.mean(a1):.3f} / {np.mean(a2):.3f}', f'{np.mean(b1):.3f} / {np.mean(b2):.3f}'])
md.append(f'Realised P_test FPR of the 2-D score @15%: {rate(c2["test"], t2):.3f}. Controls use all perturbed inputs.\n')
T(['input', 'cpop TPR@15 / AUROC', '2-factor TPR@15 / AUROC'], rows)

# ---------------------------------------------------------------- 10. per-classifier AUROC (Table 4) and FS squeezer sweep (Table 5)
H('10. Per-classifier AUROC, PGD ε=4 (P_test clean vs successful AEs)')
DET10 = INPUT + ['fs_common', 'fs_bit2', 'lid', 'maha1', 'maha_cc', 'kd1', 'kde_cc']
rows = []; per = {d: [] for d in DET10}
for c in CLFS:
    S = CS[c][0]; A = adv_scores(c, 'PGD_e4'); r = [c]
    for d in DET10: v = auroc(S[d]['test'], A[d][A['success']]); per[d].append(v); r.append(f3(v))
    rows.append(r)
for nm, fn in [('mean', np.mean), ('std', np.std), ('worst', np.min)]: rows.append([nm] + [f3(fn(per[d])) for d in DET10])
T(['clf'] + DET10, rows); js['t10'] = per

H('11. Feature Squeezing AUROC per squeezer, PGD ε=4 (orientation fixed: larger L1 change = suspicious)')
SQ = ['fs_bit1', 'fs_bit2', 'fs_bit3', 'fs_bit5', 'fs_median3', 'fs_median5', 'fs_avgpool3', 'fs_common']
rows = []
for sq in SQ:
    r = [sq]
    for c in CLFS:
        cc = np.load(f'{OUTD}/clf/{c}_clean.npz'); csp = spl.loc[cc['image_id']].values
        ad = np.load(f'{OUTD}/adv/{c}/PGD_e4.npz'); m = ad['success']
        r.append(f3(auroc(cc[sq][csp == 'test'].astype(np.float64), ad[sq][m].astype(np.float64))))
    rows.append(r)
T(['squeezer'] + CLFS, rows)
md.append('\nClean-only squeezer choice (no attacks): the squeezer is fixed by a clean-data criterion — not possible without '
          'attacks for AUROC, so FS variants are reported as common (bit3 ∨ avgpool3, Xu et al. ImageNet-style) and a fixed 2-bit.')

H('12. R2 TPR vs target FPR (mean over 8 classifiers, PGD ε=4) — data for the matched-FPR figure')
GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]; rows = []; js['t12'] = {}
for d in DETS:
    v = []
    for f in GRID:
        v.append(np.mean([rate((lambda A: A[d][A['success']])(adv_scores(c, 'PGD_e4')), thr(CS[c][0][d]['cal'], f)) for c in CLFS]))
    js['t12'][d] = v; rows.append([d] + [f3(x) for x in v])
T(['detector'] + [f'{int(f*100)}%' for f in GRID], rows)
js['r1_fpr15'] = {d: [rate(CS[t][0][d]['test'], thr(CS[s][0][d]['cal'], 0.15)) for s in CLFS for t in CLFS if s != t] for d in DETS}

# ---------------------------------------------------------------- 13. input-only scores incl. the 2-factor score, several inputs
H('13. Input-only scores (R1 = R2): AUROC and TPR@5/15 on PGD ε=2/4/8 (8 clf), TI-FGSM (4 clf), Gaussian/blur controls (2 clf, all inputs)')
def two_factor(raw, nrm): return m2(np.stack([np.log(raw), np.log(nrm)], 1))
SC = {'cpop': lambda A, s: A['cpop'][s], 'score_norm': lambda A, s: A['score_norm'][s], 'raw_var': lambda A, s: A['raw_var'][s],
      '2factor': lambda A, s: two_factor(-A['raw_var'][s], A['score_norm'][s])}
CLN = {'cpop': {k: CLEAN_IN['cpop'][cl_sp == k] for k in ['cal', 'test']}, 'score_norm': {k: CLEAN_IN['score_norm'][cl_sp == k] for k in ['cal', 'test']},
       'raw_var': {k: CLEAN_IN['raw_var'][cl_sp == k] for k in ['cal', 'test']}, '2factor': c2}
rows = []
for k, clfs in [('PGD_e2', CLFS), ('PGD_e4', CLFS), ('PGD_e8', CLFS), ('TIFGSM_e4', CLF4), ('CTRL_gauss_e4', ['resnet50', 'vit_b_16']), ('CTRL_blur_e4', ['resnet50', 'vit_b_16'])]:
    r = [k]
    for d, fn in SC.items():
        au, t5, t15 = [], [], []
        for c in clfs:
            A = adv_scores(c, k); s = A['success'] if not k.startswith('CTRL') else np.ones_like(A['success']); p = fn(A, s)
            au.append(auroc(CLN[d]['test'], p)); t5.append(rate(p, thr(CLN[d]['cal'], 0.05))); t15.append(rate(p, thr(CLN[d]['cal'], 0.15)))
        r.append(f'{np.mean(au):.3f} / {np.mean(t5):.3f} / {np.mean(t15):.3f}')
    rows.append(r)
T(['input'] + [f'{d} AUROC / TPR@5 / TPR@15' for d in SC], rows)

open(f'{OUT}/a1_tables.md', 'w').write('# A1 analysis tables (auto-generated by a1_analysis.py)\n' + '\n'.join(md) + '\n')
json.dump(js, open(f'{OUT}/a1_tables.json', 'w'), default=lambda o: float(o) if np.isscalar(o) else list(map(float, o)))
log('wrote', f'{OUT}/a1_tables.md')
