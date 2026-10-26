"""S6 — detector-aware (adaptive) attack against CPOP (AE #9).
Differences vs. submitted Exp J15 (12 steps, EOT 2, N=40, in-sample threshold, predicted labels):
  * GT labels, eligible = correctly classified eval images
  * detector threshold tau fixed beforehand from the CAL split (clean only), at target FPR --fpr
  * objectives  sum   : maximise CE + lam * cpop                              (submitted form)
                hinge : minimise relu(margin_y + kc) + lam * relu(tau + kd - cpop)   (threshold-targeted)
    margin_y = logit_y - max_{j!=y} logit_j ; cpop estimated by EOT over --eot probe draws (base field reused)
  * --steps PGD-Linf steps, random start, alpha = eps/--alpha_div. hinge: best iterate kept (fooled AND undetected under a
    fresh K=8 estimate, checked every --check steps; stops once all are found; else the final iterate). sum: final iterate
  * evaluation probe noise is fresh (unknown to the attacker), K=8
out: OUTD/adaptive/<clf>_<obj>_lam<lam>_s<steps>_k<eot>.npz
usage: python s6_adaptive.py --clf resnet50 --obj hinge --lam 10 --steps 100 --eot 4 --n 100"""
import os, argparse, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--clf', required=True); ap.add_argument('--obj', choices=['sum', 'hinge'], default='hinge')
ap.add_argument('--lam', type=float, default=10.); ap.add_argument('--steps', type=int, default=100)
ap.add_argument('--eot', type=int, default=4); ap.add_argument('--n', type=int, default=100); ap.add_argument('--bs', type=int, default=2)
ap.add_argument('--eps', type=int, default=4); ap.add_argument('--fpr', type=float, default=0.15); ap.add_argument('--check', type=int, default=10)
ap.add_argument('--alpha_div', type=float, default=8)                     # step = eps/alpha_div (submitted: 4)
A = ap.parse_args(); EPS = A.eps / 255; ALPHA = EPS / A.alpha_div
out = f'{OUTD}/adaptive/{A.clf}_{A.obj}_lam{A.lam:g}_s{A.steps}_k{A.eot}.npz'
if os.path.exists(out): log('exists', out); raise SystemExit
os.makedirs(os.path.dirname(out), exist_ok=True)

X, Y = load_cache(); c = np.load(f'{OUTD}/clf/{A.clf}_clean.npz')
ev = set(split_ids('eval').tolist()); m = np.array([i in ev for i in c['image_id']])
elig = c['image_id'][m][c['pred'][m] == c['label'][m]][:A.n]
cl = np.load(f'{OUTD}/scores/clean.npz'); cal = set(split_ids('cal').tolist())
cal_cpop = cl['cpop'][np.array([i in cal for i in cl['image_id']])]
TAU = {f: float(np.quantile(cal_cpop, f)) for f in [0.05, 0.15]}       # flag if cpop < tau (low cpop = adversarial-like)
tau = float(np.quantile(cal_cpop, A.fpr)); KD_, KC = 0.1 * abs(tau), 1.0
log(f'{A.clf} {A.obj} lam={A.lam} steps={A.steps} eot={A.eot} n={len(elig)} tau@{A.fpr}={tau:.5f} (cal n={len(cal_cpop)})')

clf = build_clf(A.clf, DEVICE); unet = load_unet(DEVICE, torch.float32)       # fp32: stable gradients through the field
tt = lambda B: torch.full((B,), T_STEP, device=DEVICE, dtype=torch.long)

def cpop_one(xa):
    """single-draw (1 - cos), differentiable; base field recomputed so each draw's graph is freed after backward."""
    z = xa * 2 - 1; u = unet(z, tt(len(z))).sample[:, :3].reshape(len(z), -1)
    v = unet(z + torch.randn_like(z) * SIGMA, tt(len(z))).sample[:, :3].reshape(len(z), -1)
    return 1 - F.cosine_similarity(u, v, dim=1)

def cpop_grad(xa, k, w):
    """d/dx of sum_i w_i * mean_k cpop_k(x_i), accumulated one EOT draw at a time (fp32 ADM does not fit 1+k at once)."""
    g = torch.zeros_like(xa)
    for _ in range(k):
        x_ = xa.detach().requires_grad_(True); (cpop_one(x_) * w).sum().div(k).backward(); g += x_.grad
    return g

@torch.no_grad()
def cpop_fresh(xa, k=K):
    z = xa * 2 - 1; u = unet(z, tt(len(z))).sample[:, :3].reshape(len(z), -1); acc = 0
    for _ in range(k):
        acc = acc + 1 - F.cosine_similarity(u, unet(z + torch.randn_like(z) * SIGMA, tt(len(z))).sample[:, :3].reshape(len(z), -1), dim=1)
    return acc / k

def margin(logits, y):
    ly = logits.gather(1, y[:, None])[:, 0]; o = logits.clone(); o.scatter_(1, y[:, None], -1e9); return ly - o.max(1)[0]

rows = {k: [] for k in ['image_id', 'success', 'cpop', 'clean_cpop', 'det05', 'det15', 'adv', 'found_step']}
for b in range(0, len(elig), A.bs):
    bid = elig[b:b + A.bs]; x = to_tensor(X[bid], DEVICE); y = torch.from_numpy(Y[bid]).to(DEVICE)
    torch.manual_seed(image_seed(int(bid[0]), f'adaptive|{A.obj}|{A.lam}'))
    xa = (x + torch.empty_like(x).uniform_(-EPS, EPS)).clamp(0, 1)
    best = xa.clone(); found = torch.full((len(bid),), -1, device=DEVICE)
    for s in range(A.steps):
        xa.requires_grad_(True); lg = clf(xa)
        if A.obj == 'sum':                                                       # ascent on CE + lam * cpop
            g, = torch.autograd.grad(F.cross_entropy(lg, y, reduction='sum'), xa)
            if A.lam > 0: g = g + A.lam * cpop_grad(xa, A.eot, torch.ones(len(xa), device=DEVICE))
            step = ALPHA * g.sign()
        else:                                                                    # descent on hinge(margin) + lam * hinge(tau - cpop)
            g, = torch.autograd.grad(F.relu(margin(lg, y) + KC).sum(), xa)
            if A.lam > 0:                                                        # subgradient: active where the EOT estimate is below tau + kd
                with torch.no_grad(): act = (cpop_fresh(xa.detach(), A.eot) < tau + KD_).float()
                if act.any(): g = g - A.lam * cpop_grad(xa, A.eot, act)
            step = -ALPHA * g.sign()
        xa = torch.min(torch.max(xa.detach() + step, x - EPS), x + EPS).clamp(0, 1)
        if (s + 1) % A.check == 0 or s == A.steps - 1:
            with torch.no_grad():
                xq = to_tensor(to_uint8(xa), DEVICE); ok = (clf(xq).argmax(1) != y) & (cpop_fresh(xq) >= tau)
                new = ok & (found < 0); best[new] = xa[new]; found[new] = s + 1
            if A.obj == 'hinge' and (found >= 0).all(): break                # hinge: every image fooled & undetected -> stop
    final = torch.where((found >= 0).view(-1, 1, 1, 1), best, xa) if A.obj == 'hinge' else xa   # sum = submitted protocol: last iterate
    u8 = to_uint8(final); xq = to_tensor(u8, DEVICE)
    with torch.no_grad():
        succ = (clf(xq).argmax(1) != y).cpu().numpy()
        torch.manual_seed(image_seed(int(bid[0]), 'adaptive_eval'))      # fresh probe noise, not the attacker's
        cp = cpop_fresh(xq).cpu().numpy(); cc = cpop_fresh(x).cpu().numpy()
    for k, v in zip(rows, [bid, succ, cp, cc, cp < TAU[0.05], cp < TAU[0.15], u8, found.cpu().numpy()]): rows[k].append(v)
    if (b // A.bs) % 10 == 0:
        S = np.concatenate(rows['success']); D = np.concatenate(rows['det15'])
        log(f'  {b + len(bid)}/{len(elig)} success={S.mean():.3f} det15|succ={D[S].mean() if S.any() else float("nan"):.3f} evade&fool={(S & ~D).mean():.3f}')
R = {k: np.concatenate(v) for k, v in rows.items()}
atomic_savez(out, tau05=TAU[0.05], tau15=TAU[0.15], **R)
S, D = R['success'], R['det15']
log(f'saved {out}: success {S.mean():.3f}  TPR@15|succ {D[S].mean() if S.any() else float("nan"):.3f}  fooled&undetected {(S & ~D).mean():.3f}')
