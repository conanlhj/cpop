"""S2 — attacks (and benign controls) on the eval split, ground-truth labels.
Eligible = eval images the classifier classifies correctly (from S1). Attack target y = GT label.
Adversarial images are quantised to uint8 BEFORE success is judged (what a deployed system receives).
Classifier-side quantities (pred, FS, penultimate feat) of the quantised image are stored alongside.

out: OUTD/adv/<clf>/<kind>_e<eps255>.npz  keys: image_id, label, adv(u8), success, pred, maxprob, feat, fs_*, linf, l2
usage: python s2_attack.py --clf resnet50 --kind PGD --eps 4 [--n 0 (=all eligible)]
kinds: attacks  FGSM BIM RFGSM PGD MIFGSM DIFGSM TIFGSM NIFGSM APGD AutoAttack CW DeepFool Square
       controls CTRL_sign CTRL_gauss CTRL_phase CTRL_shuffle CTRL_jpeg75 CTRL_blur   (need PGD e<eps> of same clf)
"""
import os, io, argparse, numpy as np, torch, torch.nn.functional as F
from common import *
import sys; sys.path.append(CODE)
import torchattacks
from utils.patches import apply_cw_patch
apply_cw_patch()

ap = argparse.ArgumentParser()
ap.add_argument('--clf', required=True); ap.add_argument('--kind', required=True); ap.add_argument('--eps', type=int, default=4)
ap.add_argument('--n', type=int, default=0); ap.add_argument('--bs', type=int, default=16)
A = ap.parse_args(); EPS = A.eps / 255
out = f'{OUTD}/adv/{A.clf}/{A.kind}_e{A.eps}.npz'
if os.path.exists(out): log('exists', out); raise SystemExit
pdir = out[:-4] + '_parts'; os.makedirs(pdir, exist_ok=True)

X, Y = load_cache(); c = np.load(f'{OUTD}/clf/{A.clf}_clean.npz')
ev = set(split_ids('eval').tolist()); m = np.array([i in ev for i in c['image_id']])
elig = c['image_id'][m][c['pred'][m] == c['label'][m]]           # correctly classified eval images
if A.n: elig = elig[:A.n]
log(f'{A.clf} {A.kind} eps={A.eps}/255: eligible {len(elig)} / eval {m.sum()}')
clf = build_clf(A.clf, DEVICE); hook = FeatHook(clf)

def make_atk(kind):
    a = EPS / 4
    if kind == 'FGSM':    return torchattacks.FGSM(clf, eps=EPS)
    if kind == 'BIM':     return torchattacks.BIM(clf, eps=EPS, alpha=a, steps=20)
    if kind == 'RFGSM':   return torchattacks.RFGSM(clf, eps=EPS, alpha=a, steps=20)
    if kind == 'PGD':     return torchattacks.PGD(clf, eps=EPS, alpha=a, steps=20, random_start=True)
    if kind == 'MIFGSM':  return torchattacks.MIFGSM(clf, eps=EPS, alpha=a, steps=20)
    if kind == 'DIFGSM':  return torchattacks.DIFGSM(clf, eps=EPS, alpha=a, steps=20)
    if kind == 'TIFGSM':  return torchattacks.TIFGSM(clf, eps=EPS, alpha=a, steps=20)
    if kind == 'NIFGSM':  return torchattacks.NIFGSM(clf, eps=EPS, alpha=a, steps=20)
    if kind == 'APGD':    return torchattacks.APGD(clf, norm='Linf', eps=EPS, steps=20)
    if kind == 'AutoAttack': return torchattacks.AutoAttack(clf, norm='Linf', eps=EPS, n_classes=1000)
    if kind == 'CW':      return torchattacks.CW(clf, c=1.0, steps=100, lr=0.01)
    if kind == 'DeepFool':return torchattacks.DeepFool(clf, steps=50)
    if kind == 'Square':  return torchattacks.Square(clf, norm='Linf', eps=EPS, n_queries=500, n_restarts=1)
    raise ValueError(kind)

# ---------------- benign controls (AE #4): matched to the PGD perturbation of the same image
if A.kind.startswith('CTRL_'):
    ref = np.load(f'{OUTD}/adv/{A.clf}/PGD_e{A.eps}.npz'); rid = {int(i): j for j, i in enumerate(ref['image_id'])}
    elig = np.array([i for i in elig if int(i) in rid])

def gen(idx, bid, x, y):
    g = torch.Generator(device=x.device); g.manual_seed(image_seed(int(bid[0]), f'{A.kind}|{A.eps}|batch'))
    if not A.kind.startswith('CTRL_'):
        torch.manual_seed(image_seed(int(bid[0]), f'{A.kind}|{A.eps}'))
        return make_atk(A.kind)(x, y)
    d_pgd = to_tensor(ref['adv'][[rid[int(i)] for i in bid]], x.device) - x       # quantised PGD delta
    k = A.kind[5:]
    if k == 'sign':      # uniform random sign, every pixel at +-eps (L_inf and L2 matched to a saturated PGD)
        d = (torch.randint(0, 2, x.shape, generator=g, device=x.device).float() * 2 - 1) * EPS
    elif k == 'gauss':   # Gaussian with per-image L2 matched to the PGD delta
        n = torch.randn(x.shape, generator=g, device=x.device)
        d = n * (d_pgd.flatten(1).norm(dim=1) / n.flatten(1).norm(dim=1)).view(-1, 1, 1, 1)
    elif k == 'phase':   # keep |FFT| of the PGD delta per channel, randomise the phase, project to L_inf ball
        Fd = torch.fft.fft2(d_pgd); ph = torch.fft.fft2(torch.randn(x.shape, generator=g, device=x.device))
        d = torch.fft.ifft2(Fd.abs() * torch.exp(1j * torch.angle(ph))).real
        d = d * (d_pgd.flatten(1).norm(dim=1) / d.flatten(1).norm(dim=1).clamp_min(1e-12)).view(-1, 1, 1, 1)
        d = d.clamp(-EPS, EPS)
    elif k == 'shuffle': # permute PGD delta pixel positions (same value histogram, spatial structure destroyed)
        B, C, H, W = d_pgd.shape; perm = torch.stack([torch.randperm(H * W, generator=g, device=x.device) for _ in range(B)])
        d = torch.gather(d_pgd.reshape(B, C, H * W), 2, perm[:, None, :].expand(B, C, H * W)).reshape(B, C, H, W)
    elif k == 'jpeg75':
        from PIL import Image
        outs = []
        for u8 in to_uint8(x):
            b = io.BytesIO(); Image.fromarray(u8).save(b, format='JPEG', quality=75); outs.append(np.array(Image.open(b)))
        return to_tensor(np.stack(outs), x.device)
    elif k == 'blur':    # gaussian blur sigma=0.5 px
        r = torch.arange(-2, 3, device=x.device).float(); w = torch.exp(-r ** 2 / (2 * 0.5 ** 2)); w = w / w.sum()
        xb = F.conv2d(F.pad(x, (2, 2, 0, 0), mode='reflect'), w.view(1, 1, 1, 5).repeat(3, 1, 1, 1), groups=3)
        return F.conv2d(F.pad(xb, (0, 0, 2, 2), mode='reflect'), w.view(1, 1, 5, 1).repeat(3, 1, 1, 1), groups=3)
    else: raise ValueError(k)
    return (x + d).clamp(0, 1)

PART = 256
for pi, pid in parts(elig, PART):
    pf = f'{pdir}/part_{pi:04d}.npz'
    if os.path.exists(pf): continue
    acc = {}
    for b in range(0, len(pid), A.bs):
        bid = pid[b:b + A.bs]; x = to_tensor(X[bid], DEVICE); y = torch.from_numpy(Y[bid]).to(DEVICE)
        try: xa = gen(b, bid, x, y).detach()
        except Exception as e: log(f'  !! batch {bid[0]} err: {e}'); continue
        u8 = to_uint8(xa); xq = to_tensor(u8, DEVICE)
        r = clf_pass(clf, hook, xq); dd = (xq - x).flatten(1)
        r.update(image_id=bid, label=Y[bid], adv=u8, success=(r['pred'] != Y[bid]),
                 linf=dd.abs().max(1)[0].cpu().numpy(), l2=dd.norm(dim=1).cpu().numpy())
        for k, v in r.items(): acc.setdefault(k, []).append(v)
    if not acc: continue
    acc = {k: np.concatenate(v) for k, v in acc.items()}; atomic_savez(pf, **acc)
    log(f'  part {pi}: {min((pi + 1) * PART, len(elig))}/{len(elig)} success={acc["success"].mean():.3f} linf_max={acc["linf"].max() * 255:.2f}/255')
merge_parts(pdir, out)
d = np.load(out); log(f'saved {out}  n={len(d["image_id"])} success_rate={d["success"].mean():.4f}')
