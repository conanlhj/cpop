"""S7 — CIFAR-10 re-run (Table 8) under the R1 protocol: GT labels, class-stratified disjoint splits of the
10k test set (ref 200 | cal 300 | test 300 | eval 200 per class), uint8 quantisation before success,
benign controls (AE #4), all classifier-side scores and probe scores stored per image_id.
Probe: google/ddpm-cifar10-32 (fp32, t=1, sigma=0.005, K=8 — as submitted).
out: OUTD/cifar/{splits.csv, probe_clean.npz, <clf>/clean.npz, <clf>/<kind>.npz}
usage: python s7_cifar.py --clf resnet20          (first call also scores clean images with the probe)"""
import os, io, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import common as C
from common import log, atomic_savez, image_seed, to_uint8, SQUEEZERS
import sys; sys.path.append(C.CODE)
import torchattacks, torchvision, torchvision.transforms as T
from diffusers import UNet2DModel
from utils.patches import apply_cw_patch
apply_cw_patch()

ap = argparse.ArgumentParser(); ap.add_argument('--clf', required=True); ap.add_argument('--bs', type=int, default=64); A = ap.parse_args()
dev = C.DEVICE; EPS = 8 / 255; OD = f'{C.OUTD}/cifar'; os.makedirs(f'{OD}/{A.clf}', exist_ok=True)
ds = torchvision.datasets.CIFAR10(root=os.environ.get('CIFAR_ROOT', f'{C.EXT}/cifar10'), train=False, download=False)
XU = np.asarray(ds.data, dtype=np.uint8); YL = np.asarray(ds.targets, dtype=np.int64)          # (10000,32,32,3)
SP = f'{OD}/splits.csv'
if not os.path.exists(SP):
    rng = np.random.default_rng(20261020); rows = []
    for k in range(10):
        ids = rng.permutation(np.where(YL == k)[0]); o = 0
        for nm, n in [('ref', 200), ('cal', 300), ('test', 300), ('eval', 200)]: rows += [(int(i), k, nm) for i in ids[o:o + n]]; o += n
    pd.DataFrame(rows, columns=['image_id', 'label', 'split']).sort_values('image_id').to_csv(SP, index=False)
spl = pd.read_csv(SP); sid = lambda s: spl[spl.split == s].image_id.values[: (16 if C.SMOKE else None)]
t = lambda u8: torch.from_numpy(np.ascontiguousarray(u8)).to(dev).permute(0, 3, 1, 2).float() / 255

unet = UNet2DModel.from_pretrained('google/ddpm-cifar10-32').to(dev).eval()
for p in unet.parameters(): p.requires_grad_(False)
def probe(u8, ids, tag):
    out = {}
    for i in range(0, len(u8), 256):
        r = C.cpop_all(unet, t(u8[i:i + 256]), ids[i:i + 256], variant=f'cifar|{tag}')
        for k, v in r.items(): out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}
PC = f'{OD}/probe_clean.npz'
if not os.path.exists(PC):
    ids = np.sort(np.concatenate([sid('cal'), sid('test'), sid('eval')]))
    atomic_savez(PC, image_id=ids, **probe(XU[ids], ids, 'main')); log('saved', PC)

HUB = {'resnet20': 'cifar10_resnet20', 'resnet56': 'cifar10_resnet56', 'vgg16_bn': 'cifar10_vgg16_bn',
       'mobilenetv2': 'cifar10_mobilenetv2_x1_0', 'repvgg_a2': 'cifar10_repvgg_a2', 'shufflenetv2': 'cifar10_shufflenetv2_x2_0'}
class Clf(nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
        s.register_buffer('mu', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)); s.register_buffer('sd', torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))
    def forward(s, x): return s.m((x - s.mu) / s.sd)
m = torch.hub.load('chenyaofo/pytorch-cifar-models', HUB[A.clf], pretrained=True).eval()
for p in m.parameters(): p.requires_grad_(False)
clf = Clf(m).to(dev).eval(); hook = C.FeatHook(clf)
def cpass(u8):
    out = {}
    for i in range(0, len(u8), 256):
        r = C.clf_pass(clf, hook, t(u8[i:i + 256]))
        for k, v in r.items(): out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}
CL = f'{OD}/{A.clf}/clean.npz'
if not os.path.exists(CL):
    ids = np.sort(np.concatenate([sid('ref'), sid('cal'), sid('test'), sid('eval')]))
    atomic_savez(CL, image_id=ids, label=YL[ids], **cpass(XU[ids])); log('saved', CL)
c = np.load(CL); ev = set(sid('eval').tolist()); mk = np.array([i in ev for i in c['image_id']])
elig = c['image_id'][mk][c['pred'][mk] == c['label'][mk]]
log(f'{A.clf}: clean acc(eval) {np.mean(c["pred"][mk] == c["label"][mk]):.4f}, eligible {len(elig)}')

def make_atk(kind):
    a = EPS / 4
    return {'FGSM': lambda: torchattacks.FGSM(clf, eps=EPS), 'PGD': lambda: torchattacks.PGD(clf, eps=EPS, alpha=a, steps=20),
            'BIM': lambda: torchattacks.BIM(clf, eps=EPS, alpha=a, steps=20), 'MIFGSM': lambda: torchattacks.MIFGSM(clf, eps=EPS, alpha=a, steps=20),
            'APGD': lambda: torchattacks.APGD(clf, norm='Linf', eps=EPS, steps=20),
            'AutoAttack': lambda: torchattacks.AutoAttack(clf, norm='Linf', eps=EPS, version='standard', n_classes=10),
            'CW': lambda: torchattacks.CW(clf, c=1.0, steps=100, lr=0.01), 'DeepFool': lambda: torchattacks.DeepFool(clf, steps=50),
            'Square': lambda: torchattacks.Square(clf, norm='Linf', eps=EPS, n_queries=1000, n_restarts=1)}[kind]()
ATTACKS = ['FGSM', 'PGD', 'BIM', 'MIFGSM', 'APGD', 'AutoAttack', 'CW', 'DeepFool', 'Square']
CTRLS = ['CTRL_sign', 'CTRL_gauss', 'CTRL_phase', 'CTRL_shuffle', 'CTRL_jpeg75', 'CTRL_blur']

def control(kind, x, d_pgd, ids):
    g = torch.Generator(device=dev); g.manual_seed(image_seed(int(ids[0]), f'cifar|{kind}')); k = kind[5:]
    if k == 'sign': d = (torch.randint(0, 2, x.shape, generator=g, device=dev).float() * 2 - 1) * EPS
    elif k == 'gauss':
        n = torch.randn(x.shape, generator=g, device=dev); d = n * (d_pgd.flatten(1).norm(dim=1) / n.flatten(1).norm(dim=1)).view(-1, 1, 1, 1)
    elif k == 'phase':
        ph = torch.fft.fft2(torch.randn(x.shape, generator=g, device=dev)); d = torch.fft.ifft2(torch.fft.fft2(d_pgd).abs() * torch.exp(1j * torch.angle(ph))).real
        d = (d * (d_pgd.flatten(1).norm(dim=1) / d.flatten(1).norm(dim=1).clamp_min(1e-12)).view(-1, 1, 1, 1)).clamp(-EPS, EPS)
    elif k == 'shuffle':
        B, Cc, H, W = d_pgd.shape; perm = torch.stack([torch.randperm(H * W, generator=g, device=dev) for _ in range(B)])
        d = torch.gather(d_pgd.reshape(B, Cc, H * W), 2, perm[:, None, :].expand(B, Cc, H * W)).reshape(B, Cc, H, W)
    elif k == 'jpeg75':
        from PIL import Image; o = []
        for u in to_uint8(x):
            b = io.BytesIO(); Image.fromarray(u).save(b, format='JPEG', quality=75); o.append(np.array(Image.open(b)))
        return t(np.stack(o))
    elif k == 'blur':
        r = torch.arange(-2, 3, device=dev).float(); w = torch.exp(-r ** 2 / 0.5); w = w / w.sum()
        xb = F.conv2d(F.pad(x, (2, 2, 0, 0), mode='reflect'), w.view(1, 1, 1, 5).repeat(3, 1, 1, 1), groups=3)
        return F.conv2d(F.pad(xb, (0, 0, 2, 2), mode='reflect'), w.view(1, 1, 5, 1).repeat(3, 1, 1, 1), groups=3)
    return (x + d).clamp(0, 1)

for kind in ATTACKS + CTRLS:
    o = f'{OD}/{A.clf}/{kind}.npz'
    if os.path.exists(o): continue
    if kind.startswith('CTRL_'):
        ref = np.load(f'{OD}/{A.clf}/PGD.npz'); ids = ref['image_id']; dp = t(ref['adv']) - t(XU[ids])
    else: ids = elig[: (16 if C.SMOKE else (600 if kind in ['AutoAttack', 'CW', 'DeepFool', 'Square'] else None))]
    advs = []
    for i in range(0, len(ids), A.bs):
        b = ids[i:i + A.bs]; x = t(XU[b]); y = torch.from_numpy(YL[b]).to(dev)
        if kind.startswith('CTRL_'): xa = control(kind, x, dp[i:i + A.bs], b)
        else: torch.manual_seed(image_seed(int(b[0]), f'cifar|{kind}')); xa = make_atk(kind)(x, y)
        advs.append(to_uint8(xa.detach()))
    adv = np.concatenate(advs); r = cpass(adv); dd = (t(adv) - t(XU[ids])).flatten(1)
    atomic_savez(o, image_id=ids, label=YL[ids], adv=adv, success=r['pred'] != YL[ids], linf=dd.abs().max(1)[0].cpu().numpy(),
                 l2=dd.norm(dim=1).cpu().numpy(), **{f'probe_{k}': v for k, v in probe(adv, ids, 'main').items()}, **r)
    log(f'{A.clf}/{kind}: n={len(ids)} success={np.mean(r["pred"] != YL[ids]):.3f}')
log('done', A.clf)
