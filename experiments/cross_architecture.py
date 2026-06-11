"""P1 Exp A — cross-architecture detector comparison, publication-grade.

8 classifiers x PGD-Linf at eps in {2,4,8}/255 x N=300, scored by:
  - CPOP        : fixed ADM diffusion probe (never sees the classifier)
  - FS_common   : FeatureSqueeze, standard fixed params (3-bit depth + 3x3 avgpool), max-L1.
  - FS_tuned    : FeatureSqueeze with the best single squeezer chosen on a VALIDATION split
                  per architecture, evaluated on a disjoint TEST split (fair "tuned" baseline).
  - LID         : PROPER LID (Ma 2018 MLE estimator) on the *target classifier's* penultimate
                  features (input to its final Linear) -- Exp C folded in.
  - LID_ext     : same estimator on a FIXED ResNet50 feature space (external-feature baseline).
Per-sample scores (incl. per-squeezer FS) -> CSV. AUROC + bootstrap 95% CI. Stability summary.

Usage:  python exp_a_cross_arch_full.py [--smoke]
"""
import sys, os, glob, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models as tvm
from diffusers import UNet2DModel
import torchattacks, pandas as pd, numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.patches import apply_cw_patch
apply_cw_patch()

ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); A = ap.parse_args()
device = torch.device('cuda')
T_STEP, SIGMA, K, BATCH, LID_K = 1, 0.005, 8, 4, 20
N = 8 if A.smoke else 300
EPS_LIST = [4/255] if A.smoke else [2/255, 4/255, 8/255]
ADM_PATH = '/workspace/data/adm_imagenet_local'
DATA_DIR = '/workspace/data/imagenet_val/val_256x256'
OUT = '/workspace/outputs/tables/p1_expA_cross_arch.csv'
SUM = '/workspace/outputs/tables/p1_expA_summary.csv'
MEAN = torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
STD  = torch.tensor([0.229,0.225,0.224],device=device).view(1,3,1,1)

unet = UNet2DModel.from_pretrained(ADM_PATH, torch_dtype=torch.float16).to(device).eval()
for p in unet.parameters(): p.requires_grad_(False)

class Clf(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
        return self.m((x - MEAN) / STD)

ARCHS = {
    'resnet50':           (tvm.resnet50,           tvm.ResNet50_Weights.IMAGENET1K_V2),
    'vgg16':              (tvm.vgg16,              tvm.VGG16_Weights.IMAGENET1K_V1),
    'densenet121':        (tvm.densenet121,        tvm.DenseNet121_Weights.IMAGENET1K_V1),
    'mobilenet_v3_large': (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    'convnext_tiny':      (tvm.convnext_tiny,      tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    'vit_b_16':           (tvm.vit_b_16,           tvm.ViT_B_16_Weights.IMAGENET1K_V1),
    'swin_t':             (tvm.swin_t,             tvm.Swin_T_Weights.IMAGENET1K_V1),
    'maxvit_t':           (tvm.maxvit_t,           tvm.MaxVit_T_Weights.IMAGENET1K_V1),
}
ARCH_TYPE = {'resnet50':'CNN','vgg16':'CNN','densenet121':'CNN','mobilenet_v3_large':'CNN',
             'convnext_tiny':'CNN','vit_b_16':'Transformer','swin_t':'Transformer','maxvit_t':'Transformer'}
if A.smoke: ARCHS = {k: ARCHS[k] for k in ['resnet50','vit_b_16']}

# ---- FeatureSqueeze squeezers (for common + tuned) ----
def bitdepth(x, b): L = 2**b - 1; return torch.round(x*L)/L
def medianf(x, k):
    p = k//2; xp = F.pad(x,(p,p,p,p),mode='reflect'); pa = xp.unfold(2,k,1).unfold(3,k,1)
    return pa.contiguous().view(*pa.shape[:4], k*k).median(-1)[0]
SQUEEZERS = {'bit1':lambda x:bitdepth(x,1), 'bit2':lambda x:bitdepth(x,2),
             'bit3':lambda x:bitdepth(x,3), 'bit5':lambda x:bitdepth(x,5),
             'median3':lambda x:medianf(x,3), 'median5':lambda x:medianf(x,5),
             'avgpool3':lambda x:F.avg_pool2d(x,3,1,1)}
FS_COLS = ['fs_'+s for s in SQUEEZERS]

@torch.no_grad()
def fs_all(imgs, clf):
    p = F.softmax(clf(imgs), 1); out = {}
    for nm, sq in SQUEEZERS.items():
        q = F.softmax(clf(sq(imgs)), 1); out[nm] = (p-q).abs().sum(1).cpu().numpy()
    common = np.maximum(out['bit3'], out['avgpool3'])   # standard FS (matches prior baseline)
    return common, out

# ---- feature hook: input to final Linear = penultimate features (uniform across archs) ----
class FeatHook:
    def __init__(self, clf):
        lins = [m for m in clf.modules() if isinstance(m, nn.Linear)]
        self.feat = None; self.h = lins[-1].register_forward_pre_hook(self._hook)
    def _hook(self, mod, inp): self.feat = inp[0].detach().flatten(1)
    def remove(self): self.h.remove()

_rn = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
for p in _rn.parameters(): p.requires_grad_(False)
def rn_feats(x):
    x = F.interpolate(x, 224, mode='bilinear', align_corners=False); x = (x-MEAN)/STD
    f = _rn.conv1(x); f = _rn.bn1(f); f = _rn.relu(f); f = _rn.maxpool(f)
    f = _rn.layer1(f); f = _rn.layer2(f); f = _rn.layer3(f); f = _rn.layer4(f)
    return _rn.avgpool(f).view(x.shape[0], -1)

@torch.no_grad()
def cpop_score(images):
    B = images.shape[0]; x = (images*2-1).to(torch.float16)
    t = torch.full((B,), T_STEP, device=device, dtype=torch.long)
    u = unet(x, t).sample[:, :3].view(B, -1).float(); acc = torch.zeros(B, device=device)
    for _ in range(K):
        v = unet(x + torch.randn_like(x)*SIGMA, t).sample[:, :3].view(B, -1).float()
        acc += 1.0 - F.cosine_similarity(u, v, dim=1)
    return -(acc/K).cpu().numpy()                       # higher -> adv (directional freezing)

@torch.no_grad()
def lid_mle(q, r, k=LID_K):
    q = q.float(); r = r.float(); d = torch.cdist(q, r); rr, _ = d.topk(k, dim=1, largest=False)
    rr = rr.clamp_min(1e-12); rk = rr[:, -1:].clamp_min(1e-12)
    return (-1.0/(torch.log(rr/rk).mean(1))).cpu().numpy()

tf = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])
class DS(torch.utils.data.Dataset):
    def __init__(s, p): s.p = p
    def __len__(s): return len(s.p)
    def __getitem__(s, i): return tf(Image.open(s.p[i]).convert('RGB'))
paths = sorted(glob.glob(os.path.join(DATA_DIR, '*.png')))
if not paths: paths = sorted(glob.glob(os.path.join(DATA_DIR,'*.JPEG')))+sorted(glob.glob(os.path.join(DATA_DIR,'*.jpg')))
print(f'{len(paths)} images | N={N} eps={[round(e*255) for e in EPS_LIST]}/255 archs={list(ARCHS)}', flush=True)

ref_imgs = torch.cat([b for b in torch.utils.data.DataLoader(DS(paths[600:1100]), batch_size=BATCH)])
ref_ext = torch.cat([rn_feats(ref_imgs[i:i+16].to(device)) for i in range(0,len(ref_imgs),16)])

def collect(name, clf, hook):
    rows = []; ref_feat = []
    for i in range(0, len(ref_imgs), 16):
        clf(ref_imgs[i:i+16].to(device)); ref_feat.append(hook.feat)
    ref_feat = torch.cat(ref_feat)
    for eps in EPS_LIST:
        atk = torchattacks.PGD(clf, eps=eps, alpha=eps/4, steps=20)
        loader = torch.utils.data.DataLoader(DS(paths[:600]), batch_size=BATCH)
        n = 0; pbar = tqdm(total=N, desc=f'{name} eps{round(eps*255)}')
        for imgs in loader:
            if n >= N: break
            imgs = imgs.to(device)
            with torch.no_grad(): y = clf(imgs).argmax(1)
            adv = atk(imgs, y)
            with torch.no_grad(): ok = clf(adv).argmax(1) != y
            if not ok.any(): continue
            ci, ai = imgs[ok], adv[ok]; take = min(ci.shape[0], N-n); ci, ai = ci[:take], ai[:take]
            cc, ac = cpop_score(ci), cpop_score(ai)
            with torch.no_grad(): clf(ci); fc = hook.feat
            with torch.no_grad(): clf(ai); fa = hook.feat
            clid, alid = lid_mle(fc, ref_feat), lid_mle(fa, ref_feat)
            clide, alide = lid_mle(rn_feats(ci), ref_ext), lid_mle(rn_feats(ai), ref_ext)
            cfs_c, cfs_all = fs_all(ci, clf); afs_c, afs_all = fs_all(ai, clf)
            e = round(eps*255)
            for j in range(take):
                base0 = {'classifier':name,'arch_type':ARCH_TYPE[name],'eps255':e,'sid':n+j,'label':0,
                         'cpop':float(cc[j]),'fs':float(cfs_c[j]),'lid':float(clid[j]),'lid_ext':float(clide[j])}
                base1 = {'classifier':name,'arch_type':ARCH_TYPE[name],'eps255':e,'sid':n+j,'label':1,
                         'cpop':float(ac[j]),'fs':float(afs_c[j]),'lid':float(alid[j]),'lid_ext':float(alide[j])}
                for s in SQUEEZERS:
                    base0['fs_'+s] = float(cfs_all[s][j]); base1['fs_'+s] = float(afs_all[s][j])
                rows.append(base0); rows.append(base1)
            n += take; pbar.update(take)
        pbar.close()
    return rows

all_rows = []
for name,(ctor,w) in ARCHS.items():
    m = ctor(weights=w).to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    clf = Clf(m).to(device).eval(); hook = FeatHook(clf)
    all_rows += collect(name, clf, hook)
    hook.remove(); del m, clf; torch.cuda.empty_cache()

df = pd.DataFrame(all_rows); os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False); print('saved', OUT, len(df), flush=True)

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if np.isnan(s).any(): m = ~np.isnan(s); y, s = y[m], s[m]
    if len(np.unique(y)) < 2: return np.nan
    return roc_auc_score(y, s)
def auroc_ci(y, s, nb=1000):
    base = auroc(y, s)
    if np.isnan(base): return np.nan, np.nan, np.nan
    y = np.asarray(y); s = np.asarray(s); rng = np.random.default_rng(0); boot = []; idx = np.arange(len(y))
    for _ in range(nb):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2: continue
        boot.append(roc_auc_score(y[b], s[b]))
    return round(base,3), round(float(np.percentile(boot,2.5)),3), round(float(np.percentile(boot,97.5)),3)

def fs_tuned_test(sub):
    """Pick best squeezer on a 50/50 validation split, report AUROC on disjoint test split."""
    sids = np.array(sorted(sub.sid.unique())); rng = np.random.default_rng(0); rng.shuffle(sids)
    val_ids = set(sids[:len(sids)//2].tolist());
    val = sub[sub.sid.isin(val_ids)]; test = sub[~sub.sid.isin(val_ids)]
    best, best_au = None, -1
    for c in FS_COLS:
        au = auroc(val.label.values, val[c].values)
        if not np.isnan(au) and au > best_au: best_au, best = au, c
    if best is None: return np.nan, np.nan, np.nan, None
    a, lo, hi = auroc_ci(test.label.values, test[best].values)
    return a, lo, hi, best

DETS = ['cpop','fs','lid','lid_ext']
summ = []
print('\n=== AUROC [95% CI] per arch x eps ===', flush=True)
for name in ARCHS:
    for e in sorted(df.eps255.unique()):
        s = df[(df.classifier==name)&(df.eps255==e)]
        row = {'arch':name,'type':ARCH_TYPE[name],'eps255':int(e),'n_adv':int((s.label==1).sum())}
        for d in DETS:
            a,lo,hi = auroc_ci(s.label.values, s[d].values); row[d]=a; row[d+'_lo']=lo; row[d+'_hi']=hi
        ta,tlo,thi,bsq = fs_tuned_test(s); row['fs_tuned']=ta; row['fs_tuned_lo']=tlo; row['fs_tuned_hi']=thi; row['fs_tuned_sq']=bsq
        summ.append(row)
        print(f'{name:20s} eps{int(e)}: CPOP {row["cpop"]}[{row["cpop_lo"]},{row["cpop_hi"]}]  '
              f'FScom {row["fs"]}  FStun {row["fs_tuned"]}({bsq})  LID {row["lid"]}  LIDext {row["lid_ext"]}', flush=True)
sdf = pd.DataFrame(summ); sdf.to_csv(SUM, index=False); print('saved', SUM, flush=True)

print('\n=== STABILITY across archs (eps=4/255) — the headline ===', flush=True)
m4 = sdf[sdf.eps255==4]
for d in ['cpop','fs','fs_tuned','lid','lid_ext']:
    v = m4[d].dropna().values
    print(f'  {d.upper():10s} mean={v.mean():.3f} std={v.std():.3f} min={v.min():.3f} max={v.max():.3f}', flush=True)
print('\n=== CNN vs Transformer mean (eps=4/255) ===', flush=True)
for t in ['CNN','Transformer']:
    sub = m4[m4.type==t]
    print(f'  {t:12s} '+'  '.join(f'{d.upper()}={sub[d].mean():.3f}' for d in ['cpop','fs','fs_tuned','lid']), flush=True)
