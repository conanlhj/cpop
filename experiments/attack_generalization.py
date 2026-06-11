"""P1 Exp B2 — broad attack generalization + extra baselines.

4 classifiers x many attacks at eps=4/255 (L-inf) plus boundary attacks (CW/DeepFool L2, Square black-box).
Detectors: CPOP (fixed ADM, classifier-agnostic), and classifier-coupled baselines
           FS_common, LID, LID_ext (fixed rn50), Mahalanobis (single-Gaussian), KD (kernel density).
Goal: show CPOP generalizes across the L-inf gradient family (our threat model), and add modern baselines.
Usage:  python exp_b2_attacks.py [--smoke]
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
T_STEP, SIGMA, K, BATCH, LID_K, EPS = 1, 0.005, 8, 4, 20, 4/255
N_FAST = 6 if A.smoke else 200
N_SLOW = 6 if A.smoke else 100     # CW/DeepFool/AutoAttack/Square are expensive
ADM_PATH='/workspace/data/adm_imagenet_local'; DATA_DIR='/workspace/data/imagenet_val/val_256x256'
OUT='/workspace/outputs/tables/p1_expB2_attacks.csv'; SUM='/workspace/outputs/tables/p1_expB2_summary.csv'
MEAN=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
STD =torch.tensor([0.229,0.225,0.224],device=device).view(1,3,1,1)

unet = UNet2DModel.from_pretrained(ADM_PATH, torch_dtype=torch.float16).to(device).eval()
for p in unet.parameters(): p.requires_grad_(False)

class Clf(nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,x):
        x=F.interpolate(x,size=224,mode='bilinear',align_corners=False); return s.m((x-MEAN)/STD)

ARCHS = {'resnet50':(tvm.resnet50,tvm.ResNet50_Weights.IMAGENET1K_V2),
         'densenet121':(tvm.densenet121,tvm.DenseNet121_Weights.IMAGENET1K_V1),
         'vit_b_16':(tvm.vit_b_16,tvm.ViT_B_16_Weights.IMAGENET1K_V1),
         'swin_t':(tvm.swin_t,tvm.Swin_T_Weights.IMAGENET1K_V1)}
TYPE={'resnet50':'CNN','densenet121':'CNN','vit_b_16':'Transformer','swin_t':'Transformer'}
if A.smoke: ARCHS={k:ARCHS[k] for k in ['resnet50','vit_b_16']}

class FeatHook:
    def __init__(s,clf):
        lins=[m for m in clf.modules() if isinstance(m,nn.Linear)]; s.feat=None
        s.h=lins[-1].register_forward_pre_hook(s._h)
    def _h(s,mod,inp): s.feat=inp[0].detach().flatten(1)
    def remove(s): s.h.remove()

_rn=tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
for p in _rn.parameters(): p.requires_grad_(False)
def rn_feats(x):
    x=F.interpolate(x,224,mode='bilinear',align_corners=False); x=(x-MEAN)/STD
    f=_rn.conv1(x);f=_rn.bn1(f);f=_rn.relu(f);f=_rn.maxpool(f)
    f=_rn.layer1(f);f=_rn.layer2(f);f=_rn.layer3(f);f=_rn.layer4(f)
    return _rn.avgpool(f).view(x.shape[0],-1)

@torch.no_grad()
def cpop_score(images):
    B=images.shape[0]; x=(images*2-1).to(torch.float16)
    t=torch.full((B,),T_STEP,device=device,dtype=torch.long)
    u=unet(x,t).sample[:,:3].view(B,-1).float(); acc=torch.zeros(B,device=device)
    for _ in range(K):
        v=unet(x+torch.randn_like(x)*SIGMA,t).sample[:,:3].view(B,-1).float()
        acc+=1.0-F.cosine_similarity(u,v,dim=1)
    return -(acc/K).cpu().numpy()

@torch.no_grad()
def lid_mle(q,r,k=LID_K):
    q=q.float();r=r.float(); d=torch.cdist(q,r); rr,_=d.topk(k,dim=1,largest=False)
    rr=rr.clamp_min(1e-12); rk=rr[:,-1:].clamp_min(1e-12)
    return (-1.0/(torch.log(rr/rk).mean(1))).cpu().numpy()

@torch.no_grad()
def fs_common(imgs,clf):
    p=F.softmax(clf(imgs),1)
    pb=F.softmax(clf(torch.round(imgs*7)/7),1)
    ps=F.softmax(clf(F.avg_pool2d(imgs,3,1,1)),1)
    return torch.maximum((p-pb).abs().sum(1),(p-ps).abs().sum(1)).cpu().numpy()

class Maha:   # single-Gaussian Mahalanobis on penultimate features (shrinkage covariance)
    def __init__(s,ref):
        ref=ref.float(); s.mu=ref.mean(0,keepdim=True)
        d=ref-s.mu; cov=(d.t()@d)/ref.shape[0]
        cov+= (cov.diagonal().mean()*1e-2)*torch.eye(cov.shape[0],device=cov.device)
        s.prec=torch.linalg.pinv(cov)
    @torch.no_grad()
    def score(s,q):
        q=q.float(); d=q-s.mu; return (d@s.prec*d).sum(1).clamp_min(0).sqrt().cpu().numpy()  # higher=adv

class KD:     # kernel density in feature space (Feinman 2017 style); higher score = adv (lower density)
    def __init__(s,ref):
        s.ref=ref.float(); dd=torch.cdist(s.ref[:200],s.ref[:200]); s.h2=2*(dd.median().item()**2+1e-9)
    @torch.no_grad()
    def score(s,q):
        q=q.float(); d2=torch.cdist(q,s.ref)**2
        logkd=torch.logsumexp(-d2/s.h2,dim=1)-np.log(s.ref.shape[0])
        return (-logkd).cpu().numpy()

tf=T.Compose([T.Resize(256),T.CenterCrop(256),T.ToTensor()])
class DS(torch.utils.data.Dataset):
    def __init__(s,p):s.p=p
    def __len__(s):return len(s.p)
    def __getitem__(s,i):return tf(Image.open(s.p[i]).convert('RGB'))
paths=sorted(glob.glob(os.path.join(DATA_DIR,'*.png'))) or sorted(glob.glob(os.path.join(DATA_DIR,'*.JPEG')))
print(f'{len(paths)} images | archs={list(ARCHS)}',flush=True)
ref_imgs=torch.cat([b for b in torch.utils.data.DataLoader(DS(paths[600:1100]),batch_size=BATCH)])
ref_ext=torch.cat([rn_feats(ref_imgs[i:i+16].to(device)) for i in range(0,len(ref_imgs),16)])

# attack set: L-inf gradient family (in-scope) + boundary/black-box (reported for completeness)
def make_atk(kind, clf):
    if kind=='FGSM':    return torchattacks.FGSM(clf, eps=EPS)
    if kind=='BIM':     return torchattacks.BIM(clf, eps=EPS, alpha=EPS/4, steps=20)
    if kind=='RFGSM':   return torchattacks.RFGSM(clf, eps=EPS, alpha=EPS/4, steps=20)
    if kind=='PGD':     return torchattacks.PGD(clf, eps=EPS, alpha=EPS/4, steps=20)
    if kind=='MIFGSM':  return torchattacks.MIFGSM(clf, eps=EPS, alpha=EPS/4, steps=20)
    if kind=='DIFGSM':  return torchattacks.DIFGSM(clf, eps=EPS, alpha=EPS/4, steps=20)
    if kind=='TIFGSM':  return torchattacks.TIFGSM(clf, eps=EPS, alpha=EPS/4, steps=20)
    if kind=='NIFGSM':  return torchattacks.NIFGSM(clf, eps=EPS, alpha=EPS/4, steps=20)
    if kind=='APGD':    return torchattacks.APGD(clf, norm='Linf', eps=EPS, steps=20)
    if kind=='AutoAttack': return torchattacks.AutoAttack(clf, norm='Linf', eps=EPS, n_classes=1000)
    if kind=='CW':      return torchattacks.CW(clf, c=1.0, steps=100, lr=0.01)
    if kind=='DeepFool':return torchattacks.DeepFool(clf, steps=50)
    if kind=='Square':  return torchattacks.Square(clf, norm='Linf', eps=EPS, n_queries=500, n_restarts=1)
ATTACKS=['FGSM','BIM','RFGSM','PGD','MIFGSM','DIFGSM','TIFGSM','NIFGSM','APGD','AutoAttack','CW','Square']
SLOW={'AutoAttack','CW','Square'}

def collect(name, clf, hook):
    rows=[]; ref_feat=[]
    for i in range(0,len(ref_imgs),16):
        clf(ref_imgs[i:i+16].to(device)); ref_feat.append(hook.feat)
    ref_feat=torch.cat(ref_feat); maha=Maha(ref_feat); kd=KD(ref_feat)
    for kind in ATTACKS:
        N=N_SLOW if kind in SLOW else N_FAST
        atk=make_atk(kind,clf); loader=torch.utils.data.DataLoader(DS(paths[:600]),batch_size=BATCH)
        n=0; pbar=tqdm(total=N,desc=f'{name}/{kind}')
        for imgs in loader:
            if n>=N: break
            imgs=imgs.to(device)
            with torch.no_grad(): y=clf(imgs).argmax(1)
            try: adv=atk(imgs,y)
            except Exception as e: print(f'  !! {name}/{kind} batch err: {e}',flush=True); continue
            with torch.no_grad(): ok=clf(adv).argmax(1)!=y
            if not ok.any(): continue
            ci,ai=imgs[ok],adv[ok]; take=min(ci.shape[0],N-n); ci,ai=ci[:take],ai[:take]
            cc,ac=cpop_score(ci),cpop_score(ai)
            with torch.no_grad(): clf(ci); fc=hook.feat
            with torch.no_grad(): clf(ai); fa=hook.feat
            clid,alid=lid_mle(fc,ref_feat),lid_mle(fa,ref_feat)
            clide,alide=lid_mle(rn_feats(ci),ref_ext),lid_mle(rn_feats(ai),ref_ext)
            cf,af=fs_common(ci,clf),fs_common(ai,clf)
            cm,am=maha.score(fc),maha.score(fa); ck,ak=kd.score(fc),kd.score(fa)
            for j in range(take):
                rows.append({'classifier':name,'arch_type':TYPE[name],'attack':kind,'label':0,
                             'cpop':float(cc[j]),'fs':float(cf[j]),'lid':float(clid[j]),'lid_ext':float(clide[j]),'maha':float(cm[j]),'kd':float(ck[j])})
                rows.append({'classifier':name,'arch_type':TYPE[name],'attack':kind,'label':1,
                             'cpop':float(ac[j]),'fs':float(af[j]),'lid':float(alid[j]),'lid_ext':float(alide[j]),'maha':float(am[j]),'kd':float(ak[j])})
            n+=take; pbar.update(take)
        pbar.close()
    return rows

all_rows=[]
for name,(ctor,w) in ARCHS.items():
    m=ctor(weights=w).to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    clf=Clf(m).to(device).eval(); hook=FeatHook(clf)
    all_rows+=collect(name,clf,hook); hook.remove(); del m,clf; torch.cuda.empty_cache()

df=pd.DataFrame(all_rows); os.makedirs(os.path.dirname(OUT),exist_ok=True)
df.to_csv(OUT,index=False); print('saved',OUT,len(df),flush=True)

def auroc(y,s):
    y=np.asarray(y);s=np.asarray(s)
    if np.isnan(s).any(): m=~np.isnan(s); y,s=y[m],s[m]
    if len(np.unique(y))<2: return np.nan
    return round(roc_auc_score(y,s),3)

DETS=['cpop','fs','lid','lid_ext','maha','kd']; summ=[]
print('\n=== AUROC per arch x attack ===',flush=True)
for name in ARCHS:
    for kind in ATTACKS:
        s=df[(df.classifier==name)&(df.attack==kind)]
        if len(s)==0: continue
        row={'arch':name,'type':TYPE[name],'attack':kind,'n_adv':int((s.label==1).sum())}
        for d in DETS: row[d]=auroc(s.label.values,s[d].values)
        summ.append(row)
        print(f'{name:14s} {kind:10s}: '+'  '.join(f'{d.upper()} {row[d]}' for d in DETS),flush=True)
sdf=pd.DataFrame(summ); sdf.to_csv(SUM,index=False); print('saved',SUM,flush=True)
print('\n=== per-attack mean AUROC across archs ===',flush=True)
for kind in ATTACKS:
    sub=sdf[sdf.attack==kind]
    if len(sub)==0: continue
    print(f'  {kind:10s} '+'  '.join(f'{d.upper()}={sub[d].mean():.3f}' for d in DETS),flush=True)
