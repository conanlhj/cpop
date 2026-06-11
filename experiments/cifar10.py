"""P1 Exp (2nd dataset) — CIFAR-10 cross-architecture detection + operating-point transfer.

Diffusion probe: google/ddpm-cifar10-32 (fixed, classifier-agnostic).
Classifiers (chenyaofo/pytorch-cifar-models): resnet20, resnet56, vgg16_bn, mobilenetv2, repvgg_a2, shufflenetv2.
Attacks: FGSM/PGD/BIM/MIFGSM/APGD (L-inf, eps=8/255) + AutoAttack/CW/DeepFool/Square (boundary/bb).
Detectors: CPOP (agnostic), FS_common, LID, Mahalanobis, KD (classifier-coupled).
Outputs per-sample scores so cross-arch AUROC, FPR-matched TPR, and threshold transfer can be computed.
Usage: python exp_cifar_crossarch.py [--smoke]
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from diffusers import UNet2DModel
import torchattacks, pandas as pd, numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.patches import apply_cw_patch
apply_cw_patch()

ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); A=ap.parse_args()
device=torch.device('cuda')
T_STEP,SIGMA,K,LID_K,EPS = 1,0.005,8,20,8/255
N = 6 if A.smoke else 200
OUT='/workspace/outputs/tables/p1_cifar_crossarch.csv'; SUM='/workspace/outputs/tables/p1_cifar_summary.csv'
CMEAN=torch.tensor([0.4914,0.4822,0.4465],device=device).view(1,3,1,1)
CSTD =torch.tensor([0.2470,0.2435,0.2616],device=device).view(1,3,1,1)

unet=UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device).eval()
for p in unet.parameters(): p.requires_grad_(False)

class Clf(nn.Module):           # input in [0,1], 32x32; chenyaofo models want CIFAR-normalized
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,x): return s.m((x-CMEAN)/CSTD)

ARCH_IDS={'resnet20':'cifar10_resnet20','resnet56':'cifar10_resnet56','vgg16_bn':'cifar10_vgg16_bn',
          'mobilenetv2':'cifar10_mobilenetv2_x1_0','repvgg_a2':'cifar10_repvgg_a2','shufflenetv2':'cifar10_shufflenetv2_x2_0'}
TYPE={'resnet20':'CNN','resnet56':'CNN','vgg16_bn':'CNN','mobilenetv2':'CNN','repvgg_a2':'CNN','shufflenetv2':'CNN'}
if A.smoke: ARCH_IDS={k:ARCH_IDS[k] for k in ['resnet20','vgg16_bn']}

class FeatHook:
    def __init__(s,clf):
        lins=[m for m in clf.modules() if isinstance(m,nn.Linear)]; s.feat=None
        s.h=lins[-1].register_forward_pre_hook(s._h)
    def _h(s,mod,inp): s.feat=inp[0].detach().flatten(1)
    def remove(s): s.h.remove()

@torch.no_grad()
def cpop_score(images):
    B=images.shape[0]; x=(images*2-1)
    t=torch.full((B,),T_STEP,device=device,dtype=torch.long)
    u=unet(x,t).sample[:,:3].view(B,-1).float(); acc=torch.zeros(B,device=device)
    for _ in range(K):
        v=unet(x+torch.randn_like(x)*SIGMA,t).sample[:,:3].view(B,-1).float()
        acc+=1.0-F.cosine_similarity(u,v,dim=1)
    return -(acc/K).cpu().numpy()

@torch.no_grad()
def lid_mle(q,r,k=LID_K):
    q=q.float();r=r.float(); d=torch.cdist(q,r); rr,_=d.topk(min(k,r.shape[0]),dim=1,largest=False)
    rr=rr.clamp_min(1e-12); rk=rr[:,-1:].clamp_min(1e-12)
    return (-1.0/(torch.log(rr/rk).mean(1))).cpu().numpy()

@torch.no_grad()
def fs_common(imgs,clf):
    p=F.softmax(clf(imgs),1)
    pb=F.softmax(clf(torch.round(imgs*7)/7),1)
    ps=F.softmax(clf(F.avg_pool2d(imgs,2,1,0).clamp(0,1)) if False else clf(F.avg_pool2d(imgs,3,1,1)),1)
    return torch.maximum((p-pb).abs().sum(1),(p-ps).abs().sum(1)).cpu().numpy()

class Maha:
    def __init__(s,ref):
        ref=ref.float(); s.mu=ref.mean(0,keepdim=True); d=ref-s.mu; cov=(d.t()@d)/ref.shape[0]
        cov+=(cov.diagonal().mean()*1e-2)*torch.eye(cov.shape[0],device=cov.device); s.prec=torch.linalg.pinv(cov)
    @torch.no_grad()
    def score(s,q): q=q.float(); d=q-s.mu; return (d@s.prec*d).sum(1).clamp_min(0).sqrt().cpu().numpy()
class KD:
    def __init__(s,ref):
        s.ref=ref.float(); dd=torch.cdist(s.ref[:200],s.ref[:200]); s.h2=2*(dd.median().item()**2+1e-9)
    @torch.no_grad()
    def score(s,q):
        q=q.float(); d2=torch.cdist(q,s.ref)**2
        return (-(torch.logsumexp(-d2/s.h2,dim=1)-np.log(s.ref.shape[0]))).cpu().numpy()

ds=torchvision.datasets.CIFAR10(root='/workspace/data',train=False,download=True,transform=T.ToTensor())
imgs_all=torch.stack([ds[i][0] for i in range(1500)])   # [0,1], 32x32
ref_imgs=imgs_all[600:1100].to(device); eval_imgs=imgs_all[:600]
print(f'CIFAR-10 | archs={list(ARCH_IDS)} | N={N}',flush=True)

def make_atk(kind,clf):
    if kind=='FGSM':    return torchattacks.FGSM(clf,eps=EPS)
    if kind=='PGD':     return torchattacks.PGD(clf,eps=EPS,alpha=EPS/4,steps=20)
    if kind=='BIM':     return torchattacks.BIM(clf,eps=EPS,alpha=EPS/4,steps=20)
    if kind=='MIFGSM':  return torchattacks.MIFGSM(clf,eps=EPS,alpha=EPS/4,steps=20)
    if kind=='APGD':    return torchattacks.APGD(clf,norm='Linf',eps=EPS,steps=20)
    if kind=='AutoAttack': return torchattacks.AutoAttack(clf,norm='Linf',eps=EPS,version='standard',n_classes=10)
    if kind=='CW':      return torchattacks.CW(clf,c=1.0,steps=100,lr=0.01)
    if kind=='DeepFool':return torchattacks.DeepFool(clf,steps=50)
    if kind=='Square':  return torchattacks.Square(clf,norm='Linf',eps=EPS,n_queries=1000,n_restarts=1)
ATTACKS=['FGSM','PGD','BIM','MIFGSM','APGD','AutoAttack','CW','DeepFool','Square']
SLOW={'AutoAttack','CW','DeepFool','Square'}

rows=[]
for name,hid in ARCH_IDS.items():
    m=torch.hub.load("chenyaofo/pytorch-cifar-models",hid,pretrained=True).to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    clf=Clf(m).to(device).eval(); hook=FeatHook(clf)
    ref_feat=[]
    for i in range(0,len(ref_imgs),50): clf(ref_imgs[i:i+50]); ref_feat.append(hook.feat)
    ref_feat=torch.cat(ref_feat); maha=Maha(ref_feat); kd=KD(ref_feat)
    for kind in ATTACKS:
        Ni=6 if A.smoke else (120 if kind in SLOW else N)
        atk=make_atk(kind,clf); n=0; pbar=tqdm(total=Ni,desc=f'{name}/{kind}')
        for i in range(0,len(eval_imgs),8):
            if n>=Ni: break
            ci=eval_imgs[i:i+8].to(device)
            with torch.no_grad(): y=clf(ci).argmax(1)
            try: adv=atk(ci,y)
            except Exception as e: print(f'  !! {name}/{kind}: {e}',flush=True); break
            with torch.no_grad(): ok=clf(adv).argmax(1)!=y
            if not ok.any(): continue
            cci,ai=ci[ok],adv[ok]; take=min(cci.shape[0],Ni-n); cci,ai=cci[:take],ai[:take]
            cc,ac=cpop_score(cci),cpop_score(ai)
            with torch.no_grad(): clf(cci); fc=hook.feat
            with torch.no_grad(): clf(ai); fa=hook.feat
            cl,al=lid_mle(fc,ref_feat),lid_mle(fa,ref_feat)
            cf,af=fs_common(cci,clf),fs_common(ai,clf)
            cm,am=maha.score(fc),maha.score(fa); ck,ak=kd.score(fc),kd.score(fa)
            for j in range(take):
                rows.append({'classifier':name,'attack':kind,'label':0,'cpop':float(cc[j]),'fs':float(cf[j]),'lid':float(cl[j]),'maha':float(cm[j]),'kd':float(ck[j])})
                rows.append({'classifier':name,'attack':kind,'label':1,'cpop':float(ac[j]),'fs':float(af[j]),'lid':float(al[j]),'maha':float(am[j]),'kd':float(ak[j])})
            n+=take; pbar.update(take)
        pbar.close()
    hook.remove(); del m,clf; torch.cuda.empty_cache()

df=pd.DataFrame(rows); os.makedirs(os.path.dirname(OUT),exist_ok=True); df.to_csv(OUT,index=False); print('saved',OUT,len(df),flush=True)
def auroc(y,s):
    y=np.asarray(y);s=np.asarray(s)
    if np.isnan(s).any(): mm=~np.isnan(s); y,s=y[mm],s[mm]
    return round(roc_auc_score(y,s),3) if len(np.unique(y))>1 else np.nan
DETS=['cpop','fs','lid','maha','kd']; summ=[]
print('\n=== CIFAR AUROC per arch x attack ===',flush=True)
for name in ARCH_IDS:
    for kind in ATTACKS:
        s=df[(df.classifier==name)&(df.attack==kind)]
        if len(s)==0: continue
        row={'arch':name,'attack':kind,'n_adv':int((s.label==1).sum())}
        for d in DETS: row[d]=auroc(s.label.values,s[d].values)
        summ.append(row); print(f'{name:13s} {kind:10s}: '+'  '.join(f'{d.upper()} {row[d]}' for d in DETS),flush=True)
pd.DataFrame(summ).to_csv(SUM,index=False); print('saved',SUM,flush=True)
print('\n=== per-attack mean AUROC across archs ===',flush=True)
sdf=pd.DataFrame(summ)
for kind in ATTACKS:
    sub=sdf[sdf.attack==kind]
    if len(sub): print(f'  {kind:10s} '+'  '.join(f'{d.upper()}={sub[d].mean():.3f}' for d in DETS),flush=True)
