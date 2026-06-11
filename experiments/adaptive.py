"""P1 Exp J — adaptive, detector-aware attack against CPOP.

A white-box attacker knows both the classifier f and the detector. It runs PGD-Linf
to MAXIMIZE   L = CE(f(x+d), y)  +  lambda * CPOP(x+d),
i.e. fool the classifier while raising CPOP back toward the clean (typical) range so the
input evades the directional-freezing detector. CPOP's Monte Carlo noise is handled by
EOT: the gradient is averaged over K_eot fresh perturbations per step. lambda=0 reduces to
standard PGD (a fair in-harness baseline).

We report, per architecture and per lambda: classifier attack-success rate, clean-vs-adv
CPOP AUROC, and detection rate (TPR) among successful adversarials at the 15% FPR threshold.
Usage:  python exp_j_adaptive.py [--smoke]
"""
import sys, os, glob, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models as tvm
from diffusers import UNet2DModel
import pandas as pd, numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); A=ap.parse_args()
device=torch.device('cuda'); T_STEP,SIGMA,EPS=1,0.005,4/255
N      = 6  if A.smoke else 40
STEPS  = 4  if A.smoke else 12
K_EOT  = 2
K_EVAL = 8
ALPHA  = EPS/4
LAMBDAS= [0.0, 20.0] if A.smoke else [0.0, 20.0, 100.0]
BATCH  = 2
ADM='/workspace/data/adm_imagenet_local'; DATA='/workspace/data/imagenet_val/val_256x256'
OUT='/workspace/outputs/tables/p1_expJ15_adaptive.csv'; SUM='/workspace/outputs/tables/p1_expJ_summary.csv'
MEAN=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
STD=torch.tensor([0.229,0.225,0.224],device=device).view(1,3,1,1)

# fp32 probe (gradients through the diffusion field must be stable)
unet=UNet2DModel.from_pretrained(ADM,torch_dtype=torch.float32).to(device).eval()
for p in unet.parameters(): p.requires_grad_(False)

class Clf(nn.Module):
    def __init__(s,m):super().__init__();s.m=m
    def forward(s,x):
        x=F.interpolate(x,size=224,mode='bilinear',align_corners=False);return s.m((x-MEAN)/STD)
ARCHS={'resnet50':(tvm.resnet50,tvm.ResNet50_Weights.IMAGENET1K_V2),
       'vit_b_16':(tvm.vit_b_16,tvm.ViT_B_16_Weights.IMAGENET1K_V1)}

def cpop_diff(z):
    """differentiable CPOP at z=2x-1 (single EOT sample), returns per-sample (1-cos)."""
    t=torch.full((z.shape[0],),T_STEP,device=device,dtype=torch.long)
    u=unet(z,t).sample[:,:3].reshape(z.shape[0],-1)
    eta=torch.randn_like(z)*SIGMA
    v=unet(z+eta,t).sample[:,:3].reshape(z.shape[0],-1)
    return 1.0-F.cosine_similarity(u,v,dim=1)

@torch.no_grad()
def cpop_eval(images):
    z=(images*2-1); t=torch.full((z.shape[0],),T_STEP,device=device,dtype=torch.long)
    u=unet(z,t).sample[:,:3].reshape(z.shape[0],-1); acc=torch.zeros(z.shape[0],device=device)
    for _ in range(K_EVAL):
        v=unet(z+torch.randn_like(z)*SIGMA,t).sample[:,:3].reshape(z.shape[0],-1)
        acc+=1.0-F.cosine_similarity(u,v,dim=1)
    return (acc/K_EVAL).cpu().numpy()                    # raw CPOP (clean high, adv low)

def attack(clf,x,y,lam):
    d=torch.zeros_like(x).uniform_(-EPS,EPS)
    xadv=torch.clamp(x+d,0,1).detach()
    for _ in range(STEPS):
        xadv.requires_grad_(True)
        if xadv.grad is not None: xadv.grad=None
        ce=F.cross_entropy(clf(xadv),y); ce.backward()   # maximize CE
        if lam>0:
            z=xadv*2-1
            for _ in range(K_EOT):
                ck=cpop_diff(z).mean()
                (lam*ck/K_EOT).backward(retain_graph=False)
                z=xadv*2-1                                # rebuild graph each EOT sample
        g=xadv.grad.detach()
        xadv=(xadv+ALPHA*g.sign()).detach()
        xadv=torch.max(torch.min(xadv,x+EPS),x-EPS).clamp_(0,1)
    return xadv.detach()

tf=T.Compose([T.Resize(256),T.CenterCrop(256),T.ToTensor()])
class DS(torch.utils.data.Dataset):
    def __init__(s,p):s.p=p
    def __len__(s):return len(s.p)
    def __getitem__(s,i):return tf(Image.open(s.p[i]).convert('RGB'))
paths=sorted(glob.glob(os.path.join(DATA,'*.png'))) or sorted(glob.glob(os.path.join(DATA,'*.JPEG')))
print(f'{len(paths)} images | N={N} steps={STEPS} K_eot={K_EOT} lambdas={LAMBDAS} archs={list(ARCHS)}',flush=True)

rows=[]
for cname,(ctor,w) in ARCHS.items():
    m=ctor(weights=w).to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    clf=Clf(m).to(device).eval()
    loader=torch.utils.data.DataLoader(DS(paths[:400]),batch_size=BATCH)
    # collect clean (correctly classified) + per-lambda adv
    clean_imgs=[]; n=0
    perlam={l:{'cpop':[],'fooled':[]} for l in LAMBDAS}
    pbar=tqdm(total=N,desc=cname)
    for imgs in loader:
        if n>=N:break
        imgs=imgs.to(device)
        with torch.no_grad(): y=clf(imgs).argmax(1); correct=(y==y)  # use predicted label as target
        take=min(imgs.shape[0],N-n); imgs,y=imgs[:take],y[:take]
        clean_imgs.append(imgs.cpu())
        for l in LAMBDAS:
            adv=attack(clf,imgs,y,l)
            with torch.no_grad(): fooled=(clf(adv).argmax(1)!=y).cpu().numpy()
            cp=cpop_eval(adv)
            perlam[l]['cpop']+=list(cp); perlam[l]['fooled']+=list(fooled)
        n+=take; pbar.update(take)
    pbar.close()
    clean=torch.cat(clean_imgs)
    clean_cpop=np.concatenate([cpop_eval(clean[i:i+BATCH].to(device)) for i in range(0,len(clean),BATCH)])
    tau=np.quantile(-clean_cpop,0.85)                    # 5% FPR threshold on D=-CPOP
    for l in LAMBDAS:
        cp=np.array(perlam[l]['cpop']); fooled=np.array(perlam[l]['fooled']).astype(bool)
        y=np.r_[np.zeros(len(clean_cpop)),np.ones(len(cp))]; s=np.r_[-clean_cpop,-cp]
        au=roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan
        succ=fooled.mean()
        det=(((-cp)>tau)[fooled]).mean() if fooled.any() else np.nan   # detection among successful adv
        rows.append({'arch':cname,'lambda':l,'clf_success':round(float(succ),3),
                     'cpop_auroc':round(float(au),3),'detect_at_5fpr':round(float(det),3),
                     'adv_cpop_mean':round(float(cp.mean()),4),'clean_cpop_mean':round(float(clean_cpop.mean()),4)})
        print(f'  {cname:10s} lam={l:6.1f}: clf_succ={succ:.3f} CPOP_AUROC={au:.3f} detect@15%FPR={det:.3f} (adv_cpop={cp.mean():.4f} vs clean {clean_cpop.mean():.4f})',flush=True)
    del m,clf; torch.cuda.empty_cache()

df=pd.DataFrame(rows); os.makedirs(os.path.dirname(OUT),exist_ok=True)
df.to_csv(OUT,index=False); df.to_csv(SUM,index=False); print('saved',OUT,flush=True)
print('\nReading: lambda=0 is standard PGD. If CPOP_AUROC / detect@15%FPR stay high as lambda grows,',flush=True)
print('the detector resists adaptive evasion; if they collapse, report as an honest limitation.',flush=True)
