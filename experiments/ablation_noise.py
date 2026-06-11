"""P1 Exp H — WIDE (t, sigma) noise ablation: is CPOP only a low-noise phenomenon?

Two noises in CPOP:
  t      = diffusion timestep (the variance-schedule point the model assumes for the input).
  sigma  = magnitude of the probe perturbation delta added to measure directional stability.
Map AUROC over a WIDE grid (sigma up to 0.5, t up to 200) to see where the signal peaks
and where it dies -> honest characterization of the operating regime, not a single t=1 point.
2 archs (resnet50, vit_b_16), PGD-Linf eps=4/255, N=100. Adversarials cached once.
Outputs CSV + an AUROC heatmap (t x sigma).
Usage:  python exp_h_noise_ablation.py [--smoke]
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
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from utils.patches import apply_cw_patch
apply_cw_patch()

ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); A=ap.parse_args()
device=torch.device('cuda'); BATCH,EPS,K=4,4/255,8
N = 8 if A.smoke else 100
T_GRID  = [1,20] if A.smoke else [1,5,20,50,100,200]
SIG_GRID= [0.005,0.1] if A.smoke else [0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5]
ADM_PATH='/workspace/data/adm_imagenet_local'; DATA_DIR='/workspace/data/imagenet_val/val_256x256'
OUT='/workspace/outputs/tables/p1_expH_noise_ablation.csv'
FIG='/workspace/outputs/figures/p1/F6_noise_ablation_heatmap.pdf'
MEAN=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
STD =torch.tensor([0.229,0.225,0.224],device=device).view(1,3,1,1)

unet=UNet2DModel.from_pretrained(ADM_PATH,torch_dtype=torch.float16).to(device).eval()
for p in unet.parameters(): p.requires_grad_(False)
class Clf(nn.Module):
    def __init__(s,m):super().__init__();s.m=m
    def forward(s,x):
        x=F.interpolate(x,size=224,mode='bilinear',align_corners=False);return s.m((x-MEAN)/STD)
ARCHS={'resnet50':(tvm.resnet50,tvm.ResNet50_Weights.IMAGENET1K_V2),
       'vit_b_16':(tvm.vit_b_16,tvm.ViT_B_16_Weights.IMAGENET1K_V1)}
if A.smoke: ARCHS={'resnet50':ARCHS['resnet50']}

@torch.no_grad()
def cpop_score(images,t_step,sigma):
    B=images.shape[0]; x=(images*2-1).to(torch.float16)
    t=torch.full((B,),t_step,device=device,dtype=torch.long)
    u=unet(x,t).sample[:,:3].reshape(B,-1).float(); acc=torch.zeros(B,device=device)
    for _ in range(K):
        v=unet(x+torch.randn_like(x)*sigma,t).sample[:,:3].reshape(B,-1).float()
        acc+=1.0-F.cosine_similarity(u,v,dim=1)
    return -(acc/K).cpu().numpy()

tf=T.Compose([T.Resize(256),T.CenterCrop(256),T.ToTensor()])
class DS(torch.utils.data.Dataset):
    def __init__(s,p):s.p=p
    def __len__(s):return len(s.p)
    def __getitem__(s,i):return tf(Image.open(s.p[i]).convert('RGB'))
paths=sorted(glob.glob(os.path.join(DATA_DIR,'*.png'))) or sorted(glob.glob(os.path.join(DATA_DIR,'*.JPEG')))
print(f'{len(paths)} images | N={N} archs={list(ARCHS)} t={T_GRID} sig={SIG_GRID}',flush=True)

cache={}
for cname,(ctor,w) in ARCHS.items():
    m=ctor(weights=w).to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    clf=Clf(m).to(device).eval(); atk=torchattacks.PGD(clf,eps=EPS,alpha=EPS/4,steps=20)
    loader=torch.utils.data.DataLoader(DS(paths[:600]),batch_size=BATCH); cl,ad=[],[]; n=0; pbar=tqdm(total=N,desc=f'atk {cname}')
    for imgs in loader:
        if n>=N:break
        imgs=imgs.to(device)
        with torch.no_grad(): y=clf(imgs).argmax(1)
        adv=atk(imgs,y)
        with torch.no_grad(): ok=clf(adv).argmax(1)!=y
        if not ok.any():continue
        ci,ai=imgs[ok],adv[ok]; take=min(ci.shape[0],N-n)
        for j in range(take): cl.append(ci[j].cpu()); ad.append(ai[j].cpu())
        n+=take; pbar.update(take)
    pbar.close(); cache[cname]=(torch.stack(cl),torch.stack(ad)); del m,clf; torch.cuda.empty_cache()

rows=[]
for t_step in T_GRID:
    for sig in SIG_GRID:
        for cname,(cl,ad) in cache.items():
            cs=np.concatenate([cpop_score(cl[i:i+BATCH].to(device),t_step,sig) for i in range(0,len(cl),BATCH)])
            as_=np.concatenate([cpop_score(ad[i:i+BATCH].to(device),t_step,sig) for i in range(0,len(ad),BATCH)])
            y=np.r_[np.zeros(len(cs)),np.ones(len(as_))]; s=np.r_[cs,as_]
            au=roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan
            rows.append({'t':t_step,'sigma':sig,'arch':cname,'auroc':round(au,3),
                         'clean_mean':round(float(cs.mean()),5),'adv_mean':round(float(as_.mean()),5),
                         'delta':round(float(cs.mean()-as_.mean()),5)})
        mau=np.mean([r['auroc'] for r in rows if r['t']==t_step and r['sigma']==sig])
        print(f'  t={t_step:3d} sig={sig:.3f}: mean AUROC={mau:.3f}',flush=True)
df=pd.DataFrame(rows); os.makedirs(os.path.dirname(OUT),exist_ok=True); df.to_csv(OUT,index=False)
print('saved',OUT,len(df),flush=True)

# heatmap (mean AUROC across archs)
piv=df.groupby(['t','sigma']).auroc.mean().reset_index().pivot(index='t',columns='sigma',values='auroc')
piv=piv.reindex(index=T_GRID,columns=SIG_GRID)
fig,ax=plt.subplots(figsize=(8,4.5)); im=ax.imshow(piv.values,aspect='auto',cmap='viridis',vmin=0.5,vmax=1.0,origin='upper')
ax.set_xticks(range(len(SIG_GRID))); ax.set_xticklabels([str(s) for s in SIG_GRID])
ax.set_yticks(range(len(T_GRID))); ax.set_yticklabels([str(t) for t in T_GRID])
ax.set_xlabel('probe perturbation σ'); ax.set_ylabel('diffusion timestep t (assumed noise level)')
for i in range(len(T_GRID)):
    for j in range(len(SIG_GRID)):
        v=piv.values[i,j]
        if not np.isnan(v): ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='w' if v<0.8 else 'k',fontsize=8)
ax.set_title('CPOP AUROC over (t, σ) — where the directional-freezing signal lives')
fig.colorbar(im,label='AUROC'); fig.tight_layout(); fig.savefig(FIG); print('saved',FIG,flush=True)

print('\n=== best per t (across sigma) ===',flush=True)
for t_step in T_GRID:
    sub=df[df.t==t_step].groupby('sigma').auroc.mean()
    print(f'  t={t_step:3d}: best AUROC={sub.max():.3f} at sigma={sub.idxmax()}  (worst={sub.min():.3f})',flush=True)
print('\nReading: if signal survives only at small t AND small sigma -> CPOP is a low-noise probe (honest scope).',flush=True)
print('If a larger-sigma optimum appears at higher t -> the two noises trade off (schedule-aware tuning helps).',flush=True)
