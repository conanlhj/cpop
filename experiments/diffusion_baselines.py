"""P1 Exp I — diffusion-prior baselines (is the angular normalization necessary?).

Same fixed ADM probe, same clean/PGD inputs, but different scores derived from the
diffusion field s(x,t):
  - cpop        : E_delta[1 - cos(s(x), s(x+delta))]            (ours: angular instability)
  - raw_var     : E_delta || s(x+delta) - s(x) ||^2 / sigma^2   (un-normalized local variation)
  - score_norm  : || s(x) ||                                    (point score magnitude
                                                                 ~ denoising residual at low t)
All are classifier-agnostic (image + probe only). For each metric we orient the score so
the adversarial class scores higher (report max(AUROC, 1-AUROC)) and report per-arch AUROC,
mean, and cross-architecture std. If a simpler magnitude score matches cpop, the angular
normalization is unnecessary; if cpop wins (esp. on stability), it is justified.
Usage:  python exp_i_diffusion_baselines.py [--smoke]
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

ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); A=ap.parse_args()
device=torch.device('cuda'); T_STEP,SIGMA,K,BATCH,EPS=1,0.005,8,4,4/255
N=8 if A.smoke else 150
ADM='/workspace/data/adm_imagenet_local'; DATA='/workspace/data/imagenet_val/val_256x256'
OUT='/workspace/outputs/tables/p1_expI_diffusion_baselines.csv'; SUM='/workspace/outputs/tables/p1_expI_summary.csv'
MEAN=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
STD=torch.tensor([0.229,0.225,0.224],device=device).view(1,3,1,1)
unet=UNet2DModel.from_pretrained(ADM,torch_dtype=torch.float16).to(device).eval()
for p in unet.parameters(): p.requires_grad_(False)

class Clf(nn.Module):
    def __init__(s,m):super().__init__();s.m=m
    def forward(s,x):
        x=F.interpolate(x,size=224,mode='bilinear',align_corners=False);return s.m((x-MEAN)/STD)
ARCHS={'resnet50':(tvm.resnet50,tvm.ResNet50_Weights.IMAGENET1K_V2),
       'densenet121':(tvm.densenet121,tvm.DenseNet121_Weights.IMAGENET1K_V1),
       'vit_b_16':(tvm.vit_b_16,tvm.ViT_B_16_Weights.IMAGENET1K_V1),
       'swin_t':(tvm.swin_t,tvm.Swin_T_Weights.IMAGENET1K_V1)}
if A.smoke: ARCHS={k:ARCHS[k] for k in ['resnet50','vit_b_16']}

@torch.no_grad()
def scores(images):
    """returns dict metric -> per-image score (numpy)."""
    B=images.shape[0]; x=(images*2-1).to(torch.float16)
    t=torch.full((B,),T_STEP,device=device,dtype=torch.long)
    u=unet(x,t).sample[:,:3].reshape(B,-1).float()
    score_norm=u.norm(dim=1)
    acc_cos=torch.zeros(B,device=device); acc_raw=torch.zeros(B,device=device)
    for _ in range(K):
        v=unet(x+torch.randn_like(x)*SIGMA,t).sample[:,:3].reshape(B,-1).float()
        acc_cos+=1.0-F.cosine_similarity(u,v,dim=1)
        acc_raw+=((v-u)**2).sum(1)/(SIGMA**2)
    cpop=acc_cos/K; raw=acc_raw/K
    return {'cpop':cpop.cpu().numpy(),'raw_var':raw.cpu().numpy(),'score_norm':score_norm.cpu().numpy()}

tf=T.Compose([T.Resize(256),T.CenterCrop(256),T.ToTensor()])
class DS(torch.utils.data.Dataset):
    def __init__(s,p):s.p=p
    def __len__(s):return len(s.p)
    def __getitem__(s,i):return tf(Image.open(s.p[i]).convert('RGB'))
paths=sorted(glob.glob(os.path.join(DATA,'*.png'))) or sorted(glob.glob(os.path.join(DATA,'*.JPEG')))
print(f'{len(paths)} images | N={N} archs={list(ARCHS)}',flush=True)
METRICS=['cpop','raw_var','score_norm']

rows=[]
for cname,(ctor,w) in ARCHS.items():
    m=ctor(weights=w).to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    clf=Clf(m).to(device).eval(); atk=torchattacks.PGD(clf,eps=EPS,alpha=EPS/4,steps=20)
    loader=torch.utils.data.DataLoader(DS(paths[:600]),batch_size=BATCH); n=0; pbar=tqdm(total=N,desc=cname)
    for imgs in loader:
        if n>=N:break
        imgs=imgs.to(device)
        with torch.no_grad(): y=clf(imgs).argmax(1)
        adv=atk(imgs,y)
        with torch.no_grad(): ok=clf(adv).argmax(1)!=y
        if not ok.any():continue
        ci,ai=imgs[ok],adv[ok]; take=min(ci.shape[0],N-n); ci,ai=ci[:take],ai[:take]
        sc,sa=scores(ci),scores(ai)
        for j in range(take):
            for mt in METRICS:
                rows.append({'classifier':cname,'metric':mt,'label':0,'score':float(sc[mt][j])})
                rows.append({'classifier':cname,'metric':mt,'label':1,'score':float(sa[mt][j])})
        n+=take; pbar.update(take)
    pbar.close(); del m,clf; torch.cuda.empty_cache()

df=pd.DataFrame(rows); os.makedirs(os.path.dirname(OUT),exist_ok=True)
df.to_csv(OUT,index=False); print('saved',OUT,len(df),flush=True)

def oriented_auroc(y,s):
    a=roc_auc_score(y,s); return max(a,1-a)
summ=[]
print('\n=== oriented AUROC per arch x metric (PGD-Linf eps=4/255) ===',flush=True)
print(f'{"arch":14s} '+'  '.join(f'{m:10s}' for m in METRICS),flush=True)
for cname in ARCHS:
    row={'arch':cname}
    for mt in METRICS:
        s=df[(df.classifier==cname)&(df.metric==mt)]
        row[mt]=round(oriented_auroc(s.label.values,s.score.values),3)
    summ.append(row)
    print(f'{cname:14s} '+'  '.join(f'{row[m]:<10.3f}' for m in METRICS),flush=True)
sdf=pd.DataFrame(summ); sdf.to_csv(SUM,index=False)
print('\n=== mean / std across archs ===',flush=True)
for mt in METRICS:
    v=sdf[mt].values; print(f'  {mt:10s} mean={v.mean():.3f} std={v.std():.3f} min={v.min():.3f}',flush=True)
print('\nReading: if cpop >= magnitude baselines (esp. higher mean / lower std), angular normalization is justified.',flush=True)
