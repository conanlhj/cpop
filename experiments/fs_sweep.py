"""P1 Exp D — FeatureSqueeze parameter sanity sweep.

Question: is FS's collapse on vgg16/densenet121 a parameter artifact, or genuine?
Sweep squeezers x score functions per architecture; report detector AUROC for each.
If FS stays low on vgg/densenet across ALL configs -> collapse is genuine, not a bug.
Also report per-arch best-config ("val-tuned" proxy) and combined-max ("common").
Attack: PGD-Linf eps=4/255. No diffusion needed -> fast.
Usage:  python exp_d_fs_sanity.py [--smoke]
"""
import sys, os, glob, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models as tvm
import torchattacks, pandas as pd, numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.patches import apply_cw_patch
apply_cw_patch()

ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); A=ap.parse_args()
device=torch.device('cuda'); BATCH,EPS=4,4/255
N=6 if A.smoke else 150
DATA_DIR='/workspace/data/imagenet_val/val_256x256'
OUT='/workspace/outputs/tables/p1_expD_fs_sanity.csv'
MEAN=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
STD =torch.tensor([0.229,0.225,0.224],device=device).view(1,3,1,1)

class Clf(nn.Module):
    def __init__(s,m):super().__init__();s.m=m
    def forward(s,x):
        x=F.interpolate(x,size=224,mode='bilinear',align_corners=False); return s.m((x-MEAN)/STD)

ARCHS={'resnet50':(tvm.resnet50,tvm.ResNet50_Weights.IMAGENET1K_V2),
       'vgg16':(tvm.vgg16,tvm.VGG16_Weights.IMAGENET1K_V1),
       'densenet121':(tvm.densenet121,tvm.DenseNet121_Weights.IMAGENET1K_V1),
       'vit_b_16':(tvm.vit_b_16,tvm.ViT_B_16_Weights.IMAGENET1K_V1)}
if A.smoke: ARCHS={k:ARCHS[k] for k in ['resnet50','vgg16']}

def bitdepth(x,b):
    L=2**b-1; return torch.round(x*L)/L
def medianf(x,k):
    p=k//2; xp=F.pad(x,(p,p,p,p),mode='reflect'); pa=xp.unfold(2,k,1).unfold(3,k,1)
    return pa.contiguous().view(*pa.shape[:4],k*k).median(-1)[0]
def avgpool3(x): return F.avg_pool2d(x,3,1,1)
SQUEEZERS={'bit1':lambda x:bitdepth(x,1),'bit2':lambda x:bitdepth(x,2),
           'bit4':lambda x:bitdepth(x,4),'bit5':lambda x:bitdepth(x,5),
           'median3':lambda x:medianf(x,3),'median5':lambda x:medianf(x,5),
           'avgpool3':avgpool3}

@torch.no_grad()
def scores(imgs,clf):
    """returns dict squeezer-> (L1, JS) detector score per image."""
    p=F.softmax(clf(imgs),1); out={}
    for nm,sq in SQUEEZERS.items():
        q=F.softmax(clf(sq(imgs)),1)
        l1=(p-q).abs().sum(1)
        mlog=torch.log(((p+q)/2).clamp_min(1e-12))
        js=0.5*(F.kl_div(mlog,p,reduction='none').sum(1)+F.kl_div(mlog,q,reduction='none').sum(1))
        out[nm]=(l1.cpu().numpy(), js.cpu().numpy())
    return out

tf=T.Compose([T.Resize(256),T.CenterCrop(256),T.ToTensor()])
class DS(torch.utils.data.Dataset):
    def __init__(s,p):s.p=p
    def __len__(s):return len(s.p)
    def __getitem__(s,i):return tf(Image.open(s.p[i]).convert('RGB'))
paths=sorted(glob.glob(os.path.join(DATA_DIR,'*.png'))) or sorted(glob.glob(os.path.join(DATA_DIR,'*.JPEG')))
print(f'{len(paths)} images | N={N}',flush=True)

rows=[]
for name,(ctor,w) in ARCHS.items():
    m=ctor(weights=w).to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    clf=Clf(m).to(device).eval(); atk=torchattacks.PGD(clf,eps=EPS,alpha=EPS/4,steps=20)
    loader=torch.utils.data.DataLoader(DS(paths[:600]),batch_size=BATCH)
    acc={nm:{'cl1':[],'al1':[],'cjs':[],'ajs':[]} for nm in SQUEEZERS}; n=0; pbar=tqdm(total=N,desc=name)
    for imgs in loader:
        if n>=N:break
        imgs=imgs.to(device)
        with torch.no_grad(): y=clf(imgs).argmax(1)
        adv=atk(imgs,y)
        with torch.no_grad(): ok=clf(adv).argmax(1)!=y
        if not ok.any():continue
        ci,ai=imgs[ok],adv[ok]; take=min(ci.shape[0],N-n); ci,ai=ci[:take],ai[:take]
        sc,sa=scores(ci,clf),scores(ai,clf)
        for nm in SQUEEZERS:
            acc[nm]['cl1']+=list(sc[nm][0]); acc[nm]['al1']+=list(sa[nm][0])
            acc[nm]['cjs']+=list(sc[nm][1]); acc[nm]['ajs']+=list(sa[nm][1])
        n+=take; pbar.update(take)
    pbar.close()
    # per-squeezer AUROC
    for nm in SQUEEZERS:
        for st,ck,ak in [('L1','cl1','al1'),('JS','cjs','ajs')]:
            y=np.r_[np.zeros(len(acc[nm][ck])),np.ones(len(acc[nm][ak]))]
            s=np.r_[acc[nm][ck],acc[nm][ak]]
            au=roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan
            rows.append({'arch':name,'squeezer':nm,'score':st,'auroc':round(au,3)})
    del m,clf; torch.cuda.empty_cache()

df=pd.DataFrame(rows); os.makedirs(os.path.dirname(OUT),exist_ok=True)
df.to_csv(OUT,index=False); print('saved',OUT,len(df),flush=True)

print('\n=== FS AUROC per arch x squeezer (L1) ===',flush=True)
piv=df[df.score=='L1'].pivot(index='squeezer',columns='arch',values='auroc')
print(piv.to_string(),flush=True)
print('\n=== per-arch BEST single squeezer (val-tuned proxy) vs common(bit5/median) ===',flush=True)
for name in ARCHS:
    s=df[(df.arch==name)]
    best=s.loc[s.auroc.idxmax()]
    print(f'  {name:14s} best={best.auroc} ({best.squeezer}/{best.score})  '
          f'bit5/L1={s[(s.squeezer=="bit5")&(s.score=="L1")].auroc.values}',flush=True)
print('\nIf vgg16/densenet121 stay low across ALL squeezers -> FS collapse is genuine, not a param bug.',flush=True)
