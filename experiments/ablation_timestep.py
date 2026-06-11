"""Refined dimension-aware (t, sigma) probe for ImageNet (eps=4/255), per collaborator analysis.

Tests whether the ImageNet detection gap is an effective-radius issue: the L2 probe radius rho=sigma*sqrt(D)
is ~8x larger on ImageNet (D=196608) than CIFAR (D=3072) at the same pixel sigma. We sweep SMALL sigma
(down to the CIFAR-radius-matched 0.000625 and the eps/sigma-ratio-matched 0.0025) crossed with sub-1 t,
and log AUROC AND the score-norm ||s_theta(x,t)|| for clean/adv (to catch norm collapse at small t).
Prediction: if sigma=0.005 already leaves the local-linear regime on ImageNet, smaller sigma should keep
(or raise) AUROC; if we are inside the local regime, AUROC is ~sigma-invariant (flat).
"""
import sys, os, glob
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models as tvm
from diffusers import UNet2DModel
import torchattacks, numpy as np, math
from PIL import Image

device=torch.device('cuda'); EPS,K,BATCH,N=4/255,8,4,150
ADM='/workspace/data/adm_imagenet_local'; DATA='/workspace/data/imagenet_val/val_256x256'
MEAN=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
STD =torch.tensor([0.229,0.225,0.224],device=device).view(1,3,1,1)
D=3*256*256; sqrtD=math.sqrt(D)
TS=[0.1,0.25,0.5,0.75,1.0]
SIGS=[0.0003,0.000625,0.00125,0.0025,0.005]   # rho=sigma*sqrtD spans 0.13..2.22; 0.000625=CIFAR-radius-matched

unet=UNet2DModel.from_pretrained(ADM,torch_dtype=torch.float16).to(device).eval()
for p in unet.parameters(): p.requires_grad_(False)
class Clf(nn.Module):
    def __init__(s,m):super().__init__();s.m=m
    def forward(s,x):
        x=F.interpolate(x,224,mode='bilinear',align_corners=False);return s.m((x-MEAN)/STD)
ARCHS={'resnet50':(tvm.resnet50,tvm.ResNet50_Weights.IMAGENET1K_V2),
       'densenet121':(tvm.densenet121,tvm.DenseNet121_Weights.IMAGENET1K_V1),
       'vit_b_16':(tvm.vit_b_16,tvm.ViT_B_16_Weights.IMAGENET1K_V1)}

@torch.no_grad()
def cpop_and_norm(imgs,t_val,sigma):
    B=imgs.shape[0]; x=(imgs*2-1).to(torch.float16)
    t=torch.full((B,),float(t_val),device=device,dtype=torch.float16)
    u=unet(x,t).sample[:,:3].view(B,-1).float()
    acc=torch.zeros(B,device=device)
    for _ in range(K):
        v=unet(x+torch.randn_like(x)*sigma,t).sample[:,:3].view(B,-1).float()
        acc+=1.0-F.cosine_similarity(u,v,dim=1,eps=1e-12)
    return (acc/K).cpu().numpy(), u.norm(dim=1).cpu().numpy()

def auroc(clean,adv):  # adv expected lower CPOP -> detect with -CPOP
    s=list(-np.array(clean))+list(-np.array(adv)); y=[0]*len(clean)+[1]*len(adv)
    import bisect; srt=sorted(s); n1=len(adv); n0=len(clean); rs=0.0
    for v,lab in zip(s,y):
        if lab==1: rs+=(bisect.bisect_left(srt,v)+1+bisect.bisect_right(srt,v))/2.0
    return (rs-n1*(n1+1)/2)/(n1*n0)

tf=T.Compose([T.Resize(256),T.CenterCrop(256),T.ToTensor()])
class DS(torch.utils.data.Dataset):
    def __init__(s,p):s.p=p
    def __len__(s):return len(s.p)
    def __getitem__(s,i):return tf(Image.open(s.p[i]).convert('RGB'))
paths=sorted(glob.glob(DATA+'/*.png')) or sorted(glob.glob(DATA+'/*.JPEG'))
print(f'D={D} sqrtD={sqrtD:.1f} | rho=sigma*sqrtD for SIGS: '+', '.join(f'{s}->{s*sqrtD:.2f}' for s in SIGS),flush=True)
print(f'archs={list(ARCHS)} TS={TS}',flush=True)

aurocs={}; norms={}
for name,(ctor,w) in ARCHS.items():
    m=ctor(weights=w).to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    clf=Clf(m).to(device).eval(); atk=torchattacks.PGD(clf,eps=EPS,alpha=EPS/4,steps=20)
    cleans=[];advs=[];n=0
    for imgs in torch.utils.data.DataLoader(DS(paths[:600]),batch_size=BATCH):
        if n>=N: break
        imgs=imgs.to(device)
        with torch.no_grad(): y=clf(imgs).argmax(1)
        adv=atk(imgs,y)
        with torch.no_grad(): ok=clf(adv).argmax(1)!=y
        if not ok.any(): continue
        take=min(int(ok.sum()),N-n); cleans.append(imgs[ok][:take].cpu()); advs.append(adv[ok][:take].cpu()); n+=take
    cleans=torch.cat(cleans); advs=torch.cat(advs)
    print(f'\n[{name}] {len(cleans)} pairs',flush=True)
    for t_val in TS:
        for sg in SIGS:
            cc,cn=zip(*[cpop_and_norm(cleans[i:i+BATCH].to(device),t_val,sg) for i in range(0,len(cleans),BATCH)])
            ac,an=zip(*[cpop_and_norm(advs[i:i+BATCH].to(device),t_val,sg) for i in range(0,len(advs),BATCH)])
            cc=np.concatenate(cc);ac=np.concatenate(ac);cn=np.concatenate(cn);an=np.concatenate(an)
            a=auroc(cc,ac); aurocs.setdefault((t_val,sg),[]).append(a); norms.setdefault((t_val,sg),[]).append((cn.mean(),an.mean()))
            print(f'  t={t_val:<5} sig={sg:<8} (rho={sg*sqrtD:.2f}) AUROC={a:.3f}  ||s||clean={cn.mean():.3f} adv={an.mean():.3f}',flush=True)
    del m,clf; torch.cuda.empty_cache()

print('\n=== mean AUROC over archs (rows t, cols sigma) ===',flush=True)
print('t\\sig  '+'  '.join(f'{s:>8}' for s in SIGS),flush=True)
best=(-1,None)
for t_val in TS:
    row=[float(np.mean(aurocs[(t_val,sg)])) for sg in SIGS]
    for sg,m in zip(SIGS,row):
        if m>best[0]: best=(m,(t_val,sg))
    print(f'{t_val:<6} '+'  '.join(f'{r:>8.3f}' for r in row),flush=True)
print(f'\nBEST {best[0]:.3f} at (t,sigma)={best[1]}',flush=True)
print('(reference: ImageNet eps=4 default t=1,sigma=0.005 mean AUROC ~0.903 over 8 archs)',flush=True)
