import csv, statistics as st, os
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
OUT='/w/paper_v2/figures/'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.grid':True,'grid.alpha':0.3,'figure.dpi':150,'savefig.bbox':'tight'})

rows=[r for r in csv.DictReader(open('/w/paper_P1/tables/main/p1_expA_cross_arch.csv')) if r['eps255']=='4']
archs=sorted(set(r['classifier'] for r in rows))
def q(v,p):
    v=sorted(v); return v[min(len(v)-1,max(0,int(round(p*(len(v)-1)))))]
def f(r,c):
    try: return float(r[c])
    except: return None
fs_sq=['fs_bit1','fs_bit2','fs_bit3','fs_bit5','fs_median3','fs_median5','fs_avgpool3']
def tpr(cl,ad,a):
    t=q(cl,1-a); return sum(1 for x in ad if x>t)/len(ad)
alphas=[0.05,0.10,0.15,0.20]
def curve(detcol,tuned=False):
    out=[]
    for a in alphas:
        vals=[]
        for arch in archs:
            ar=[r for r in rows if r['classifier']==arch]; cl=[r for r in ar if r['label']=='0']; ad=[r for r in ar if r['label']=='1']
            if tuned:
                sids=sorted(set(int(r['sid']) for r in ar)); val=set(sids[:len(sids)//2]); best=-1; bs=fs_sq[0]
                for sq in fs_sq:
                    t=tpr([f(r,sq) for r in cl if int(r['sid']) in val],[f(r,sq) for r in ad if int(r['sid']) in val],a)
                    if t>best: best=t; bs=sq
                vals.append(tpr([f(r,bs) for r in cl if int(r['sid']) not in val],[f(r,bs) for r in ad if int(r['sid']) not in val],a))
            else:
                vals.append(tpr([f(r,detcol) for r in cl],[f(r,detcol) for r in ad],a))
        out.append(st.mean(vals))
    return out

# Fig: TPR vs FPR (fair, FPR-matched, same-classifier) -- value-annotated, no title
fig,ax=plt.subplots(figsize=(6.6,4.4)); xs=[a*100 for a in alphas]
for det,lab,mk,c,tn in [('cpop','CPOP (no access)','o','#1f77b4',False),(None,'FS tuned (per-classifier)','s','#ff7f0e',True),('fs','FS common','^','#d62728',False),('lid','LID','d','#2ca02c',False)]:
    y=curve(det,tuned=tn); ax.plot(xs,y,marker=mk,ls='-' if det=='cpop' else '--',lw=2.2 if det=='cpop' else 1.6,label=lab,color=c)
    if det=='cpop':
        for xi,yi in zip(xs,y): ax.annotate(f'{yi:.2f}',(xi,yi),textcoords='offset points',xytext=(0,7),ha='center',fontsize=8.5,color=c)
ax.axvline(15,ls=':',color='gray',alpha=.6)
ax.set_xlabel('target false positive rate (%)'); ax.set_ylabel('true positive rate'); ax.set_xticks(xs); ax.set_ylim(0,1)
ax.legend(fontsize=9.5,loc='lower right'); fig.savefig(OUT+'fig_tpr_fpr.pdf'); plt.close(); print('tpr_fpr ok')

# Fig: operating-point transfer boxplot (realized FPR after transfer, 15% target)
E=[r for r in csv.DictReader(open('/w/paper_P1/tables/main/p1_expE_threshold_transfer.csv')) if r['self']=='False']
# Exp E csv is at 5%; regenerate realized-FPR-after-transfer at 15% from Exp A per-sample for consistency
def transfer_realfpr(det,alpha=0.15):
    P={a:{'cl':[f(r,det) for r in rows if r['classifier']==a and r['label']=='0']} for a in archs}
    vals=[]
    for s in archs:
        tau=q(P[s]['cl'],1-alpha)
        for t in archs:
            if t==s: continue
            vals.append(sum(1 for x in P[t]['cl'] if x>tau)/len(P[t]['cl']))
    return vals
dets=[('cpop','CPOP'),('fs','FS common'),('fs_bit2','FS tuned'),('lid','LID')]
data=[transfer_realfpr(d) for d,_ in dets]
fig,ax=plt.subplots(figsize=(6.6,4.0)); bp=ax.boxplot(data,labels=[l for _,l in dets],showmeans=True,patch_artist=True)
for i,p in enumerate(bp['boxes']): p.set_facecolor('#1f77b4' if i==0 else '#bbbbbb'); p.set_alpha(.6)
ax.axhline(0.15,ls='--',color='green',label='target FPR = 15%'); ax.set_ylabel('realized FPR after threshold transfer'); ax.legend(fontsize=10)
fig.savefig(OUT+'fig_optransfer.pdf'); plt.close(); print('optransfer ok')

# Fig: two-regime (original style, no title): scatter + eps-binned mean + Regime B fit + Regime A plateau
df=pd.read_csv('/c/outputs/tables/theorem2_r_dependence.csv'); df=df[df['eps']>0]
fig,ax=plt.subplots(figsize=(6.6,4.4))
ax.scatter(df['l2_norm'],df['cpop'],s=16,alpha=0.35,color='#888',linewidths=0)
evs=sorted(df['eps'].unique()); mx=[df[df.eps==e]['l2_norm'].mean() for e in evs]; my=[df[df.eps==e]['cpop'].mean() for e in evs]
ax.plot(mx,my,'o-',color='#1A1A2E',lw=2,ms=8,label=r'$\epsilon$-binned mean')
rb=df[df['l2_norm']>1.0]
sl,b=np.polyfit(rb['log_l2'],rb['log_cpop'],1); xf=np.logspace(0,np.log10(rb['l2_norm'].max()),50)
ax.plot(xf,10**(sl*np.log10(xf)+b),'-',color='#E15759',lw=2.5,label=f'Regime B fit: slope $={sl:.2f}$')
pl=df[df['l2_norm']<0.5]['cpop'].mean(); ax.axhline(pl,color='#59A14F',ls='--',lw=2,alpha=.7,label=f'Regime A plateau $\\approx {pl:.3f}$')
ax.axvline(1.0,color='gray',ls=':',alpha=.6); ax.text(1.05,df['cpop'].min()*1.5,r'$r^{*}\approx 1$',fontsize=10,color='gray',style='italic')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$\|\eta\|_2$  (L2 perturbation norm)'); ax.set_ylabel('CPOP')
ax.legend(loc='lower left',fontsize=9.5); fig.savefig(OUT+'fig_two_regime.pdf'); plt.close(); print('two_regime ok')

# Fig: noise ablation heatmap WITH per-cell numbers, no title
H=list(csv.DictReader(open('/c/outputs/tables/p1_expH_noise_ablation.csv')))
ts=sorted(set(int(r['t']) for r in H)); sgs=sorted(set(float(r['sigma']) for r in H))
agg=defaultdict(list)
for r in H: agg[(int(r['t']),float(r['sigma']))].append(float(r['auroc']))
M=np.full((len(ts),len(sgs)),np.nan)
for i,t in enumerate(ts):
    for j,s in enumerate(sgs):
        if (t,s) in agg: M[i,j]=st.mean(agg[(t,s)])
fig,ax=plt.subplots(figsize=(7.0,4.4)); im=ax.imshow(M,aspect='auto',origin='lower',cmap='viridis',vmin=0.5,vmax=1.0)
for i in range(len(ts)):
    for j in range(len(sgs)):
        v=M[i,j]
        if v==v: ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8,color='white' if v<0.78 else 'black')
ax.set_xticks(range(len(sgs))); ax.set_xticklabels([f'{s:g}' for s in sgs],rotation=45); ax.set_yticks(range(len(ts))); ax.set_yticklabels(ts)
ax.set_xlabel(r'probe scale $\sigma$'); ax.set_ylabel('timestep $t$'); ax.grid(False); fig.colorbar(im,ax=ax,label='AUROC')
fig.savefig(OUT+'fig_noise.pdf'); plt.close(); print('noise ok')

# Fig: clean/adv/OOD CPOP histogram, no title
d=defaultdict(list)
for r in csv.DictReader(open('/c/outputs/tables/imagenet_ood_3class.csv')): d[r['category']].append(float(r['iso_cpop']))
fig,ax=plt.subplots(figsize=(6.6,3.9)); bins=np.linspace(0,0.72,46)
ax.hist(d['adv'],bins=bins,alpha=.6,color='#d62728',label='adversarial',density=True)
ax.hist(d['clean'],bins=bins,alpha=.6,color='#1f77b4',label='clean',density=True)
ax.hist(d['ood'],bins=bins,alpha=.6,color='#2ca02c',label='out-of-distribution',density=True)
ax.set_xlabel('CPOP'); ax.set_ylabel('density'); ax.set_xlim(0,0.72); ax.legend(fontsize=10)
fig.savefig(OUT+'fig_ood_hist.pdf'); plt.close(); print('ood_hist ok')
print('FIGS:',sorted(os.listdir(OUT)))
