"""G0 — does the ADM probe load completely? (diffusers 0.29.2 drops mid_block.attentions.0.group_norm when
resnet_time_scale_shift != 'default' and attn_norm_num_groups is unset; model card points to diffusers PR #6730.)
Compares the probe as used in the submission ('asis') with the fully loaded one ('fixed', attn_norm_num_groups=32):
  1. checkpoint keys that each model ignores / misses
  2. eps-hat agreement at t=1 on clean images
  3. denoising MSE ||eps_hat(x_t,t) - eps||^2 at several t  (which model actually denoises)
  4. CPOP AUROC, ResNet-50 PGD-Linf eps=4/255 (GT labels, uint8), clean = test split; Spearman between probes
out: OUTD/g0_groupnorm.json"""
import json, numpy as np, torch, torchattacks
from scipy.stats import rankdata, spearmanr
from diffusers import UNet2DModel
from common import *

torch.set_num_threads(4); dev = DEVICE
N_CLEAN, N_ADV = (16, 8) if SMOKE else (512, 256)
if os.path.isdir(ADM_PATH): adm_dir = ADM_PATH
else:
    from huggingface_hub import snapshot_download; adm_dir = snapshot_download(ADM_PATH)
sd = torch.load(f'{adm_dir}/diffusion_pytorch_model.bin', map_location='cpu')
res = {}

def load(fixed):
    kw = dict(attn_norm_num_groups=32) if fixed else {}
    u = UNet2DModel.from_pretrained(ADM_PATH, torch_dtype=torch.float16, **kw).to(dev).eval()
    for p in u.parameters(): p.requires_grad_(False)
    mk = set(u.state_dict().keys())
    return u, sorted(set(sd) - mk), sorted(mk - set(sd))
U = {}
for name, fixed in [('asis', False), ('fixed', True)]:
    U[name], ignored, missing = load(fixed)
    res[f'{name}_ignored_ckpt_keys'] = ignored; res[f'{name}_missing_keys'] = missing
    log(name, 'ignored:', ignored, 'missing:', missing)
gk = 'mid_block.attentions.0.group_norm.weight'
if gk in U['fixed'].state_dict():
    res['fixed_gn_matches_ckpt'] = bool(torch.allclose(U['fixed'].state_dict()[gk].float().cpu(), sd[gk].float(), atol=1e-3, rtol=1e-3))
    res['ckpt_gn_weight_stats'] = [float(sd[gk].float().mean()), float(sd[gk].float().std())]

X, Y = load_cache()
cl_ids = split_ids('test')[:N_CLEAN]; ev_ids = split_ids('eval')
xc = to_tensor(X[cl_ids], dev)

# 2. eps-hat agreement at t=1
with torch.no_grad():
    ea = torch.cat([eps_hat(U['asis'], (xc[i:i + 32] * 2 - 1).half()) for i in range(0, len(xc), 32)])
    ef = torch.cat([eps_hat(U['fixed'], (xc[i:i + 32] * 2 - 1).half()) for i in range(0, len(xc), 32)])
cos = F.cosine_similarity(ea, ef, dim=1); rel = (ea - ef).norm(dim=1) / ef.norm(dim=1)
res['t1_cos_asis_vs_fixed'] = [float(cos.mean()), float(cos.min())]
res['t1_relL2_asis_vs_fixed'] = [float(rel.mean()), float(rel.max())]
log('t=1 cos(asis,fixed) mean/min', res['t1_cos_asis_vs_fixed'], 'rel L2 mean/max', res['t1_relL2_asis_vs_fixed'])

# 3. denoising MSE
betas = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64); abar = torch.cumprod(1 - betas, 0)
g = torch.Generator(device=dev); g.manual_seed(0)
mse = {}
for t in [1, 10, 50, 100, 200, 500]:
    a = float(abar[t]); out = {'asis': [], 'fixed': []}
    for i in range(0, len(xc), 32):
        x0 = xc[i:i + 32] * 2 - 1; e = torch.randn(x0.shape, generator=g, device=dev)
        xt = (a ** 0.5 * x0 + (1 - a) ** 0.5 * e).half()
        for nm in out:
            with torch.no_grad(): out[nm].append(((eps_hat(U[nm], xt, t) - e.flatten(1)) ** 2).mean(1))
    mse[t] = {nm: float(torch.cat(v).mean()) for nm, v in out.items()}; log('MSE t', t, mse[t])
res['denoise_mse'] = mse

# 4. CPOP AUROC on ResNet-50 PGD
clf = build_clf('resnet50', dev); ok = []
for i in range(0, len(ev_ids), 128):
    b = ev_ids[i:i + 128]
    with torch.no_grad(): ok.append(b[(clf(to_tensor(X[b], dev)).argmax(1).cpu().numpy() == Y[b])])
    if sum(map(len, ok)) >= N_ADV: break
elig = np.concatenate(ok)[:N_ADV]; adv = []
for i in range(0, len(elig), 32):
    b = elig[i:i + 32]; torch.manual_seed(image_seed(int(b[0]), 'g0|PGD'))
    atk = torchattacks.PGD(clf, eps=4 / 255, alpha=1 / 255, steps=20)
    adv.append(to_uint8(atk(to_tensor(X[b], dev), torch.from_numpy(Y[b]).to(dev))))
adv = np.concatenate(adv)
with torch.no_grad(): succ = (clf(to_tensor(adv, dev)).argmax(1).cpu().numpy() != Y[elig])
res['pgd_success'] = float(succ.mean()); ids_a = elig[succ]; xa = to_tensor(adv[succ], dev)

def auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))
sc = {}
for nm in U:
    c = np.concatenate([cpop_all(U[nm], xc[i:i + 32], cl_ids[i:i + 32])['cpop'] for i in range(0, len(xc), 32)])
    a = np.concatenate([cpop_all(U[nm], xa[i:i + 32], ids_a[i:i + 32])['cpop'] for i in range(0, len(xa), 32)])
    sc[nm] = (c, a); res[f'auroc_{nm}'] = auroc(-a, -c)
    res[f'cpop_{nm}_clean_mean'] = float(c.mean()); res[f'cpop_{nm}_adv_mean'] = float(a.mean())
res['spearman_clean'] = float(spearmanr(sc['asis'][0], sc['fixed'][0])[0])
res['spearman_adv'] = float(spearmanr(sc['asis'][1], sc['fixed'][1])[0])
res['n_clean'], res['n_adv'] = len(cl_ids), int(succ.sum())
log({k: v for k, v in res.items() if 'keys' not in k})
os.makedirs(OUTD, exist_ok=True)
json.dump(res, open(f'{OUTD}/g0_groupnorm.json', 'w'), indent=1); log('saved')
