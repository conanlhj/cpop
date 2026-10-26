"""Revised evaluation protocol (R1) — shared pieces.

Paths (environment variables): EXT = data root (outputs go to EXT/r1, the ImageNet cache to EXT/imagenet_val_256),
ADM_PATH = ADM checkpoint (local diffusers directory or Hugging Face id).
Changes vs. the pipeline of the original submission (tag v1-submission):
  * full ILSVRC2012 val (50k) via guided-diffusion center_crop_arr, ground-truth labels
  * class-stratified disjoint splits (ref / cal / test / eval), every row keyed by image_id
  * classifier normalisation STD fixed to torchvision's [0.229, 0.224, 0.225]
  * per-image deterministic probe noise (seed = f(image_id, variant)) -> reproducible scores
  * one ADM pass yields cpop, raw_var, score_norm (+ min field norm diagnostics)
"""
import os, hashlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models as tvm

EXT = os.environ.get('EXT', '/ext'); CODE = os.path.dirname(os.path.abspath(__file__))
CACHE = f'{EXT}/imagenet_val_256/val256_uint8.npy'; LABELS = f'{EXT}/imagenet_val_256/labels.npy'
SPLITS = f'{EXT}/r1/splits.csv'
SMOKE = os.environ.get('SMOKE', '0') == '1'           # smoke runs write to a separate tree
OUTD = f'{EXT}/r1_smoke' if SMOKE else f'{EXT}/r1'
DEVICE = os.environ.get('DEVICE', 'cuda')
ADM_PATH = os.environ.get('ADM_PATH', 'xutongda/adm_imagenet_256x256_unconditional')
T_STEP, SIGMA, K = 1, 0.005, 8                      # paper default probe configuration
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]   # fixed (submitted code had G/B swapped)
CLF_RES = 224

ARCHS = {
    'resnet50':           (tvm.resnet50,           tvm.ResNet50_Weights.IMAGENET1K_V2,          'CNN'),
    'vgg16':              (tvm.vgg16,              tvm.VGG16_Weights.IMAGENET1K_V1,             'CNN'),
    'densenet121':        (tvm.densenet121,        tvm.DenseNet121_Weights.IMAGENET1K_V1,       'CNN'),
    'mobilenet_v3_large': (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2,'CNN'),
    'convnext_tiny':      (tvm.convnext_tiny,      tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,     'CNN'),
    'vit_b_16':           (tvm.vit_b_16,           tvm.ViT_B_16_Weights.IMAGENET1K_V1,          'Transformer'),
    'swin_t':             (tvm.swin_t,             tvm.Swin_T_Weights.IMAGENET1K_V1,            'Transformer'),
    'maxvit_t':           (tvm.maxvit_t,           tvm.MaxVit_T_Weights.IMAGENET1K_V1,          'Transformer'),
}
ARCHS4 = ['resnet50', 'densenet121', 'vit_b_16', 'swin_t']   # Table 3/6 subset (as submitted)

# ---------------------------------------------------------------- data
def load_cache(mmap=True):
    return np.load(CACHE, mmap_mode='r' if mmap else None), np.load(LABELS).astype(np.int64)

def to_tensor(u8, device):
    """uint8 (B,H,W,3) -> float [0,1] (B,3,H,W)."""
    return torch.from_numpy(np.ascontiguousarray(u8)).to(device).permute(0, 3, 1, 2).float().div_(255.)

def to_uint8(x):
    """float [0,1] (B,3,H,W) -> uint8 (B,H,W,3), round-to-nearest (what a deployed system would store)."""
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

def read_splits():
    import pandas as pd
    return pd.read_csv(SPLITS)

def split_ids(name, smoke_n=int(os.environ.get('SMOKE_N', '40'))):
    s = read_splits(); ids = s[s.split == name].image_id.values.astype(np.int64)
    return ids[:smoke_n] if SMOKE else ids

def parts(ids, size):
    for i in range(0, len(ids), size): yield i // size, ids[i:i + size]

def merge_parts(pdir, out):
    """concatenate part_*.npz (same keys) into one npz."""
    fs = sorted(f for f in os.listdir(pdir) if f.startswith('part_') and f.endswith('.npz') and '.tmp' not in f)
    ds = [np.load(os.path.join(pdir, f)) for f in fs]
    atomic_savez(out, **{k: np.concatenate([d[k] for d in ds]) for k in ds[0].files})

# ---------------------------------------------------------------- classifiers
class Clf(nn.Module):
    """[0,1] 256x256 input -> bilinear 224 -> ImageNet normalisation -> torchvision model."""
    def __init__(self, m):
        super().__init__(); self.m = m
        self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1))
    def forward(self, x):
        x = F.interpolate(x, size=CLF_RES, mode='bilinear', align_corners=False)
        return self.m((x - self.mean) / self.std)

def build_clf(name, device):
    ctor, w, _ = ARCHS[name]
    m = ctor(weights=w).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return Clf(m).to(device).eval()

class FeatHook:
    """penultimate features = input of the last nn.Linear (uniform across architectures)."""
    def __init__(self, clf):
        lins = [m for m in clf.modules() if isinstance(m, nn.Linear)]
        self.feat = None; self.h = lins[-1].register_forward_pre_hook(self._hook)
    def _hook(self, mod, inp): self.feat = inp[0].detach().flatten(1)
    def remove(self): self.h.remove()

# ---------------------------------------------------------------- Feature Squeezing (Xu et al. 2018)
def bitdepth(x, b): L = 2 ** b - 1; return torch.round(x * L) / L
def medianf(x, k):
    p = k // 2; xp = F.pad(x, (p, p, p, p), mode='reflect'); pa = xp.unfold(2, k, 1).unfold(3, k, 1)
    return pa.contiguous().view(*pa.shape[:4], k * k).median(-1)[0]
SQUEEZERS = {'bit1': lambda x: bitdepth(x, 1), 'bit2': lambda x: bitdepth(x, 2), 'bit3': lambda x: bitdepth(x, 3),
             'bit5': lambda x: bitdepth(x, 5), 'median3': lambda x: medianf(x, 3),
             'median5': lambda x: medianf(x, 5), 'avgpool3': lambda x: F.avg_pool2d(x, 3, 1, 1)}

@torch.no_grad()
def clf_pass(clf, hook, x):
    """one call per image batch: logits-derived quantities, penultimate features, FS L1 per squeezer."""
    logits = clf(x); feat = hook.feat.half().cpu().numpy(); p = F.softmax(logits.float(), 1)
    out = {'pred': logits.argmax(1).cpu().numpy().astype(np.int16),
           'maxprob': p.max(1)[0].cpu().numpy().astype(np.float32), 'feat': feat}
    for nm, sq in SQUEEZERS.items():
        q = F.softmax(clf(sq(x)).float(), 1); out['fs_' + nm] = (p - q).abs().sum(1).cpu().numpy().astype(np.float32)
    out['fs_common'] = np.maximum(out['fs_bit3'], out['fs_avgpool3'])   # standard FS (ImageNet config approx.)
    return out

# ---------------------------------------------------------------- CPOP probe
def load_unet(device, dtype=torch.float16):
    """attn_norm_num_groups=32: without it diffusers 0.29.2 drops mid_block.attentions.0.group_norm (the submitted
    runs did); G0 shows the effect is negligible (AUROC 0.91598 both, Spearman 0.99999) -> load the full checkpoint."""
    from diffusers import UNet2DModel
    u = UNet2DModel.from_pretrained(ADM_PATH, torch_dtype=dtype, attn_norm_num_groups=32).to(device).eval()
    for p in u.parameters(): p.requires_grad_(False)
    return u

def image_seed(image_id, variant):
    h = hashlib.sha256(f'{int(image_id)}|{variant}'.encode()).digest()
    return int.from_bytes(h[:8], 'little') & ((1 << 63) - 1)

def probe_noise(ids, variant, k, shape, device):
    """deterministic per-image probe noise, (k, B, C, H, W). Independent of batch composition."""
    out = torch.empty((k, len(ids)) + tuple(shape), device=device)
    for j, i in enumerate(ids):
        g = torch.Generator(device=device); g.manual_seed(image_seed(i, variant))
        out[:, j] = torch.randn((k,) + tuple(shape), generator=g, device=device)
    return out

@torch.no_grad()
def eps_hat(unet, x, t_step=T_STEP):
    """ADM epsilon prediction (first 3 of 6 output channels; the other 3 are learned variance)."""
    t = torch.full((x.shape[0],), t_step, device=x.device, dtype=torch.long)
    return unet(x, t).sample[:, :3].reshape(x.shape[0], -1).float()

@torch.no_grad()
def cpop_all(unet, imgs, ids, variant='main', k=K, sigma=SIGMA, t_step=T_STEP, probe_chunk=None):
    """imgs: float [0,1] (B,3,256,256). Input scaling x = 2*img - 1 (ADM convention); probes are NOT clipped.
    Returns dict of per-image numpy arrays:
      cpop      = mean_k (1 - cos(e(x), e(x + sigma*n_k)))        [raw; LOW = adversarial-like]
      raw_var   = mean_k ||e(x+sigma n_k) - e(x)||^2 / sigma^2     [unnormalised variation]
      score_norm= ||e(x)||                                          [field magnitude]
      min_pnorm = min_k ||e(x + sigma n_k)||                        [small-norm diagnostic]
    """
    dt = next(unet.parameters()).dtype
    x = (imgs * 2 - 1).to(dt); B = x.shape[0]
    u = eps_hat(unet, x, t_step); n = probe_noise(ids, variant, k, x.shape[1:], x.device).to(dt)
    acc_cos = torch.zeros(B, device=x.device); acc_raw = torch.zeros(B, device=x.device)
    mn = torch.full((B,), float('inf'), device=x.device)
    for j in range(k):
        v = eps_hat(unet, x + n[j] * sigma, t_step)
        acc_cos += 1.0 - F.cosine_similarity(u, v, dim=1)
        acc_raw += ((v - u) ** 2).sum(1) / sigma ** 2
        mn = torch.minimum(mn, v.norm(dim=1))
    return {'cpop': (acc_cos / k).cpu().numpy(), 'raw_var': (acc_raw / k).cpu().numpy(),
            'score_norm': u.norm(dim=1).cpu().numpy(), 'min_pnorm': mn.cpu().numpy()}

# ---------------------------------------------------------------- misc
def atomic_savez(path, **kw):
    tmp = path + '.tmp.npz'; np.savez(tmp, **kw); os.replace(tmp, path)

def log(*a):
    print(*a, flush=True)
