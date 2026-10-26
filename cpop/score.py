"""Reference implementation of CPOP and the simplified feature-space baselines.

CPOP (Cosine Perturbation Orientation Probe) scores an image from a fixed pretrained diffusion
model alone -- it never queries the protected classifier:

    CPOP(x) = E_{delta ~ N(0, sigma^2 I)} [ 1 - cos( v(x), v(x + delta) ) ],   v = eps_hat_theta(., t)

where eps_hat_theta is the model's noise prediction at a small integer timestep t. To leading order
CPOP is (sigma^2 / 2) times the squared Frobenius norm of the Jacobian of the normalized field
v / ||v||. Dense, high-frequency perturbations -- gradient-based L_inf attacks and benign noise of
the same kind alike -- tend to align the noise estimate with the perturbation, which the probe
then barely rotates, giving a *small* CPOP. The detector score is therefore

    D(x) = - CPOP(x)        (higher D => more suspicious)

This module is environment-free; ``experiments/`` holds the full evaluation protocol of the paper
(splits, attacks, controls, baselines, analysis), and ``experiments/common.py`` its batched,
seeded variant of the same computation.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

ADM_ID = "xutongda/adm_imagenet_256x256_unconditional"   # diffusers conversion of guided-diffusion 256x256_diffusion_uncond


def load_adm(path: str = ADM_ID, device: str = "cuda", dtype=torch.float16):
    """Load the default probe. ``attn_norm_num_groups=32`` is required: without it, diffusers 0.29
    silently drops the mid-block attention GroupNorm weights of this conversion."""
    from diffusers import UNet2DModel
    unet = UNet2DModel.from_pretrained(path, torch_dtype=dtype, attn_norm_num_groups=32).to(device).eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    return unet


@torch.no_grad()
def cpop(images: torch.Tensor, unet, t: int = 1, sigma: float = 0.005, K: int = 8) -> torch.Tensor:
    """Compute CPOP for a batch of images in [0, 1].

    Args:
        images: (B, 3, H, W) tensor in [0, 1].
        unet:   an epsilon-prediction diffusion model whose ``unet(x, t).sample`` holds the noise
                prediction in its first three channels for inputs ``x`` in [-1, 1] (e.g. ``load_adm()``
                or ``google/ddpm-cifar10-32``). Weights are frozen; nothing is tuned per classifier.
        t:      integer timestep index of the 1000-step schedule (t=1: sqrt(1 - abar_t) = 0.0148).
        sigma:  std of the isotropic pixel-space probe perturbation delta.
        K:      number of Monte Carlo probe samples.

    Returns:
        (B,) tensor of CPOP values. The detection score is ``-cpop(...)``.
    """
    B = images.shape[0]
    x = (images * 2 - 1).to(next(unet.parameters()).dtype)
    tt = torch.full((B,), int(t), device=images.device, dtype=torch.long)
    base = unet(x, tt).sample[:, :3].reshape(B, -1).float()
    acc = torch.zeros(B, device=images.device)
    for _ in range(K):
        probe = unet(x + torch.randn_like(x) * sigma, tt).sample[:, :3].reshape(B, -1).float()  # not clipped
        acc += 1.0 - F.cosine_similarity(base, probe, dim=1, eps=1e-12)
    return acc / K


@torch.no_grad()
def detector_score(images: torch.Tensor, unet, **kw) -> torch.Tensor:
    """Suspiciousness score D(x) = -CPOP(x); calibrate one threshold on clean data."""
    return -cpop(images, unet, **kw)


# --------------------------------------------------------------------------------------------------
# Simplified classifier-coupled baselines (require access to the protected model). The paper's
# numbers come from experiments/s5_feature_baselines.py, which also has class-conditional variants.
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def feature_squeezing(images: torch.Tensor, clf) -> torch.Tensor:
    """Feature Squeezing (Xu et al. 2018), 'common' config: 3-bit depth + 3x3 average pooling.
    Returns the max L1 softmax change across squeezers (higher => adversarial)."""
    p = F.softmax(clf(images), 1)
    pb = F.softmax(clf(torch.round(images * 7) / 7), 1)               # 3-bit colour depth
    ps = F.softmax(clf(F.avg_pool2d(images, 3, 1, 1)), 1)             # 3x3 average pooling
    return torch.maximum((p - pb).abs().sum(1), (p - ps).abs().sum(1))


@torch.no_grad()
def lid_mle(query_feats: torch.Tensor, ref_feats: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Local Intrinsic Dimensionality (Ma et al. 2018), MLE estimator in a feature space
    (higher => adversarial)."""
    q, r = query_feats.float(), ref_feats.float()
    d = torch.cdist(q, r)
    rr, _ = d.topk(min(k, r.shape[0]), dim=1, largest=False)
    rr = rr.clamp_min(1e-12)
    rk = rr[:, -1:].clamp_min(1e-12)
    return -1.0 / (torch.log(rr / rk).mean(1))


class Mahalanobis:
    """Single-Gaussian Mahalanobis distance on penultimate features (Lee et al. 2018, simplified),
    with shrinkage covariance. Higher => adversarial. Fitted on a clean reference set."""
    def __init__(self, ref_feats: torch.Tensor):
        ref = ref_feats.float()
        self.mu = ref.mean(0, keepdim=True)
        d = ref - self.mu
        cov = (d.t() @ d) / ref.shape[0]
        cov += (cov.diagonal().mean() * 1e-2) * torch.eye(cov.shape[0], device=cov.device)
        self.prec = torch.linalg.pinv(cov)

    @torch.no_grad()
    def score(self, q: torch.Tensor) -> torch.Tensor:
        d = q.float() - self.mu
        return (d @ self.prec * d).sum(1).clamp_min(0).sqrt()


class KernelDensity:
    """Feature-space kernel density (Feinman et al. 2017 style); higher => adversarial (lower
    density). Bandwidth = median pairwise distance of the clean reference set."""
    def __init__(self, ref_feats: torch.Tensor):
        self.ref = ref_feats.float()
        dd = torch.cdist(self.ref[:200], self.ref[:200])
        self.h2 = 2 * (dd.median().item() ** 2 + 1e-9)

    @torch.no_grad()
    def score(self, q: torch.Tensor) -> torch.Tensor:
        import math
        d2 = torch.cdist(q.float(), self.ref) ** 2
        log_kd = torch.logsumexp(-d2 / self.h2, dim=1) - math.log(self.ref.shape[0])
        return -log_kd
