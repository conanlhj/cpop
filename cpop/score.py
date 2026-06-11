"""Canonical reference implementation of CPOP and the baseline detectors used in the paper.

CPOP (Cosine Perturbation Orientation Probe) scores an image from the *local directional
stability* of a fixed pretrained diffusion denoising field -- without ever querying the
protected classifier:

    CPOP(x) = E_{delta ~ N(0, sigma^2 I)} [ 1 - cos( s_theta(x, t), s_theta(x+delta, t) ) ]

where s_theta(., t) is the diffusion model's denoising output at a low timestep t. Adversarial
inputs sit off the natural-image manifold where the restoring (normal) component of the score
dominates and freezes its direction, giving a *small* CPOP; the detector score is therefore

    D(x) = - CPOP(x)        (higher D => more suspicious)

This module is environment-free; the per-table experiment scripts under ``experiments/`` show
how it was driven on ImageNet/CIFAR-10 (they reproduce the exact numbers in the paper).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def cpop(images: torch.Tensor, unet, t: float = 1.0, sigma: float = 0.005, K: int = 8) -> torch.Tensor:
    """Compute CPOP for a batch of images in [0, 1].

    Args:
        images: (B, 3, H, W) tensor in [0, 1].
        unet:   a diffusion model whose ``unet(x, t).sample`` returns the denoising output for
                inputs ``x`` in [-1, 1] (e.g. a ``diffusers`` ``UNet2DModel`` such as ADM-ImageNet
                or DDPM-CIFAR10). Weights are frozen; the probe is never adapted to any classifier.
        t:      diffusion timestep (noise level the model assumes). A continuous value is accepted.
        sigma:  std of the isotropic pixel-space probe perturbation delta.
        K:      number of Monte Carlo probe samples.

    Returns:
        (B,) tensor of CPOP values. The detection score is ``-cpop(...)``.
    """
    B = images.shape[0]
    x = (images * 2 - 1).to(next(unet.parameters()).dtype)
    tt = torch.full((B,), float(t), device=images.device, dtype=x.dtype)
    base = unet(x, tt).sample[:, :3].reshape(B, -1).float()
    acc = torch.zeros(B, device=images.device)
    for _ in range(K):
        probe = unet(x + torch.randn_like(x) * sigma, tt).sample[:, :3].reshape(B, -1).float()
        acc += 1.0 - F.cosine_similarity(base, probe, dim=1, eps=1e-12)
    return acc / K


@torch.no_grad()
def detector_score(images: torch.Tensor, unet, **kw) -> torch.Tensor:
    """Suspiciousness score D(x) = -CPOP(x); calibrate one threshold on clean data."""
    return -cpop(images, unet, **kw)


# --------------------------------------------------------------------------------------------------
# Baseline detectors used for comparison (classifier-coupled; require access to the protected model)
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
