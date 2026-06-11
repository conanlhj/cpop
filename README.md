# CPOP — Classifier-Agnostic Adversarial-Example Detection via Diffusion Score Geometry

Reference code and result data for the paper
**"Portable Operating Points via Classifier-Agnostic Diffusion Probing for Adversarial Example Detection."**

CPOP (**C**osine **P**erturbation **O**rientation **P**robe) detects adversarial examples from the
**local directional stability** of a fixed pretrained diffusion denoising field, **without ever
querying the protected classifier**:

```
CPOP(x) = E_{δ ~ N(0, σ²I)} [ 1 − cos( s_θ(x, t), s_θ(x+δ, t) ) ]      detector:  D(x) = −CPOP(x)
```

Because the score comes from the image and the diffusion prior alone, **one calibrated threshold
transfers across classifiers** (a portable operating point), unlike classifier-coupled detectors
whose threshold silently mis-fires after a model swap.

The portable, environment-free implementation is in [`cpop/score.py`](cpop/score.py) (CPOP +
the baseline detectors FS, LID, Mahalanobis, KD).

## Minimal usage

```python
import torch
from diffusers import UNet2DModel
from cpop import cpop, detector_score

unet = UNet2DModel.from_pretrained("path/to/adm-imagenet").to("cuda").eval()  # frozen probe
imgs = torch.rand(8, 3, 256, 256, device="cuda")          # images in [0, 1]
score = detector_score(imgs, unet, t=1.0, sigma=0.005, K=8)   # higher => more suspicious
```

## Repository layout

```
cpop/score.py            # canonical CPOP + baselines (FS, LID, Mahalanobis, KD)
experiments/             # the scripts that produced each paper table/figure (as run)
data/                    # per-sample / summary CSVs backing every table and figure
```

## Paper results → script → data

| Paper result | Script | Data |
|---|---|---|
| Operating-point transfer (portability) | `experiments/operating_point_transfer.py` | `data/p1_expE_*.csv` (+ `p1_expA_cross_arch.csv`) |
| Cross-architecture stability | `experiments/cross_architecture.py` | `data/p1_expA_*.csv` |
| Attack generalization + Mahalanobis/KD baselines | `experiments/attack_generalization.py` | `data/p1_expB2_*.csv` |
| Second dataset (CIFAR-10) | `experiments/cifar10.py` | `data/p1_cifar_*.csv` |
| Is the angular normalization necessary? | `experiments/diffusion_baselines.py` | `data/p1_expI_*.csv` |
| Probe transfer (image-prior property) | — | `data/p1_expF_*.csv` |
| Feature-Squeezing parameter sweep | `experiments/fs_sweep.py` | `data/p1_expD_fs_sanity.csv` |
| Hyperparameter / noise-scale ablation | `experiments/ablation_noise.py` | `data/p1_expH_noise_ablation.csv` |
| Probe-scale / timestep analysis | `experiments/ablation_timestep.py` | (printed; see paper §5) |
| Adaptive, detector-aware attack | `experiments/adaptive.py` | `data/p1_expJ15_adaptive.csv` |
| Two-regime behavior (theory figure) | — | `data/theorem2_r_dependence.csv` |
| Score distributions (clean/adv/OOD) | — | `data/imagenet_ood_3class.csv` |
| All figures | `experiments/make_figures.py` | the CSVs above |

The matched-FPR sensitivity curve and the operating-point transfer are computed from the
per-sample scores in `data/p1_expA_cross_arch.csv` and `data/p1_expB2_attacks.csv`.

## Reproducing

- **From data (no GPU):** every number/figure can be re-derived from the CSVs in `data/`
  (`experiments/make_figures.py` regenerates the figures).
- **From scratch (GPU):** the `experiments/` scripts were run inside a CUDA Docker image with
  `torch 2.3.1`, `diffusers 0.29.2`, `torchattacks` (see `requirements.txt`). They are provided
  **as run** for transparency and contain environment-specific paths (e.g. `/workspace/...`,
  a local ADM checkpoint, an ImageNet-val folder) and a `src/` import for `experiments/utils`;
  adjust these to your setup. The portable method itself has no such dependencies — see
  `cpop/score.py`.

## Scope (stated plainly)

CPOP targets **non-adaptive, gradient-based L∞** evasion attacks against image classifiers in
heterogeneous-classifier deployments. It is a **pre-filter**, not a certified defense: a strong
adaptive, detector-aware attacker can evade it (paper §5), and minimal-norm (CW) / black-box
(Square) attacks lie outside the threat model.

## Citation

A BibTeX entry will be added on publication.

## License

MIT — see [LICENSE](LICENSE).
