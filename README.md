# CPOP — Classifier-Agnostic Diffusion Probing for Noise-Like Adversarial Perturbations

Reference code and per-sample result data for the paper
**"Classifier-Agnostic Diffusion Probing for Detecting Noise-Like Adversarial Perturbations"**
(revised version; the original submission was titled "Portable Operating Points via
Classifier-Agnostic Diffusion Probing for Adversarial Example Detection").

CPOP (**C**osine **P**erturbation **O**rientation **P**robe) scores an input from the image and a
fixed pretrained diffusion model alone; it never queries the protected classifier:

```
CPOP(x) = E_{δ ~ N(0, σ²I)} [ 1 − cos( v(x), v(x+δ) ) ],   v = ε̂_θ(·, t)        detector:  D(x) = −CPOP(x)
```

To leading order CPOP is ½σ² times the squared Jacobian norm of the normalized field v/‖v‖.
It responds to dense, high-frequency perturbations: gradient-based L∞ attacks are flagged, and so
is benign noise matched to them, whereas blur, JPEG compression, and smooth or blocky attacks are
not. Because the score does not depend on the classifier, one clean calibration serves every
classifier by construction.

## Minimal usage

```python
import torch
from cpop import load_adm, detector_score

unet = load_adm(device="cuda")                       # ADM 256x256 unconditional, fp16, frozen
imgs = torch.rand(8, 3, 256, 256, device="cuda")     # images in [0, 1]
score = detector_score(imgs, unet, t=1, sigma=0.005, K=8)   # higher => more suspicious
```

`load_adm` sets `attn_norm_num_groups=32`; without it, `diffusers` 0.29 silently drops the
mid-block attention normalization weights of this checkpoint.

## Repository layout

```
cpop/score.py           # CPOP, the ADM loader, and simplified feature-space baselines
experiments/            # the full evaluation protocol of the paper (s* = GPU steps, a* = CPU analysis)
experiments/run_all.sh  # Docker runner for every step
data/r1/                # per-sample scores and metadata behind every table and figure
data/fig1_norm_dependence_v1.csv   # data of Figure 1 (unchanged from the original submission)
```

## Protocol

- **Data.** ILSVRC2012 validation set (50k), center-cropped to 256×256 as in guided-diffusion
  (`prep_cache.py`), ground-truth labels, class-stratified disjoint splits
  `ref / cal / test / eval` = 20k / 10k / 10k / 3k (`make_splits.py`, `data/r1/splits.csv`).
- **Probe.** ADM unconditional 256×256 (`xutongda/adm_imagenet_256x256_unconditional`), fp16,
  integer timestep t=1, σ=0.005, K=8, x = 2·img − 1, probes not clipped, per-image deterministic
  probe noise.
- **Classifiers.** Eight torchvision ImageNet models; bilinear 256→224 and torchvision
  normalization. Each classifier is attacked on the `eval` images it labels correctly; an attack
  succeeds when the 8-bit adversarial image is misclassified with respect to the ground truth.
- **Operating regimes.** R1 (one numerical threshold shared across classifiers), R2 (clean-only
  recalibration per classifier), access-free (R1 = R2 for input-only scores).

## Paper results → script → data

All `a*` scripts read only `data/r1/` and run on a CPU in seconds. Running them on the released
data reproduces `data/r1/analysis/*.md` byte for byte.

| Paper result | GPU step | Analysis (output section) |
|---|---|---|
| Table 2 — alignment of the noise estimate with the perturbation | `s9_alignment.py` | `a4_probe.py` |
| Table 3 — detection under R1 / R2 / access-free | `s1`, `s2`, `s3`, `s5` | `a1_analysis.py` (§1) |
| Table 4 — calibration-set size | `s3_score.py clean` | `a1_analysis.py` (§2) |
| Table 5 — cross-architecture AUROC | `s3`, `s5` | `a1_analysis.py` (§10) |
| Table 6 — Feature Squeezing squeezers | `s1`, `s2` | `a1_analysis.py` (§11) |
| Table 7 — attacks and benign controls | `s2_attack.py --kind CTRL_*` | `a3_controls.py`, `a1_analysis.py` (§6) |
| Table 8 — attack family | `s2_attack.py` | `a1_analysis.py` (§6) |
| Table 9 — ε-controlled cohort | `s2_attack.py --eps 2/4/8` | `a1_analysis.py` (§5) |
| Table 10 — CIFAR-10 | `s7_cifar.py`, `s5 --dataset cifar` | `a2_cifar.py` |
| Table 11 — probe transfer across image priors | `s8_probe_k.py probe` | `a4_probe.py` |
| Table 12 — input-only scores | `s3_score.py` | `a1_analysis.py` (§9, §13) |
| Table 13 — probe scale σ, timestep t, Monte Carlo count K | `s3_score.py sigma tgrid`, `s8_probe_k.py k` | `a1_analysis.py` (§8), `a4_probe.py` |
| Table 14 — system-level effect | — | `a1_analysis.py` (§4) |
| Table 15 — adaptive attacks | `s6_adaptive.py` | `data/r1/adaptive/*.npz` |
| Cost (latency, memory) | `s0_bench.py` | `data/r1/bench.csv` |
| Figures 2 and 3 | — | `make_figures.py data/r1/analysis/a1_tables.json out/` |
| ADM GroupNorm check | `g0_groupnorm_check.py` | `data/r1/g0_groupnorm.json` |

Released `.npz` files keep every per-image score and label; adversarial images and penultimate
features are omitted for size.

## Reproducing from scratch

```bash
pip install -r requirements.txt          # or build a CUDA image with these packages
export DATA=/path/to/workdir IMAGENET=/path/to/ILSVRC2012 IMAGE=your-image:tag
experiments/run_all.sh prep              # ImageNet cache, splits, GroupNorm check
experiments/run_all.sh all               # smoke test, then all GPU steps and the analysis
```

The runs for the paper used one RTX 4090.

## Changes from the original submission

The code released with the original submission is kept under the tag
[`v1-submission`](../../tree/v1-submission). The revision rebuilt the evaluation:

- disjoint calibration and test splits on the full validation set with ground-truth labels
  (the original clean set reused the sources of successful attacks);
- classifier normalization fixed to torchvision's standard deviations (the original code swapped
  the G and B values; predictions agree on 99.2–100% of images, and CPOP does not use these
  constants);
- the full ADM checkpoint is loaded (see `load_adm`; the effect on CPOP is a Spearman correlation
  of 0.99999 and less than 1e-4 in AUROC);
- added benign controls, three operating regimes, clean-only recalibrated feature-space
  baselines, an ε-controlled cohort, system-level metrics, cost measurements, probe-transfer and
  probe-configuration studies, an alignment diagnostic, and a threshold-targeted adaptive attack.

## Scope

CPOP is a pre-filter for noise-like perturbations, with non-adaptive gradient-based L∞ attacks as
its main use case. It also flags benign noise of the same kind, it passes smooth and blocky
perturbations, and a detector-aware attacker evades it completely. When clean data from the
protected classifier are available, recalibrated feature-space detectors are more sensitive.

## Citation

A BibTeX entry will be added on publication.

## License

MIT — see [LICENSE](LICENSE).
