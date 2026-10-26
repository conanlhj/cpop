"""S0 — deployment cost of the probe (AE #7): latency / throughput / peak memory on the local GPU.
Reports CPOP (1 base + K probe evaluations, fp16, 256x256) for batch B and K, and plain classifier forwards for reference.
out: OUTD/bench.csv"""
import time, numpy as np, pandas as pd, torch
from common import *

dev = DEVICE; unet = load_unet(dev, torch.float16 if dev == 'cuda' else torch.float32)
rows = []
def timeit(fn, reps):
    fn(); torch.cuda.synchronize() if dev == 'cuda' else None          # warm-up
    if dev == 'cuda': torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(reps): fn()
    if dev == 'cuda': torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps, (torch.cuda.max_memory_allocated() / 2 ** 30 if dev == 'cuda' else np.nan)

for B in ([1, 8, 32] if not SMOKE else [1, 2]):
    x = torch.rand(B, 3, 256, 256, device=dev); ids = np.arange(B)
    for k in [0, 4, 8]:
        try:
            if k == 0: fn = lambda: eps_hat(unet, (x * 2 - 1).to(next(unet.parameters()).dtype))
            else: fn = lambda k=k: cpop_all(unet, x, ids, k=k)
            sec, mem = timeit(fn, 3 if SMOKE else 10)
            rows.append(dict(model='ADM-CPOP', batch=B, K=k, unet_evals=1 + k, sec_per_batch=sec, ms_per_img=1e3 * sec / B,
                             img_per_s=B / sec, peak_mem_gb=mem)); log(rows[-1])
        except torch.cuda.OutOfMemoryError as e:
            log('OOM', B, k); torch.cuda.empty_cache()
for name in ['resnet50', 'vit_b_16']:
    clf = build_clf(name, dev)
    for B in ([1, 32] if not SMOKE else [1]):
        x = torch.rand(B, 3, 256, 256, device=dev)
        with torch.no_grad(): sec, mem = timeit(lambda: clf(x), 3 if SMOKE else 20)
        rows.append(dict(model=name, batch=B, K=None, unet_evals=0, sec_per_batch=sec, ms_per_img=1e3 * sec / B, img_per_s=B / sec, peak_mem_gb=mem)); log(rows[-1])
    del clf
props = torch.cuda.get_device_properties(0).name if dev == 'cuda' else 'cpu'
df = pd.DataFrame(rows); df['device'] = props; os.makedirs(OUTD, exist_ok=True); df.to_csv(f'{OUTD}/bench.csv', index=False); log(df.to_string())
