"""class-stratified disjoint splits of ILSVRC2012 val (50/class): ref 20 | cal 10 | test 10 | eval 3 | reserve 7."""
import os, numpy as np, pandas as pd
from common import LABELS, SPLITS, log
PLAN = [('ref', 20), ('cal', 10), ('test', 10), ('eval', 3), ('reserve', 7)]
y = np.load(LABELS).astype(int); rng = np.random.default_rng(20261020); rows = []
for c in range(1000):
    ids = np.where(y == c)[0]; assert len(ids) == 50, (c, len(ids)); ids = rng.permutation(ids); o = 0
    for name, n in PLAN:
        rows += [(int(i), c, name) for i in ids[o:o + n]]; o += n
df = pd.DataFrame(rows, columns=['image_id', 'label', 'split']).sort_values('image_id')
# image_id = 0-based index into the cache; file = ILSVRC2012_val_{image_id+1:08d}.JPEG
os.makedirs(os.path.dirname(SPLITS), exist_ok=True); df.to_csv(SPLITS, index=False)
log(df.split.value_counts().to_dict()); log('first-1000 overlap by split:', df[df.image_id < 1000].split.value_counts().to_dict())
