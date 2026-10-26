"""ILSVRC2012 val (50k JPEG) -> guided-diffusion center_crop_arr 256x256 uint8 cache + GT labels.
Output (OUT dir): val256_uint8.npy (50000,256,256,3) memmap, labels.npy (torchvision index), wnids.npy, DONE.
Resumable: chunks marked in progress file."""
import os, sys, numpy as np, scipy.io as sio
from multiprocessing import Pool
from PIL import Image
SRC=os.environ.get('SRC','/im')   # ILSVRC2012 root: val_full/*.JPEG + ILSVRC2012_devkit_t12/; OUT=os.environ.get('OUT','/ext/imagenet_val_256'); N=50000; S=256
def center_crop_arr(pil, s=S):  # guided-diffusion/improved-diffusion image_datasets.py
    while min(*pil.size) >= 2*s: pil = pil.resize(tuple(x//2 for x in pil.size), resample=Image.BOX)
    sc = s/min(*pil.size); pil = pil.resize(tuple(round(x*sc) for x in pil.size), resample=Image.BICUBIC)
    a = np.array(pil); y=(a.shape[0]-s)//2; x=(a.shape[1]-s)//2; return a[y:y+s, x:x+s]
def load(i): return center_crop_arr(Image.open(f'{SRC}/val_full/ILSVRC2012_val_{i+1:08d}.JPEG').convert('RGB'))
if __name__=='__main__':
    os.makedirs(OUT, exist_ok=True)
    m=sio.loadmat(f'{SRC}/ILSVRC2012_devkit_t12/data/meta.mat',squeeze_me=True)['synsets']
    id2w={int(s[0]):s[1] for s in m if int(s[0])<=1000}; wn=sorted(id2w.values()); w2i={w:i for i,w in enumerate(wn)}
    gt=np.array([w2i[id2w[int(l)]] for l in open(f'{SRC}/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')],dtype=np.int16)
    assert len(gt)==N and gt[0]==65
    np.save(f'{OUT}/labels.npy',gt); np.save(f'{OUT}/wnids.npy',np.array(wn))
    fn=f'{OUT}/val256_uint8.npy'
    arr=np.lib.format.open_memmap(fn,mode='r+' if os.path.exists(fn) else 'w+',dtype=np.uint8,shape=(N,S,S,3))
    prog=f'{OUT}/progress.txt'; done=set(int(l) for l in open(prog)) if os.path.exists(prog) else set()
    CH=1000
    with Pool(28) as p:
        for c in range(0,N,CH):
            if c in done: continue
            arr[c:c+CH]=np.stack(p.map(load,range(c,min(N,c+CH)),chunksize=8)); arr.flush()
            open(prog,'a').write(f'{c}\n'); print('chunk',c,flush=True)
    open(f'{OUT}/DONE','w').write('ok\n'); print('DONE',flush=True)
