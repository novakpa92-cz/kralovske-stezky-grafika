"""JEDNA vrstva: vnitřní hrana rukávů (k trupu) srovnaná do plynulé čáry – pryč zadní díl rukávu, cípy a schody.
python3 nastroje/rukav_vnitrni_hrana.py <vrstva> y0 y1 [okno=41]"""
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import median_filter, gaussian_filter1d
D = Path("img/postava"); n = sys.argv[1]; y0, y1 = int(sys.argv[2]), int(sys.argv[3]); OK = int(sys.argv[4]) if len(sys.argv) > 4 else 41
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA")); A = k[..., 3] > 100
def useky(r):
    x = np.where(r)[0]
    if not len(x): return []
    c = np.where(np.diff(x) > 1)[0]; return list(zip(np.r_[x[0], x[c + 1]], np.r_[x[c], x[-1]]))
for sd in (-1, 1):
    ys, e = [], []
    for y in range(y0, y1):
        s = [u for u in useky(A[y]) if (u[1] < 512 - 60 if sd < 0 else u[0] > 512 + 60)]
        if not s: continue
        u = s[-1] if sd < 0 else s[0]
        ys.append(y); e.append(u[1] if sd < 0 else u[0])
    if len(e) < 10: continue
    e = np.array(e, float); m = median_filter(e, OK, mode="nearest")
    m = gaussian_filter1d(np.minimum(e, m) if sd < 0 else np.maximum(e, m), 4)
    for y, lim in zip(ys, m):
        s = [u for u in useky(A[y]) if (u[1] < 512 - 60 if sd < 0 else u[0] > 512 + 60)]
        u = s[-1] if sd < 0 else s[0]
        if sd < 0: k[y, int(lim) + 1:u[1] + 1, 3] = 0
        else: k[y, u[0]:int(lim), 3] = 0
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, "vnitřní hrana rukávů srovnána")
