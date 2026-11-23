"""JEDNA vrstva: hladký spodní lem — od řádku Y dolů se spodek každého sloupce srovná do plynulé křivky
(po souvislých úsecích, např. dvě poly zbroje u rozparku), poloprůhledné cáry pryč.
python3 nastroje/lem_hladky.py <vrstva> <Y> [prah_alfy=200] [sigma=8]"""
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
D = Path("img/postava"); n = sys.argv[1]; Y = int(sys.argv[2])
T = int(sys.argv[3]) if len(sys.argv) > 3 else 200; S = float(sys.argv[4]) if len(sys.argv) > 4 else 8
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA")); H, W = k.shape[:2]
ys = np.arange(H)[:, None]
k[..., 3] = np.where((ys >= Y) & (k[..., 3] < T), 0, k[..., 3])
A = k[..., 3] >= T
cb = np.where(A.any(0), H - 1 - np.argmax(A[::-1], axis=0), -1).astype(float)
idx = np.where(cb >= Y)[0]
for sg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
    if len(sg) < 5: k[:, sg, 3] = np.where(ys >= Y, 0, k[:, sg, 3]); continue
    v = gaussian_filter1d(minimum_filter1d(cb[sg], 15), S, mode="nearest") - 1
    for x, yv in zip(sg, v):
        y0 = int(np.floor(yv)); fr = yv - y0
        k[y0 + 2:, x, 3] = 0; k[y0 + 1, x, 3] = int(k[y0 + 1, x, 3] * fr)
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, "lem srovnán", len(idx))
