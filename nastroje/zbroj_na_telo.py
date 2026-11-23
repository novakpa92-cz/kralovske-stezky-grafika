"""JEDNA zbroj (vesta): trup zbroje se v každém řádku roztáhne do šířky těla (po vnitřní hranu paže),
aby boky a podpaží kryla zbroj a nic zespodu nevyčuhovalo. Látka se natáhne (vzor zůstane).
python3 nastroje/zbroj_na_telo.py <vrstva> <y_od> [y_do]"""
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, uniform_filter1d
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]; Y0 = int(sys.argv[2])
sx = "zena" if "-zena-" in n else "muz"
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA")); A = k[..., 3] > 60
body = np.array(Image.open(D / f"postava-telo-{sx}.webp").convert("RGBA"))[..., 3] > 5
def useky(row):
    x = np.where(row)[0]
    if not len(x): return []
    c = np.where(np.diff(x) > 1)[0]; return list(zip(np.r_[x[0], x[c + 1]], np.r_[x[c], x[-1]]))
Y1 = int(sys.argv[3]) if len(sys.argv) > 3 else int(np.where(A[:, 512 - 60])[0].max())
pp = next(y for y in range(Y0, 1000) if len([u for u in useky(body[y]) if not (u[0] <= 512 <= u[1])]) >= 2)
li, ri = [], []
for y in range(pp, pp + 120):
    us = useky(body[y]); l = [u for u in us if u[1] < 452]; r = [u for u in us if u[0] > 572]
    if l: li.append((y, max(u[1] for u in l)))
    if r: ri.append((y, min(u[0] for u in r)))
fl = np.polyfit(*zip(*li), 1); fr = np.polyfit(*zip(*ri), 1)
ys = np.arange(Y0, Y1 + 1)
cl, cr, tl, tr = [], [], [], []
for y in ys:
    sg = [u for u in useky(A[y]) if abs((u[0] + u[1]) / 2 - 512) < 200]
    l, r = (min(u[0] for u in sg), max(u[1] for u in sg)) if sg else (512, 512)
    cl.append(l); cr.append(r)
    bs = [u for u in useky(body[y]) if u[0] <= 512 <= u[1]]
    if y >= pp and bs: L, R = bs[0]
    else: L, R = int(np.polyval(fl, y)) + 3, int(np.polyval(fr, y)) - 3
    tl.append(min(l, L)); tr.append(max(r, R))
tl = uniform_filter1d(median_filter(np.array(tl, float), 21), 11); tr = uniform_filter1d(median_filter(np.array(tr, float), 21), 11)
src = k.copy()
for i, y in enumerate(ys):
    l, r, L, R = cl[i], cr[i], int(round(min(tl[i], cl[i]))), int(round(max(tr[i], cr[i])))
    if r - l < 20: continue
    xs = np.arange(L, R + 1)
    # střed (±40 px) beze změny, natahují se jen boční části
    m = 512
    lo = np.interp(xs, [L, m - 40], [l, m - 40]); hi = np.interp(xs, [m + 40, R], [m + 40, r])
    sxs = np.where(xs < m - 40, lo, np.where(xs > m + 40, hi, xs))
    k[y, L:R + 1] = src[y, np.clip(np.round(sxs).astype(int), 0, 1023)]
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, "pp", pp, "řádky", Y0, Y1)
