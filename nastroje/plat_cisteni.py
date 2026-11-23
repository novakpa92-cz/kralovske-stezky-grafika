"""Celoplátová zbroj (JEDNA vrstva): poloprůhledné halo pryč, zadní (vnitřní) strana nákrčníku vystřižená,
tmavé řemínky/přezky vyčnívající z obrysu pryč, růžové prsty v rukavicích pryč, obrys stehen/holení srovnaný.
python3 nastroje/plat_cisteni.py <vrstva>"""
import sys
import numpy as np, cv2
from PIL import Image
from pathlib import Path
from scipy.ndimage import minimum_filter1d, maximum_filter1d, gaussian_filter1d
D = Path("img/postava"); n = sys.argv[1]
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA"))
H, W = k.shape[:2]; ys, xs = np.mgrid[:H, :W]
k[..., 3] = np.where(k[..., 3] < 215, 0, k[..., 3])                                   # halo
A = k[..., 3] > 100; rgb = k[..., :3].astype(np.float32); l = cv2.GaussianBlur(rgb.mean(2), (0, 0), 2)
s = (rgb.max(2) - rgb.min(2)) / np.maximum(rgb.max(2), 1); rg = (rgb[..., 0] - rgb[..., 1]) / np.maximum(rgb[..., 0], 1)
med = np.median(l[A])
top = np.where(A.any(1))[0].min()
d = A & (ys < top + 30) & (np.abs(xs - 512) < 60) & (l < min(70, med * 0.6))             # vnitřek nákrčníku
for x in np.where(d.any(0))[0]:
    k[:np.where(d[:, x])[0].max() + 1, x, 3] = 0
pink = A & (np.abs(xs - 512) > 200) & (ys > 780) & (ys < 1000) & (rg > 0.12) & (s > 0.18) & (l > 60)
k[cv2.dilate(pink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, 3] = 0             # kůže prstů v kresbě
A = k[..., 3] > 100
if med > 90:                                                                        # světlý plát: tmavé řemínky u obrysu pryč
    a8 = A.astype(np.uint8); edge = (a8 > 0) & (cv2.erode(a8, np.ones((31, 31), np.uint8)) == 0)
    dark = edge & (l < med * 0.55) & (ys > top + 60)
    c, lab, st, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    for i in range(1, c):
        if 20 < st[i, 4] < 900: k[cv2.dilate((lab == i).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0, 3] = 0
    A = k[..., 3] > 100
add = np.zeros(A.shape, bool)                                                       # vnější obrys nohou srovnat (zuby, zářezy)
def useky(r):
    x = np.where(r)[0]
    if not len(x): return []
    c = np.where(np.diff(x) > 1)[0]; return list(zip(np.r_[x[0], x[c + 1]], np.r_[x[c], x[-1]]))
y0 = 800; rows = np.arange(y0, H)
for sd in (-1, 1):
    e = np.full(len(rows), np.nan)
    for i, y in enumerate(rows):
        sg = [u for u in useky(A[y]) if (u[0] < 512 if sd < 0 else u[1] > 512)]
        if sg: e[i] = min(u[0] for u in sg) if sd < 0 else max(u[1] for u in sg)
    ok = ~np.isnan(e)
    if ok.sum() < 50: continue
    ee = np.interp(np.arange(len(e)), np.where(ok)[0], e[ok])
    env = gaussian_filter1d(minimum_filter1d(ee, 31) if sd < 0 else maximum_filter1d(ee, 31), 5)
    for i, y in enumerate(rows):
        if not ok[i]: continue
        if sd < 0 and e[i] > env[i] + 1 and e[i] - env[i] < 25: add[y, int(env[i]):int(e[i])] = True
        if sd > 0 and e[i] < env[i] - 1 and env[i] - e[i] < 25: add[y, int(e[i]) + 1:int(env[i]) + 1] = True
bgr = cv2.cvtColor(k[..., :3], cv2.COLOR_RGB2BGR)
f = cv2.cvtColor(cv2.inpaint(bgr, add.astype(np.uint8), 9, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
k[add, :3] = f[add]; k[add, 3] = 255
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, "vyčištěno", int(add.sum()))
