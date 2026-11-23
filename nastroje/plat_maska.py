"""Celoplátová zbroj: pod ní se schová všechno (oděv i tělo) od nákrčníku dolů; zůstane hlava, krk a ruce, které zbroj nekryje.
python3 nastroje/plat_maska.py <vrstva>"""
import sys, json
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA")); A = k[..., 3] > 60
t = np.array(Image.open(D / "postava-telo-zena.webp").convert("RGBA")).astype(np.float32)
ys, xs = np.mgrid[:1536, :1024]
ct = np.where(A.any(0), np.argmax(A, axis=0), 1536); top = np.where(A.any(1))[0].min()
lim = np.where(np.abs(xs[0] - 512) < 90, ct + 4, top + 6)
Ajad = cv2.erode((k[..., 3] > 250).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
m = (ys >= lim[None, :]) & ~Ajad
rg = (t[..., 0] - t[..., 1]) / np.maximum(t[..., 0], 1)
cb = np.where(A.any(0), 1535 - np.argmax(A[::-1], axis=0), -1)
ruce = (t[..., 3] > 60) & (rg >= .19) & (np.abs(xs - 512) > 190) & (ys > 700) & (ys < 1000)   # holé ruce pod rukávy
for sd in (xs[0] < 512 - 190, xs[0] > 512 + 190):       # holá ruka jen pod čarou spodku manžety (přímka proložená spodkem rukávu)
    zz = A & sd[None, :] & (ys > 600) & (ys < 1000)
    if not zz.any(): continue
    cb2 = np.where(zz.any(0), zz.shape[0] - 1 - np.argmax(zz[::-1], axis=0), -1); ye = cb2.max()
    cx = np.where(sd & (cb2 > ye - 25))[0]
    p = np.polyfit(cx, cb2[cx], 1) if len(cx) > 5 else np.array([0., ye])
    cut = np.polyval(p, xs[0]) - 3
    ruce &= ~(sd[None, :] & (ys <= cut[None, :]))
MB = m & ~ruce
def uloz(mm, p):
    a = (mm * 255).astype(np.uint8); Image.fromarray(np.dstack([np.full_like(a, 255)] * 3 + [a]), "RGBA").save(p, "WEBP", lossless=True)
uloz(m, D / f"{n}-skryva.webp"); uloz(MB, D / f"{n}-skryva-telo.webp")
j = json.load(open(D / "skryva.json")); j["obleceni"] = sorted(set(j["obleceni"]) | {n}); j["telo"] = sorted(set(j["telo"]) | {n})
json.dump(j, open(D / "skryva.json", "w"), ensure_ascii=False)
print(n, "maska plátu")
