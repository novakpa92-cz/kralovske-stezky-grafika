"""Konec rukávu (manžeta) zbroje se plynulým zkosením předloktí posune na střed zápěstí těla pod ním; tmavé zuby na okraji pryč.
python3 nastroje/manzeta_na_ruku.py <vrstva> [rady=90]"""
import sys
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 90
sx = "zena" if "-zena-" in n else "muz"
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA")); A = k[..., 3] > 100
T = np.array(Image.open(D / f"postava-telo-{sx}.webp").convert("RGBA"))[..., 3] > 100
H, W = A.shape; ys, xs = np.mgrid[:H, :W].astype(np.float32)
def useky(r):
    x = np.where(r)[0]
    if not len(x): return []
    c = np.where(np.diff(x) > 1)[0]; return list(zip(np.r_[x[0], x[c + 1]], np.r_[x[c], x[-1]]))
mx = xs.copy()
for lev in (True, False):
    sd = (xs[0] < 512 - 190) if lev else (xs[0] > 512 + 190)
    zz = A & sd[None, :]; zz[:600] = False
    ye = np.where(zz.any(1))[0].max()
    cut = lambda y, M: max([u for u in useky(M[y] & sd) ], key=lambda u: u[1] - u[0])
    a0, b0 = cut(ye - 12, zz); ba, bb = cut(ye + 4, T & ~A)
    sh = (ba + bb) / 2 - (a0 + b0) / 2
    w = np.clip((ys - (ye - N)) / N, 0, 1)
    reg = sd[None, :] & (ys > ye - N - 5) & (ys < ye + 10)
    mx = np.where(reg, mx - sh * w, mx)
    print(n, "L" if lev else "P", "konec", ye, "posun", round(sh, 1))
out = cv2.remap(k, mx.astype(np.float32), ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
A2 = out[..., 3] > 100; l = out[..., :3].astype(np.float32).mean(2); med = np.median(l[A2])
for lev in (True, False):
    sd = (xs[0] < 512 - 190) if lev else (xs[0] > 512 + 190)
    zz = A2 & sd[None, :]; zz[:600] = False; ye = np.where(zz.any(1))[0].max()
    d = zz & (ys > ye - 14) & (l < med * 0.5)
    out[cv2.dilate(d.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, 3] = 0
Image.fromarray(out).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(out).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
