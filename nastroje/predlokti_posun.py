"""Předloktí zbroje (oddělené od trupu) posune vodorovně, aby sedělo na paži těla; posun plynule roste od y0 do y1 a pak drží.
python3 nastroje/predlokti_posun.py <vrstva> L|P y0 y1 posun [y_konec=760]"""
import sys
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n, s = sys.argv[1], sys.argv[2]; y0, y1, P = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA")); A = k[..., 3] > 0
H, W = A.shape; ys, xs = np.mgrid[:H, :W].astype(np.float32); mx = xs.copy()
lev = s == "L"
ye = int(sys.argv[6]) if len(sys.argv) > 6 else 760
for y in range(y0, ye):
    r = A[y, :512] if lev else A[y, 512:]
    x = np.where(r)[0]
    if not len(x): continue
    c = np.where(np.diff(x) > 1)[0]
    if lev:
        if not len(c): continue
        g = (x[c[0]] + x[c[0] + 1]) / 2                       # mezera mezi paží a trupem
        reg = xs[y] < g
    else:
        if not len(c): continue
        g = 512 + (x[c[-1]] + x[c[-1] + 1]) / 2
        reg = xs[y] > g
    w = min(1.0, (y - y0) / max(1, y1 - y0)); w = w * w * (3 - 2 * w)
    reg = reg & (np.abs(xs[y] - (g - P * w)) > 0)
    mx[y, reg] = xs[y, reg] - P * w
    # cílové pixely u mezery, kam by se natáhl trup: nechat prázdné
out = cv2.remap(k, mx, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
Image.fromarray(out).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(out).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, s, "posun", P)
