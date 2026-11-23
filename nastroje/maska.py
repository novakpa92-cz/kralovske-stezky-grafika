"""Maska pro JEDNU vrstvu (ruční oprava po bodech, ne hromadně).
python3 nastroje/maska.py <vrstva> telo-boky|telo-rukavy|obl-boky ...  — přidá k existující masce."""
import sys, json
import numpy as np
from PIL import Image
from pathlib import Path
D = Path("img/postava")
def alfa(p): return np.array(Image.open(p).convert("RGBA"))[:, :, 3] > 100
def roz(a, rx, ry):
    out = a.copy()
    for d in range(1, rx + 1): out[:, d:] |= a[:, :-d]; out[:, :-d] |= a[:, d:]
    b = out.copy()
    for d in range(1, ry + 1): out[d:, :] |= b[:-d, :]; out[:-d, :] |= b[d:, :]
    return out
def useky(row):
    xs = np.where(row)[0]
    if not len(xs): return []
    cut = np.where(np.diff(xs) > 1)[0]
    return list(zip(np.r_[xs[0], xs[cut + 1]], np.r_[xs[cut], xs[-1]]))
def latka(sx):
    im = np.array(Image.open(D / f"postava-telo-{sx}.webp").convert("RGBA")).astype(float) / 255
    mx = im[..., :3].max(2); mn = im[..., :3].min(2); sat = (mx - mn) / np.maximum(mx, 1e-6)
    ys = np.arange(im.shape[0])[:, None]
    return (im[..., 3] > .4) & (sat < .36) & (mx > .45) & (ys >= 330)
def uloz(m, p):
    a = (m * 255).astype(np.uint8)
    Image.fromarray(np.dstack([np.full_like(a, 255)] * 3 + [a]), "RGBA").save(p, "WEBP", lossless=True)

n, ops = sys.argv[1], sys.argv[2:]
sx = "zena" if "-zena" in n else "muz"
U = roz(alfa(D / f"{n}.webp"), 3, 3)
body = alfa(D / f"postava-telo-{sx}.webp")
H, W = U.shape
Z = np.zeros_like(U)
for y in range(430, 790):
    sb = useky(body[y]); su = [s for s in useky(U[y]) if s[0] <= 512 <= s[1]]
    ruce = [s for s in sb if not (s[0] <= 512 <= s[1]) and abs((s[0] + s[1]) / 2 - 512) > 150]
    if "boky" in " ".join(ops) and su:
        L, R = su[0]; z = np.zeros(W, bool); z[max(0, L - 70):L] = True; z[R + 1:R + 71] = True
        for a, b in ruce: z[max(0, a - 6):b + 7] = False
        Z[y] |= z
    if "rukavy" in " ".join(ops):
        for a, b in ruce:
            lo, hi = max(0, a - 45), min(W, b + 46); seg = U[y, lo:hi]
            if seg.sum() < 6: continue
            xs = np.where(seg)[0] + lo; z = np.zeros(W, bool); z[lo:hi] = True; z[xs.min():xs.max() + 1] = False
            Z[y] |= z
if "telo-mezery" in ops:            # mezi trupem a rukávem vrchní vrstvy: tělo, co není kůže, pryč
    im = np.array(Image.open(D / f"postava-telo-{sx}.webp").convert("RGBA")).astype(float) / 255
    mx = im[..., :3].max(2); mn = im[..., :3].min(2); sat = (mx - mn) / np.maximum(mx, 1e-6)
    neni_kuze = sat < .40
    for y in range(400, 800):
        xs = np.where(U[y])[0]
        if len(xs) < 2: continue
        Z[y, xs.min():xs.max() + 1] |= body[y, xs.min():xs.max() + 1] & neni_kuze[y, xs.min():xs.max() + 1]
Z &= ~U
sez = json.load(open(D / "skryva.json"))
if any(o.startswith("telo") for o in ops):
    p = D / f"{n}-skryva-telo.webp"
    m = (alfa(p) if p.exists() else np.zeros_like(Z)) | (Z & (roz(latka(sx), 2, 2) | ("telo-mezery" in ops)))
    uloz(m, p); sez["telo"] = sorted(set(sez["telo"]) | {n})
if any(o.startswith("obl") for o in ops):
    p = D / f"{n}-skryva.webp"
    m = (alfa(p) if p.exists() else np.zeros_like(Z)) | Z
    uloz(m, p); sez["obleceni"] = sorted(set(sez["obleceni"]) | {n})
json.dump(sez, open(D / "skryva.json", "w"), ensure_ascii=False)
print(n, ops, int(Z.sum()))
