"""Uši pod helmou (Pavel 27. 11.: „uši občas koukaj skrz železo ven“).
Helma, která má po stranách plech nebo látku v úrovni uší, dostane helma-<x>-skryva-telo.webp = uši těla (měkký okraj)
a zapíše se do skryva.json („telo“), takže hra ucho pod helmou nekreslí."""
import os, sys, glob, json, numpy as np
from PIL import Image
from scipy import ndimage
D = os.path.join(os.path.dirname(__file__), "..", "img", "postava") + "/"
UCESY = ("bez", "copanek", "dlouhe", "kratke", "kudrnate", "ofina", "stredni", "tonzura")
def usi(sx):
    a = np.array(Image.open(D + f"postava-telo-{sx}.webp").convert("RGBA"))[..., 3] > 60
    tvar = [np.where(a[y, 350:674])[0] + 350 for y in (190, 194)]
    L = max(t.min() for t in tvar) + 2; P = min(t.max() for t in tvar) - 2
    m = np.zeros_like(a); yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    m = a & (yy >= 194) & (yy <= 262) & ((xx < L) | (xx > P)) & (xx > 380) & (xx < 644)
    return m, L, P
def zakryva_usi(helma, L, P):
    h = np.array(helma)[..., 3] > 128
    lev = h[205:245, L - 14:L + 6].mean(); pra = h[205:245, P - 6:P + 14].mean()
    return lev > 0.35 and pra > 0.35
if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else D[:-1]
    U = {sx: usi(sx) for sx in ("zena", "muz")}
    sk = json.load(open(D + "skryva.json")); t = set(sk["telo"]); hotovo = []
    for f in sorted(glob.glob(src + "/helma-*.webp")):
        n = os.path.basename(f)[:-5]
        if n.endswith(("-m", "-skryva", "-stin", "-skryva-telo")) or n.endswith(tuple("-" + u for u in UCESY)): continue
        sx = "zena" if n.endswith("-zena") else "muz"; m, L, P = U[sx]
        if not zakryva_usi(Image.open(f).convert("RGBA"), L, P): continue
        mm = ndimage.gaussian_filter(ndimage.binary_dilation(m, iterations=2).astype(np.float32), 1.2)
        out = np.zeros(m.shape + (4,), np.uint8); out[..., 3] = (mm * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(out).save(src + "/" + n + "-skryva-telo.webp", lossless=True); t.add(n); hotovo.append(n)
    sk["telo"] = sorted(t); open(src + "/skryva.json", "w").write(json.dumps(sk, ensure_ascii=False)); print(hotovo)
