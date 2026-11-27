"""Masky „schová pod sebou“ souměrně: co se schová na jedné paži, schová se i na druhé (zrcadlo kolem středu postavy)."""
import glob, sys, numpy as np
from PIL import Image
D = "/home/claude/kralovske-stezky-grafika/img/postava/"
def soumerne(zbroj, y0, y1, stred=507):
    n = 0
    for f in glob.glob(D + zbroj + "*skryva*.webp"):
        if f.endswith("-m.webp"): continue
        im = np.array(Image.open(f).convert("RGBA")); a = im[..., 3].astype(int)
        W = a.shape[1]; xs = np.arange(W); src = np.clip(2 * stred - xs, 0, W - 1)
        zr = a[:, src]
        nova = a.copy(); nova[y0:y1] = np.maximum(a[y0:y1], zr[y0:y1])
        n += int((nova != a).sum()); im[..., 3] = nova.astype(np.uint8)
        Image.fromarray(im).save(f, "WEBP", lossless=True)
    return n
if __name__ == "__main__":
    print(soumerne(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
