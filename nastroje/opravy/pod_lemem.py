"""Masky „schová pod sebou“ nesmí nic schovávat pod lemem zbroje (sukně, kalhoty, nohy zůstávají celé)."""
import sys, glob, os, numpy as np
from PIL import Image
D = "/home/claude/kralovske-stezky-grafika/img/postava/"
def oprav(zbroj):
    A = np.array(Image.open(D + zbroj + ".webp").convert("RGBA"))[..., 3] > 60
    H, W = A.shape
    stred = np.where(A[:, 480:544].any(1))[0].max()
    lim = np.full(W, stred)
    for x in range(W):
        ys = np.where(A[:, x])[0]
        if len(ys) and ys.max() > 600: lim[x] = ys.max()
    Y = np.arange(H)[:, None]
    n = 0
    for f in glob.glob(D + zbroj + "*skryva*.webp"):
        if f.endswith("-m.webp"): continue
        im = np.array(Image.open(f).convert("RGBA"))
        pod = Y > (lim[None, :] + 2)
        n += int((im[..., 3][pod] > 0).sum())
        im[..., 3][pod] = 0
        Image.fromarray(im).save(f, "WEBP", lossless=True)
    return n
if __name__ == "__main__":
    for z in sys.argv[1:]: print(z, oprav(z))
