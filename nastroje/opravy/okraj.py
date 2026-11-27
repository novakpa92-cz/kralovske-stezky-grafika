"""Oděv nesmí vyčuhovat kolem zbroje: v pásu kolem obrysu zbroje (nad lemem) se oděv schová."""
import glob, sys, numpy as np, cv2
from PIL import Image
D = "/home/claude/kralovske-stezky-grafika/img/postava/"
def okraj(zbroj, sirka=25, y0=0, y1=None, xmax=None, stred=507):
    A = np.array(Image.open(D + zbroj + ".webp").convert("RGBA"))[..., 3] > 60
    H, W = A.shape
    hem = np.where(A[:, 480:540].any(1))[0].max()
    y1 = y1 or hem
    pas = (cv2.dilate(A.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sirka + 1, 2 * sirka + 1))) > 0) & ~A
    Y, X = np.mgrid[:H, :W]
    pas &= (Y >= y0) & (Y < y1)
    if xmax: pas &= np.abs(X - stred) < xmax
    n = 0
    for f in glob.glob(D + zbroj + "*-skryva.webp"):
        im = np.array(Image.open(f).convert("RGBA"))
        n += int((im[..., 3][pas] < 255).sum()); im[..., 3][pas] = 255
        Image.fromarray(im).save(f, "WEBP", lossless=True)
    return n
if __name__ == "__main__":
    a = sys.argv; print(okraj(a[1], int(a[2]), int(a[3]), int(a[4]) if a[4] != "0" else None, int(a[5]) if len(a) > 5 else None))
