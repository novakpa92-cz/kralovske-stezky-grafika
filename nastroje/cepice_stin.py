"""Kontaktní stín pod pokrývkou hlavy (Pavel 27. 11.: vrstvy musí plynule navazovat).
Z vrstvy helma-<x>.webp udělá helma-<x>-stin.webp: měkké ztmavení těsně pod okrajem a po stranách (na čele a vlasech),
nikdy přes samotnou čepici. Hra ho kreslí „source-atop“, takže ztmaví jen to, co pod ním je (tvář, vlasy), ne pozadí.
Zároveň zjemní okraj čepice (bez zubů)."""
import sys, os, numpy as np
from PIL import Image
from scipy import ndimage
def stin(cap_rgba, sila=0.6, posun=7, rozmaz=5.0):
    a = cap_rgba[..., 3].astype(np.float32) / 255
    sh = ndimage.shift(a, (posun, 0), order=1)                       # světlo shora → stín dolů
    sh = ndimage.gaussian_filter(np.maximum(sh, ndimage.gaussian_filter(a, 2)), rozmaz)
    sh = sh * (1 - ndimage.gaussian_filter(a, 0.8)) * sila         # ne přes čepici
    sh[:80] = 0
    out = np.zeros(cap_rgba.shape, np.uint8); out[..., 0], out[..., 1], out[..., 2] = 38, 26, 18
    out[..., 3] = (sh * 255).clip(0, 255).astype(np.uint8); return out
def zjemni(cap_rgba, drobky=None):
    o = cap_rgba.copy(); a = o[..., 3].astype(np.float32)
    m = a > 20; lab, n = ndimage.label(m)                                       # odtržené drobky pryč
    if n > 1:
        vel = ndimage.sum(m, lab, range(1, n + 1)); hl = vel.max()
        a[np.isin(lab, [i + 1 for i, v in enumerate(vel) if v < (drobky if drobky is not None else max(400, hl * 0.004))])] = 0
    hr = (ndimage.maximum_filter(a, 3) - ndimage.minimum_filter(a, 3)) > 40   # jen okrajový pás
    sm = ndimage.gaussian_filter(a, 0.7); o[..., 3] = np.where(hr, np.minimum(a, sm), a).astype(np.uint8); return o
if __name__ == "__main__":
    for f in sys.argv[2:]:
        im = np.array(Image.open(f).convert("RGBA")); im = zjemni(im)
        Image.fromarray(im).save(f, lossless=True)
        Image.fromarray(stin(im)).save(os.path.join(sys.argv[1], os.path.basename(f)[:-5] + "-stin.webp"), lossless=True)
