"""Čepice podle účesu z přegenerovaného obrázku: vrstva = čepice + vlasy pod ní (z obrázku), jen v oblasti masky a vlasů.
Vybere variantu, u které nevykukuje holá lebka těla nad čepicí."""
import sys, glob, numpy as np, cv2
from PIL import Image
from scipy import ndimage
D = "/home/claude/kralovske-stezky-grafika/img/postava/"
IN = "/mnt/user-data/outputs/ks-in/"
def vrstva(f, sx, vl):
    S = np.array(Image.open(f).convert("RGB").resize((1024, 1536), Image.LANCZOS))
    M = np.array(Image.open(IN + f"ksmask-cepice-{sx}-{vl}.png").convert("L").resize((1024, 1536))) > 127
    H = np.array(Image.open(D + f"vlasy-{sx}-{vl}.webp").convert("RGBA"))[..., 3] > 30 if vl != "bez" else np.zeros_like(M)
    obl = (M | cv2.dilate(H.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    obl[560:] = False
    pozadi = S.min(-1) > 236
    lab, n = ndimage.label(pozadi); okraj = set(np.unique(np.r_[lab[0], lab[:, 0], lab[:, -1]])) - {0}; pozadi = np.isin(lab, list(okraj))
    a = obl & ~ndimage.binary_dilation(pozadi, iterations=1)
    # obličej (mimo masku i vlasy) zůstane z těla
    a = cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    lab, n = ndimage.label(a); vel = ndimage.sum(a, lab, range(1, n + 1)); a = np.isin(lab, [i + 1 for i, v in enumerate(vel) if v > 400])
    L = np.zeros((1536, 1024, 4), np.uint8); L[..., :3] = S; L[..., 3] = (ndimage.gaussian_filter(a.astype(np.float32), .7) * 255).clip(0, 255).astype(np.uint8)
    T = np.array(Image.open(D + f"postava-telo-{sx}.webp").convert("RGBA"))[..., 3] > 30
    lebka = T.copy(); lebka[200:] = False                      # temeno těla (nad čelem)
    holo = int((lebka & ~a).sum())
    return L, holo
def vyber(sx, slug, vl, soubory):
    best = None
    for f in soubory:
        L, holo = vrstva(f, sx, vl)
        print(" ", f.split("/")[-1], "holá lebka px:", holo)
        if best is None or holo < best[1]: best = (L, holo, f)
    L, holo, f = best
    nazev = f"helma-{slug}-{sx}-{vl}" if sx == "zena" else f"helma-{slug}-{vl}"
    im = Image.fromarray(L, "RGBA"); im.save(D + nazev + ".webp", "WEBP", quality=92); im.resize((512, 768), Image.LANCZOS).save(D + nazev + "-m.webp", "WEBP", quality=88)
    return nazev, f.split("/")[-1], holo
if __name__ == "__main__":
    sx, slug, vl = sys.argv[1:4]
    print(vyber(sx, slug, vl, sys.argv[4:]))
