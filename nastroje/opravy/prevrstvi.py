"""Vrstvu zbroje doplní z původního obrázku: kov a kůže (ne len, kůže těla ani pozadí) uvnitř oblasti zbroje."""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage
D = "/home/claude/kralovske-stezky-grafika/img/postava/"
def oprav(vrstva, zdroj, y0, y1, x0=0, x1=1024, nahled=None):
    L = np.array(Image.open(D + vrstva + ".webp").convert("RGBA"))
    S = np.array(Image.open(zdroj).convert("RGB").resize((1024, 1536), Image.LANCZOS)).astype(int)
    r, g, b = S[..., 0], S[..., 1], S[..., 2]; lum = S.mean(-1)
    lnene = (r - b > 14) & (r - b < 70) & (lum > 135) & (np.abs(r - g) < 30)          # béžový len košile
    kuze = (r - b > 45) & (r > 150) & (r - g > 25)                                    # kůže těla
    pozadi = S.min(-1) > 232
    a = L[..., 3] > 40
    oblast = np.zeros_like(a); oblast[y0:y1, x0:x1] = True
    pridat = oblast & ~a & ~lnene & ~kuze & ~pozadi
    pridat = cv2.morphologyEx(pridat.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    # jen co souvisí se stávající zbrojí
    lab, n = ndimage.label(pridat | a)
    hl = set(np.unique(lab[a])) - {0}
    pridat &= np.isin(lab, list(hl))
    L[pridat, :3] = S[pridat]; L[pridat, 3] = 255
    im = Image.fromarray(L, "RGBA"); im.save(D + vrstva + ".webp", "WEBP", quality=92)
    im.resize((512, 768), Image.LANCZOS).save(D + vrstva + "-m.webp", "WEBP", quality=88)
    return int(pridat.sum())
if __name__ == "__main__":
    print(oprav(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:])))
