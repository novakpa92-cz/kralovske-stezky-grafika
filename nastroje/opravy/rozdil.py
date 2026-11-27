import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage
D = "/home/claude/kralovske-stezky-grafika/img/postava/"
def oprav(vrstva, zdroj, zaklad, box, prah=22):
    L = np.array(Image.open(D + vrstva + ".webp").convert("RGBA"))
    S = np.array(Image.open(zdroj).convert("RGB").resize((1024, 1536), Image.LANCZOS)).astype(np.float32)
    B = np.array(Image.open(zaklad).convert("RGB").resize((1024, 1536), Image.LANCZOS)).astype(np.float32)
    sb = np.stack([ndimage.gaussian_filter(S[..., i], 2.5) for i in range(3)], -1)
    bb = np.stack([ndimage.gaussian_filter(B[..., i], 2.5) for i in range(3)], -1)
    zm = np.abs(sb - bb).max(-1) > prah
    zm &= S.min(-1) < 232
    a = L[..., 3] > 40
    obl = np.zeros_like(a); x0, y0, x1, y1 = box; obl[y0:y1, x0:x1] = True
    pridat = zm & obl & ~a
    pridat = cv2.morphologyEx(pridat.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    pridat = ndimage.binary_fill_holes(pridat | a) & ~a & obl
    L[pridat, :3] = S[pridat].astype(np.uint8); L[pridat, 3] = 255
    im = Image.fromarray(L, "RGBA"); im.save(D + vrstva + ".webp", "WEBP", quality=92)
    im.resize((512, 768), Image.LANCZOS).save(D + vrstva + "-m.webp", "WEBP", quality=88)
    return int(pridat.sum())
