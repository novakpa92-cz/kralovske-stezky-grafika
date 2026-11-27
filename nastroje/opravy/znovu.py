"""Vrstva zbroje znovu z rozdílu proti výchozímu obrázku (co se změnilo = zbroj), největší souvislý kus, zaplněné díry."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
D = "/home/claude/kralovske-stezky-grafika/img/postava/"
def znovu(vrstva, zdroj, zaklad, y0, y1, prah=22, uloz=True, odstran=None):
    S = np.array(Image.open(zdroj).convert("RGB").resize((1024, 1536), Image.LANCZOS)).astype(np.float32)
    B = np.array(Image.open(zaklad).convert("RGB").resize((1024, 1536), Image.LANCZOS)).astype(np.float32)
    sb = np.stack([ndimage.gaussian_filter(S[..., i], 2.5) for i in range(3)], -1)
    bb = np.stack([ndimage.gaussian_filter(B[..., i], 2.5) for i in range(3)], -1)
    m = (np.abs(sb - bb).max(-1) > prah) & (S.min(-1) < 232)
    Y = np.arange(1536)[:, None]; m &= (Y >= y0) & (Y <= y1)
    if odstran is not None: m &= ~odstran
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    lab, n = ndimage.label(m); vel = ndimage.sum(m, lab, range(1, n + 1)); m = lab == (int(np.argmax(vel)) + 1)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    m = ndimage.binary_fill_holes(m)
    a = ndimage.gaussian_filter(m.astype(np.float32), 0.8)
    L = np.zeros((1536, 1024, 4), np.uint8); L[..., :3] = S.astype(np.uint8); L[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    if uloz:
        im = Image.fromarray(L, "RGBA"); im.save(D + vrstva + ".webp", "WEBP", quality=92)
        im.resize((512, 768), Image.LANCZOS).save(D + vrstva + "-m.webp", "WEBP", quality=88)
    return L
