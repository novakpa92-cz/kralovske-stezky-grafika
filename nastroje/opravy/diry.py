"""Zalepí díry ve vrstvě zbroje pixely z původního obrázku z ComfyUI (stejný rám). Volitelně box (x0,y0,x1,y1) — jen tam."""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage
D = "/home/claude/kralovske-stezky-grafika/img/postava/"
def zalep(vrstva, zdroj, box=None, max_dira=60000, uzavri=25, uloz=True):
    L = np.array(Image.open(D + vrstva + ".webp").convert("RGBA"))
    S = np.array(Image.open(zdroj).convert("RGB").resize((L.shape[1], L.shape[0]), Image.LANCZOS))
    a = L[..., 3] > 40
    uz = cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (uzavri, uzavri))) > 0
    plne = ndimage.binary_fill_holes(uz)
    dira = plne & ~a & (S.min(-1) < 235)
    lab, n = ndimage.label(dira)
    vel = ndimage.sum(dira, lab, range(1, n + 1))
    dira = np.isin(lab, [i + 1 for i, v in enumerate(vel) if v < max_dira])
    if box:
        m = np.zeros_like(dira); m[box[1]:box[3], box[0]:box[2]] = True; dira &= m
    dira = ndimage.binary_dilation(dira, iterations=1) & ~a | dira
    L[dira, :3] = S[dira]; L[dira, 3] = 255
    if uloz:
        im = Image.fromarray(L, "RGBA"); im.save(D + vrstva + ".webp", "WEBP", quality=92)
        im.resize((512, 768), Image.LANCZOS).save(D + vrstva + "-m.webp", "WEBP", quality=88)
    return int(dira.sum())
if __name__ == "__main__":
    print(zalep(sys.argv[1], sys.argv[2]))
