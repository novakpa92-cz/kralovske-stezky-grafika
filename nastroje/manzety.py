"""Konec rukávu u JEDNÉ vrstvy: tmavý vnitřek rukávu (zadní strana) se zaplní látkou rukávu,
takže rukáv působí uzavřeně a ruka z něj vychází. python3 nastroje/manzety.py odev-zena-saty"""
import sys
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA"))
a = k[..., 3] > 100; lum = k[..., :3].astype(float).mean(2)
ys = np.arange(1536)[:, None]; xs = np.arange(1024)[None, :]
hole = np.zeros(a.shape, np.uint8)
for side in (-1, 1):
    arm = a & ((xs - 512) * side > 225) & (ys > 500) & (ys < 840)
    if not arm.any(): continue
    bot = np.where(arm.any(1))[0].max()
    ref = np.median(lum[arm & (ys < bot - 70) & (ys > bot - 200)])
    dark = (arm & (ys >= bot - 50) & (lum < ref * .62)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    c, lab, st, cen = cv2.connectedComponentsWithStats(dark)
    if c > 1:
        i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); hole |= (lab == i).astype(np.uint8)
hole = cv2.dilate(hole, np.ones((5, 5), np.uint8))
hole[~a] = 0
bgr = cv2.cvtColor(k[..., :3], cv2.COLOR_RGB2BGR)
fill = cv2.cvtColor(cv2.inpaint(bgr, hole, 11, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
k[hole > 0, :3] = fill[hole > 0]
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, int(hole.sum()))
