"""Zarovná konec rukávu u JEDNÉ vrstvy: v každém sloupci manžety odřízne spodních D px (tmavý okraj),
s měkkým přechodem. Jen sloupce, kde je spodek rukávu (ne vnitřní hrana). python3 nastroje/manzety_orez.py <vrstva> [D]"""
import sys
import numpy as np
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]; C = int(sys.argv[2]) if len(sys.argv) > 2 else 12
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA"))
a = k[..., 3] > 100
ys = np.arange(1536)[:, None]; xs = np.arange(1024)[None, :]
for side in (-1, 1):
    arm = a & ((xs - 512) * side > 225) & (ys > 500) & (ys < 840)
    if not arm.any(): continue
    bot = np.where(arm.any(1))[0].max()
    for x in np.where(arm.any(0))[0]:
        b = np.where(arm[:, x])[0].max()
        if b < bot - 30: continue
        k[b - C + 1:b + 8, x, 3] = 0
        for i, f in enumerate((.35, .7)):
            k[b - C - i, x, 3] = int(k[b - C - i, x, 3] * f)
    # zbytky poloprůhledných pixelů pod manžetou
    pod = ((xs - 512) * side > 225) & (ys > bot - C) & (ys < bot + 12) & (k[..., 3] < 140)
    k[..., 3] = np.where(pod, 0, k[..., 3])
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print("ok", n)
