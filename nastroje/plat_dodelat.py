"""Celoplátová zbroj: tmavý rozkrok pod šosy pryč, díry uvnitř obrysu (po řemíncích) dokreslit.
python3 nastroje/plat_dodelat.py <vrstva>"""
import sys
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA")); H, W = k.shape[:2]; ys, xs = np.mgrid[:H, :W]
A = k[..., 3] > 100; l = cv2.GaussianBlur(k[..., :3].astype(np.float32).mean(2), (0, 0), 2); med = np.median(l[A])
if med > 90:
    cr = A & (np.abs(xs - 512) < 30) & (ys > 790) & (ys < 900) & (l < med * 0.5)
    c, lab, st, _ = cv2.connectedComponentsWithStats(cr.astype(np.uint8))
    for i in range(1, c):
        if st[i, 4] > 30: k[cv2.dilate((lab == i).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0, 3] = 0
A = (k[..., 3] > 100).astype(np.uint8)
inv = 1 - A; c, lab, st, _ = cv2.connectedComponentsWithStats(inv)
holes = np.zeros(A.shape, bool)
for i in range(1, c):
    x, y, w, h, a = st[i]
    if a < 1500 and x > 0 and y > 0 and x + w < W and y + h < H and y > 700: holes |= lab == i
bgr = cv2.cvtColor(k[..., :3], cv2.COLOR_RGB2BGR)
f = cv2.cvtColor(cv2.inpaint(bgr, holes.astype(np.uint8), 7, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
k[holes, :3] = f[holes]; k[holes, 3] = 255
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, "díry", int(holes.sum()))
