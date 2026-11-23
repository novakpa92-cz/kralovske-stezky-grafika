"""Spodní okraj manžet (konců rukávů) zbroje: díry a tmavé zuby se dokreslí barvou okolí, útržky pryč.
python3 nastroje/manzeta_dokreslit.py <vrstva> [rady=16] [tmave=90]"""
import sys
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 16; TM = float(sys.argv[3]) if len(sys.argv) > 3 else 90
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA")); H, W = k.shape[:2]; ys, xs = np.mgrid[:H, :W]
A8 = (k[..., 3] > 20).astype(np.uint8); c, lab, st, _ = cv2.connectedComponentsWithStats(A8)
k[np.isin(lab, [i for i in range(1, c) if st[i, 4] < 600]), 3] = 0
A = k[..., 3] > 100; l = k[..., :3].astype(np.float32).mean(2)
fix = np.zeros((H, W), bool)
for sd in (xs < 512 - 190, xs > 512 + 190):
    zz = A & sd & (ys > 600) & (ys < 800); ye = np.where(zz.any(1))[0].max()
    pas = sd & (ys > ye - N) & (ys <= ye + 2)
    hull = cv2.morphologyEx(zz.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    fix |= pas & hull & (~A | (l < TM))
bgr = cv2.cvtColor(k[..., :3], cv2.COLOR_RGB2BGR)
f = cv2.cvtColor(cv2.inpaint(bgr, fix.astype(np.uint8), 5, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
k[fix, :3] = f[fix]; k[fix, 3] = 255
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, "manžety dokresleny", int(fix.sum()))
