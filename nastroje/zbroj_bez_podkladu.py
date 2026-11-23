"""JEDNA zbroj: pryč světlé zbytky podkladu (košilka) — pixely světlejší než materiál zbroje, kromě kovu/přezek.
python3 nastroje/zbroj_bez_podkladu.py <vrstva> <práh_jasu> [y_od y_do]"""
import sys
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]; T = float(sys.argv[2])
y0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0; y1 = int(sys.argv[4]) if len(sys.argv) > 4 else 1536
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA")); rgb = k[..., :3].astype(float)
lum = rgb.mean(2); s = (rgb.max(2) - rgb.min(2)) / np.maximum(rgb.max(2), 1)
ys = np.arange(1536)[:, None]
m = ((k[..., 3] > 0) & (lum > T) & (s < .45) & (ys >= y0) & (ys < y1)).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
m = cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0
k[..., 3] = np.where(m, 0, k[..., 3])
al = (k[..., 3] > 60).astype(np.uint8); c, lab, st, _ = cv2.connectedComponentsWithStats(al)
big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); k[..., 3] = np.where((lab != big) & (lab > 0), 0, k[..., 3])
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, "pryč", int(m.sum()))
