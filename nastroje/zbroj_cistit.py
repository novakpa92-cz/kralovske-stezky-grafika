"""Vyčistí obrázek JEDNÉ zbroje od zbytků podkladu (kousky košilky/šedé látky na okrajích, plovoucí úlomky).
python3 nastroje/zbroj_cistit.py <vrstva> [T]  — T = citlivost barvy (výchozí 38)"""
import sys
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]; T = float(sys.argv[2]) if len(sys.argv) > 2 else 38
sx = "zena" if "-zena-" in n else "muz"
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA")); rgb = k[..., :3].astype(float); a = k[..., 3]
t = np.array(Image.open(D / f"postava-telo-{sx}.webp").convert("RGBA")).astype(float)
ys = np.arange(1536)[:, None]
mx = t[..., :3].max(2); mn = t[..., :3].min(2); sat = (mx - mn) / np.maximum(mx, 1)
chem = np.median(t[(t[..., 3] > 200) & (sat < .3) & (ys > 450) & (ys < 1100)][:, :3], axis=0)
lum = rgb.mean(2); s = (rgb.max(2) - rgb.min(2)) / np.maximum(rgb.max(2), 1)
cizi = (a > 0) & ((np.linalg.norm(rgb - chem, axis=2) < T) | ((s < .10) & (lum > 120)))     # košilka nebo světle šedá látka
cizi = cizi.astype(np.uint8)
okraj = cv2.dilate((a < 30).astype(np.uint8), np.ones((5, 5), np.uint8))
c, lab, st, _ = cv2.connectedComponentsWithStats(cizi)
pryc = np.zeros(a.shape, bool)
for i in range(1, c):
    m = lab == i
    if (okraj[m] > 0).any(): pryc |= m                     # cizí kus na okraji zbroje
pryc = cv2.dilate(pryc.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
k[..., 3] = np.where(pryc, 0, a)
# plovoucí úlomky (malé samostatné kusy)
al = (k[..., 3] > 60).astype(np.uint8)
c, lab, st, _ = cv2.connectedComponentsWithStats(al)
big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
for i in range(1, c):
    if i != big and st[i, cv2.CC_STAT_AREA] < 4000: k[..., 3][lab == i] = 0
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, "chem", chem.round(), "pryč", int(pryc.sum()))
