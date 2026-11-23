"""JEDNA zbroj: obrys trupu v pase vyhladí (bez zářezů a výstupků), díry dokreslí a světlý lem na okraji přebarví.
python3 nastroje/trup_obrys.py <vrstva> [od] [do]   (řádky plátna, výchozí 540–800)"""
import sys
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import median_filter, uniform_filter1d
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]
Y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 540; Y1 = int(sys.argv[3]) if len(sys.argv) > 3 else 800
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA")); A = k[..., 3] > 100
L = np.full(1536, -1); R = np.full(1536, -1)
for y in range(Y0, Y1):
    x = np.where(A[y])[0]
    if not len(x): continue
    c = np.where(np.diff(x) > 1)[0]; st = np.r_[x[0], x[c + 1]]; en = np.r_[x[c], x[-1]]
    for a, b in zip(st, en):
        if a <= 512 <= b: L[y], R[y] = a, b
ys = np.arange(Y0, Y1); ok = L[ys] > 0
for E in (L, R):
    v = E[ys].astype(float); v[~ok] = np.interp(ys[~ok], ys[ok], v[ok])
    v = uniform_filter1d(median_filter(v, 25), 15)
    E[ys] = np.round(v).astype(int)
# souměrnost: obě strany trupu na větší z obou vzdáleností od středu (vykousnutá strana se doplní)
d = np.maximum(512 - L[ys], R[ys] - 512).astype(float)
d = uniform_filter1d(median_filter(d, 31), 21)
L[ys] = np.round(512 - d).astype(int); R[ys] = np.round(512 + d).astype(int)
H, W = A.shape
dira = np.zeros((H, W), np.uint8)
for y in ys:
    l, r = L[y], R[y]
    row = A[y]
    # vně hladkého obrysu trupu (jen kousek do stran, ať se nesahá na rukávy)
    for xx in list(range(max(0, l - 25), l)) + list(range(r + 1, min(W, r + 26))):
        if row[xx]:
            # patří k rukávu? (navazuje dál od trupu) — pak nechat
            far = row[max(0, xx - 30):xx].all() if xx < l else row[xx + 1:xx + 31].all()
            if not far: k[y, xx, 3] = 0
    dira[y, l:r + 1] = ~row[l:r + 1]
    # světlý okraj uvnitř obrysu
lum = k[..., :3].astype(float).mean(2)
pas = np.zeros((H, W), bool)
for y in ys:
    pas[y, L[y]:L[y] + 12] = True; pas[y, R[y] - 11:R[y] + 1] = True
ref = np.median(lum[(k[..., 3] > 200) & ~pas & (np.arange(H)[:, None] >= Y0) & (np.arange(H)[:, None] < Y1) & (np.abs(np.arange(W) - 512) < 120)[None, :]])
svetle = pas & (lum > ref + 22)
dira = (dira > 0) | svetle
zrc = k[:, ::-1]                                   # zrcadlo trupu: díry na jedné straně dokreslit látkou z druhé
mz = dira & (zrc[..., 3] > 200) & ~svetle
k[mz, :3] = zrc[mz, :3]; k[mz, 3] = 255; dira = dira & ~mz
bgr = cv2.cvtColor(k[..., :3], cv2.COLOR_RGB2BGR)
fill = cv2.cvtColor(cv2.inpaint(bgr, dira.astype(np.uint8), 9, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
k[dira, :3] = fill[dira]; k[dira, 3] = 255
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, "dokresleno", int(dira.sum()), "ref", round(ref))
