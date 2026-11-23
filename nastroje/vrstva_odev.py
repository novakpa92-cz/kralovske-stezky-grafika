"""Vrstva 2 (oděv) na těle ve spodním prádle: spodní prádlo (košilka/bryčesy), které nezakrývá oděv, zmizí.
Kůže (ruce, nohy, krk) zůstává. Spouští se pro JEDEN oděv: python3 nastroje/vrstva_odev.py odev-zena-saty"""
import sys, json
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava")
n = sys.argv[1]; sx = "zena" if "-zena-" in n else "muz"
im = np.array(Image.open(D / f"postava-telo-{sx}.webp").convert("RGBA")).astype(float) / 255
mx = im[..., :3].max(2); mn = im[..., :3].min(2); sat = (mx - mn) / np.maximum(mx, 1e-6)
ys = np.arange(1536)[:, None]; xs = np.arange(1024)[None, :]
pradlo = ((im[..., 3] > .4) & (sat < .37) & (ys >= 330) & (ys <= 1240)).astype(np.uint8)
pradlo = cv2.morphologyEx(pradlo, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
k, lab, st, _ = cv2.connectedComponentsWithStats(pradlo)
pradlo = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)       # jen souvislá košilka, ne skvrny na rukou
pradlo = cv2.dilate(pradlo, np.ones((5, 5), np.uint8)) > 0                       # i s měkkým okrajem
pradlo &= ~((sat > .43) & (im[..., 3] > .4))                                     # ale ne do jasné kůže
U = np.array(Image.open(D / f"{n}.webp").convert("RGBA"))[..., 3] > 200
skryt = pradlo & ~U
# boty: chodidlo, které vyčuhuje z boty oděvu, zmizí (od horního okraje boty dolů)
telo = im[..., 3] > .02
for side in (-1, 1):
    cand = (U & ((xs - 512) * side > 20) & (ys > 1330)).astype(np.uint8)
    c, lab, st, _ = cv2.connectedComponentsWithStats(cand)
    if c < 2: continue
    i = max(range(1, c), key=lambda j: st[j, cv2.CC_STAT_TOP] + st[j, cv2.CC_STAT_HEIGHT])   # nejnižší kus = bota (ne lem sukně)
    bota = lab == i
    if st[i, cv2.CC_STAT_TOP] < 1340:              # bota splývá se sukní: pod sukní není vidět noha, jen bota
        skryt |= telo & ~U & ((xs - 512) * side > 20) & (ys >= 1330)
        continue
    if bota.sum() < 500: continue
    top = np.where(bota.any(1))[0].min()
    for y in range(top + 12, 1536):                 # jen co je mimo botu do stran nebo pod ní (otvor boty zůstane)
        row = np.where(bota[y])[0]
        strana = ((np.arange(1024) - 512) * side > 20) & telo[y] & ~U[y]
        if len(row): strana &= (np.arange(1024) < row.min()) | (np.arange(1024) > row.max())
        skryt[y] |= strana
a = (skryt * 255).astype(np.uint8)
Image.fromarray(np.dstack([np.full_like(a, 255)] * 3 + [a]), "RGBA").save(D / f"{n}-skryva-telo.webp", "WEBP", lossless=True)
j = json.load(open(D / "skryva.json")); j["telo"] = sorted(set(j["telo"]) | {n}); json.dump(j, open(D / "skryva.json", "w"), ensure_ascii=False)
print(n, int(skryt.sum()))
