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
a = (skryt * 255).astype(np.uint8)
Image.fromarray(np.dstack([np.full_like(a, 255)] * 3 + [a]), "RGBA").save(D / f"{n}-skryva-telo.webp", "WEBP", lossless=True)
j = json.load(open(D / "skryva.json")); j["telo"] = sorted(set(j["telo"]) | {n}); json.dump(j, open(D / "skryva.json", "w"), ensure_ascii=False)
print(n, int(skryt.sum()))
