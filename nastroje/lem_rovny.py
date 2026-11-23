"""JEDNA vrstva: roztřepený spodní lem — od řádku Y dolů se nechá jen to, co přežije otevření velkým jádrem
(useknou se cáry a třásně), pak měkký okraj. python3 nastroje/lem_rovny.py <vrstva> <Y> [jádro]"""
import sys
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]; Y = int(sys.argv[2]); J = int(sys.argv[3]) if len(sys.argv) > 3 else 31
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA"))
al = (k[..., 3] > 100).astype(np.uint8)
op = cv2.morphologyEx(al, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (J, J)))
ys = np.arange(1536)[:, None]
pryc = (ys >= Y) & (op == 0)
k[..., 3] = np.where(pryc, 0, k[..., 3])
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, "pryč", int((pryc & (al > 0)).sum()))
