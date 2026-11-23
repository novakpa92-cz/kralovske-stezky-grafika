"""JEDNA vrstva: díry uvnitř obrysu dokreslí (okolní látkou), obvod vyhladí (bez zubů a okousání).
python3 nastroje/vyhladit.py <vrstva> [r]   r = síla vyhlazení obvodu (výchozí 7)"""
import sys
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]; r = int(sys.argv[2]) if len(sys.argv) > 2 else 7
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA"))
a = (k[..., 3] > 128).astype(np.uint8)
el = lambda s: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
tvar = cv2.morphologyEx(a, cv2.MORPH_CLOSE, el(2 * r + 1))          # zacelit zuby a zářezy
tvar = cv2.morphologyEx(tvar, cv2.MORPH_OPEN, el(r + 1 | 1))         # useknout třásně a úlomky
ff = tvar.copy(); m = np.zeros((1538, 1026), np.uint8); cv2.floodFill(ff, m, (0, 0), 2)
tvar = ((ff != 2) | (tvar > 0)).astype(np.uint8)                      # vyplnit díry uvnitř
dira = (tvar > 0) & (a == 0)                                          # kde je nově látka → dokreslit
bgr = cv2.cvtColor(k[..., :3], cv2.COLOR_RGB2BGR)
fill = cv2.cvtColor(cv2.inpaint(bgr, dira.astype(np.uint8), 9, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
k[dira, :3] = fill[dira]
alfa = cv2.GaussianBlur(tvar.astype(float), (5, 5), 1.1)              # měkký 2px okraj
k[..., 3] = (np.clip(alfa * 1.6 - .3, 0, 1) * 255).astype(np.uint8) * (cv2.dilate(tvar, el(3)) > 0)
Image.fromarray(k).save(D / f"{n}.webp", "WEBP", quality=92); Image.fromarray(k).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, "dokresleno", int(dira.sum()), "odebráno", int(((a > 0) & (tvar == 0)).sum()))
