"""JEDNA vrstva (zbroj s krátkými rukávy): rukáv se protáhne dolů (a mírně ven) od ramene, aby kryl rukáv spodní vrstvy.
Trup (od X_TRUP dovnitř) se nehýbe, přechod plynulý.
python3 nastroje/rukav_prodlouzit.py <vrstva> L|P  y_rameno  y_konec_ted  y_konec_novy  posun_ven  x_trup  x_vnejsi"""
import sys
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n = sys.argv[1]; st = sys.argv[2]
Y0, Y1, Y2, DX, XT, XV = map(float, sys.argv[3:9])
k = np.array(Image.open(D / f"{n}.webp").convert("RGBA")).astype(np.float32); H, W = k.shape[:2]
ys, xs = np.mgrid[:H, :W].astype(np.float32)
odst = (XT - xs) if st == "L" else (xs - XT)
w = np.clip(odst / 25.0, 0, 1) * (ys > Y0) * (ys < Y2 + 60) * (np.abs(xs - XV) < abs(XT - XV) + 80)
s = (Y2 - Y0) / (Y1 - Y0)
sy = Y0 + (ys - Y0) / s                                             # zdroj y (protažení)
t = np.clip((ys - Y0) / (Y2 - Y0), 0, 1.2)
sx = xs + (DX if st == "L" else -DX) * t                            # posun ven: zdroj je víc uvnitř
mx = xs * (1 - w) + sx * w; my = ys * (1 - w) + sy * w
out = cv2.remap(k, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(D / f"{n}.webp", "WEBP", quality=92)
Image.fromarray(out.clip(0, 255).astype(np.uint8)).resize((512, 768), Image.LANCZOS).save(D / f"{n}-m.webp", "WEBP", quality=90)
print(n, st, "rukáv protažen", round(s, 2))
