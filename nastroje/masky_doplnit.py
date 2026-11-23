"""r27-3: masky „schová pod sebou“ u zbrojí doplní okraje, aby oděv (kožich, kožíšek…) nevyčuhoval
po stranách rukávů a boků, u zápěstí a — u zbrojí s límcem — nad krkem a na ramenou.
Spouštět z kořene repa: python3 nastroje/masky_doplnit.py (zálohy původních masek si drž zvlášť)."""
import json
import numpy as np
from PIL import Image
from pathlib import Path

D = Path("img/postava")
LIMEC = ("platova", "goticky-plat", "cerny-plat", "platovy-kyrys")   # zbroje s rukávy a límcem/obojkem
VODOROVNE, DOKOLA = 0, 8

def roztahni(a, rx, ry):
    out = a.copy()
    for d in range(1, rx + 1):
        out[:, d:] |= a[:, :-d]; out[:, :-d] |= a[:, d:]
    b = out.copy()
    for d in range(1, ry + 1):
        out[d:, :] |= b[:-d, :]; out[:-d, :] |= b[d:, :]
    return out

sez = json.load(open(D / "skryva.json"))
for name in sez["obleceni"]:
    if not name.startswith("zbroj-"):
        continue
    mf = D / f"{name}-skryva.webp"
    if not mf.exists() or not any(name.endswith(k) for k in LIMEC):
        continue          # vesty (brigantiny, kožená…) nechávají rukávy oděvu — jejich masky se nemění
    arm = np.array(Image.open(D / f"{name}.webp").convert("RGBA"))[:, :, 3] > 100
    arm = roztahni(arm, 3, 3)                       # kroužky mají dírky — nejdřív je zacelit
    old = np.array(Image.open(mf).convert("RGBA"))[:, :, 3] > 100
    new = old | roztahni(arm, DOKOLA, 0)
    if any(name.endswith(k) for k in LIMEC):
        top = np.where(arm.any(axis=0), arm.argmax(axis=0), arm.shape[0])
        ys = np.arange(arm.shape[0])[:, None]
        nad = (ys < top[None, :]) & (ys >= 150) & arm.any(axis=0)[None, :]
        new |= roztahni(nad, 45, 0)
    # pod konci rukávů (paže mimo trup) schovat i manžety oděvu — kožešina by jinak visela u zápěstí
    H, W = arm.shape
    cols = np.where(np.abs(np.arange(W) - W // 2) > 225)[0]
    for x in cols:
        ys = np.where(arm[:, x])[0]
        if len(ys):
            new[ys.max():min(H, ys.max() + 55), x] = True
    a = (new * 255).astype(np.uint8)
    img = Image.fromarray(np.dstack([np.full_like(a, 255)] * 3 + [a]), "RGBA")
    img.save(mf, "WEBP", lossless=True)
    print(name, int(old.sum()), "→", int(new.sum()))
