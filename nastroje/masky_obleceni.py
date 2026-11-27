"""1.0-1-6 (Pavel): masky pro přemalování oblečení na naši postavu tak, aby se nehnula póza.
Co je v masce, AI smí přemalovat; ruce, obličej (a podle druhu i předloktí / chodidla) jsou mimo masku → zůstanou na pixel.
python3 nastroje/masky_obleceni.py <výstupní složka>   → ksmask-<druh>-<muz|zena>.png (bílá = malovat)"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kontrola_kontext import oblasti

P = Path(__file__).resolve().parent.parent / "img" / "postava"


def paze_pod_lokty(t):
    """Řádky, kde jsou paže oddělené od trupu: vrací (masku paží, rozsah trupu po řádcích)."""
    H, W = t.shape
    paze = np.zeros_like(t); trup = {}
    for y in range(H):
        r = np.nonzero(t[y])[0]
        if len(r) == 0:
            continue
        seg = np.split(r, np.nonzero(np.diff(r) > 1)[0] + 1)
        if len(seg) >= 3:
            for s in (seg[0], seg[-1]):
                paze[y, s[0]:s[-1] + 1] = True
            trup[y] = (seg[1][0], seg[-2][-1])
    return paze, trup


def masky(sx):
    t = np.array(Image.open(P / f"postava-telo-{sx}.webp"))[..., 3] > 128
    H, W = t.shape
    ob = oblasti(sx)
    ys = np.nonzero(t.any(1))[0]; top, bot = ys.min(), ys.max()
    krk = top + 175                                       # pod bradou
    ruce = ndimage.binary_dilation(ob["ruce"], iterations=28)        # latentní maska má měkký okraj — ruce chránit s rezervou
    hlava = np.zeros_like(t); hlava[:krk] = True
    chodidla = ndimage.binary_dilation(ob["chodidla"], iterations=6)
    paze, trup = paze_pod_lokty(t)
    y_lok = min(trup) if trup else H
    telo = ndimage.binary_dilation(t, iterations=45)
    out = {}
    # trup: krk – boky/stehna, paže od lokte dolů zamčené (i pozadí vedle nich)
    m = telo.copy(); m[:krk] = False; m[1060:] = False; m &= ~ruce
    for y in range(max(y_lok + 30, 575), 1060):          # od loktů dolů jen mezera mezi vnitřními okraji paží
        r = np.nonzero(t[y])[0]
        if len(r) == 0:
            continue
        seg = np.split(r, np.nonzero(np.diff(r) > 1)[0] + 1)
        stred = [s for s in seg if s[-1] >= W // 2 - 170 and s[0] <= W // 2 + 170]      # trup / nohy, ne paže a prsty
        if stred:
            row = np.zeros(W, bool); row[max(0, stred[0][0] - 12): stred[-1][-1] + 13] = True; m[y] &= row
    m = ndimage.binary_opening(m, structure=np.ones((9, 9))) & ~ndimage.binary_dilation(ruce, iterations=8)
    out["trup"] = m
    # rukávy: jako trup, ale paže smí až po zápěstí (ruce zamčené)
    m = telo.copy(); m[:krk] = False; m[1060:] = False; m &= ~ruce
    out["rukavy"] = m
    # celá: rukávy + nohy až nad chodidla
    m = telo.copy(); m[:krk] = False; m &= ~ruce; m &= ~chodidla
    out["cela"] = m
    # oděv na holé tělo: celé tělo pod krkem (i boty), sukně do šířky
    m = ndimage.binary_dilation(t, iterations=70); m[:krk] = False; m &= ~ruce
    out["odev"] = m
    # plášť: široký pás od ramen dolů za tělem i přes paže, ruce a hlava zamčené
    m = ndimage.binary_dilation(t, iterations=110); m[:krk - 10] = False; m &= ~ruce
    out["plast"] = m
    return out


if __name__ == "__main__":
    o = Path(sys.argv[1]); o.mkdir(parents=True, exist_ok=True)
    for sx in ("muz", "zena"):
        for k, m in masky(sx).items():
            Image.fromarray((m * 255).astype(np.uint8)).convert("RGB").save(o / f"ksmask-{k}-{sx}.png")
            print(k, sx, int(m.sum()))
