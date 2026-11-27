"""1.0-1-6 (Pavel): automatická kontrola obrázku z FLUX Kontext proti výchozí postavě.
Výsledek smí změnit jen to, co se obléká; obličej, ruce, chodidla a pozadí musí zůstat na místě.
python3 nastroje/kontrola_kontext.py <vychozi.png> <vysledek.png> <muz|zena> [povol=ruce,hlava,chodidla]
Vypíše odchylky po oblastech a PROSLO / NEPROSLO (návratový kód 0 / 1)."""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "img" / "postava"
PRAH = {"oblicej": 8.0, "hlava": 9.0, "ruce": 11.0, "chodidla": 11.0, "pozadi": 5.0, "paze": 3.0}   # průměrná odchylka (0–255) v oblasti


def oblasti(sx):
    """Masky oblastí podle alfy holého těla 1024×1536."""
    t = np.array(Image.open(P / f"postava-telo-{sx}.webp"))[..., 3] > 128
    H, W = t.shape
    ys = np.nonzero(t.any(1))[0]
    top, bot = ys.min(), ys.max()
    o = {}
    hlava = np.zeros_like(t); hlava[top:top + 165] = True; o["hlava"] = hlava & ndimage.binary_dilation(t, iterations=4)
    # ruce = spodní konce postranních segmentů (paže oddělené od trupu)
    ruce = np.zeros_like(t)
    for y in range(500, bot):
        r = np.nonzero(t[y])[0]
        if len(r) == 0:
            continue
        seg = np.split(r, np.nonzero(np.diff(r) > 1)[0] + 1)
        if len(seg) >= 3:
            for s in (seg[0], seg[-1]):
                ruce[y, s[0]:s[-1] + 1] = True
    lab, n = ndimage.label(ruce)
    hr = np.zeros_like(t)
    for i in range(1, n + 1):
        yy = np.nonzero((lab == i).any(1))[0]
        if len(yy) > 80:                                  # dlaň = spodních 110 px paže
            m = (lab == i).copy(); m[: yy.max() - 110] = False; hr |= m
    o["ruce"] = ndimage.binary_dilation(hr, iterations=3)
    ch = np.zeros_like(t); ch[bot - 110: bot + 1] = True; o["chodidla"] = ch & ndimage.binary_dilation(t, iterations=4)
    o["pozadi"] = ~ndimage.binary_dilation(t, iterations=110)
    # obličej od obočí dolů (oči, nos, ústa) — čepice ho nesmí změnit
    tt = np.array(Image.open(P / f"postava-telo-{sx}.webp").convert("RGB")).astype(float)
    reg = tt[150:260, 470:560].mean(-1).mean(1); y_oci = 150 + int(np.argmin(reg[10:90])) + 10
    row = np.nonzero(t[y_oci])[0]; cx = (row.min() + row.max()) / 2; rx = (row.max() - row.min()) / 2 - 18
    yy, xx = np.mgrid[:H, :W]
    o["oblicej"] = ((((xx - cx) / rx) ** 2 + ((yy - (y_oci + 32)) / 46) ** 2) <= 1) & (yy > y_oci - 10)
    return o


def silueta(im):
    """Postava na bílém pozadí: největší souvislá nebílá oblast."""
    a = np.asarray(im.convert("RGB"))
    m = a.min(-1) < 232
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    if not n:
        return m
    s = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (np.argmax(s) + 1)


def zarovnej(vych, vys):
    """Kontext občas postavu přiblíží nebo posune — vrátí výsledek v rámu výchozí postavy (podle temene, chodidel a středu hlavy)."""
    A, B = Image.open(vych).convert("RGB"), Image.open(vys).convert("RGB")
    W, H = A.size
    B = B.resize((W, round(B.height * W / B.width)), Image.LANCZOS) if B.width != W else B
    sa, sb = silueta(A), silueta(B)
    def body(s):
        ys = np.nonzero(s.any(1))[0]; top, bot = ys.min(), ys.max()
        hx = np.nonzero(s[top:top + 90].any(0))[0]
        return top, bot, (hx.min() + hx.max()) / 2
    ta, ba, xa = body(sa); tb, bb, xb = body(sb)
    k = (ba - ta) / max(1, (bb - tb))
    # bod v B: x_b = (x_a - xa) / k + xb ; y_b = (y_a - ta) / k + tb
    C = B.transform((W, H), Image.AFFINE, (1 / k, 0, xb - xa / k, 0, 1 / k, tb - ta / k), resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return C, round(k, 4), (round(xa - xb, 1), round(ta - tb, 1))


def kontrola(vych, vys, sx, povol=()):
    a = np.asarray(Image.open(vych).convert("RGB"), dtype=np.float32)
    C = vys if isinstance(vys, Image.Image) else Image.open(vys)
    b = np.asarray(C.convert("RGB").resize((a.shape[1], a.shape[0]), Image.LANCZOS), dtype=np.float32)
    d = np.abs(a - b).mean(-1)
    vysl, ok = {}, True
    # paže se nesmí přesunout: kde v těle byla paže / tělo, nesmí teď být bílé pozadí (procento pixelů těla)
    t = np.array(Image.open(P / f"postava-telo-{sx}.webp"))[..., 3] > 200
    t = ndimage.binary_erosion(t, iterations=3)
    bile = (b.min(-1) > 236) & (a.min(-1) < 200)                     # zbělalo, co bílé nebylo
    vysl["paze"] = round(100.0 * float((t & bile).sum()) / max(1, int(t.sum())), 2)
    if "paze" not in povol and vysl["paze"] > PRAH["paze"]:
        ok = False
    for k, m in oblasti(sx).items():
        v = float(d[m].mean()) if m.any() else 0.0
        vysl[k] = round(v, 1)
        if k not in povol and v > PRAH[k]:
            ok = False
    return ok, vysl


if __name__ == "__main__":
    vych, vys, sx = sys.argv[1:4]
    povol = next((x[6:].split(",") for x in sys.argv[4:] if x.startswith("povol=")), [])
    C, k, posun = zarovnej(vych, vys)
    ok, v = kontrola(vych, C, sx, povol)
    print("zarovnání: měřítko", k, "posun", posun)
    print(("PROSLO" if ok else "NEPROSLO"), v)
    sys.exit(0 if ok else 1)
