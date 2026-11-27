"""1.0-1-6 noc: výchozí obrázky a masky pro čepice (každý účes zvlášť) a helmy (holá hlava).
python3 nastroje/hlavy_vstupy.py <výstupní složka>"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
from pathlib import Path
P = Path(__file__).resolve().parent.parent / "img" / "postava"
VLASY = ["kratke", "dlouhe", "kudrnate", "ofina", "copanek", "stredni", "tonzura", "bez"]


def zaklad(sx, vl):
    c = Image.new("RGBA", (1024, 1536), (255, 255, 255, 255))
    for v in ["postava-telo-" + sx] + (["odev-muz-lnena-kosile"] if sx == "muz" else ["odev-zena-lnena-kosilka", "nohy-zena-lnena-kosilka"]) + ([f"vlasy-{sx}-{vl}"] if vl != "bez" else []):
        c.alpha_composite(Image.open(P / f"{v}.webp").convert("RGBA"))
    if vl == "tonzura" and sx == "muz" and (P / "vlasy-muz-tonzura-kuze.webp").exists():
        c.alpha_composite(Image.open(P / "vlasy-muz-tonzura-kuze.webp").convert("RGBA"))
    return c.convert("RGB")


def oci(sx):
    t = np.array(Image.open(P / f"postava-telo-{sx}.webp").convert("RGB")).astype(float)
    reg = t[150:260, 470:560].mean(-1).mean(1)
    return 150 + int(np.argmin(reg[10:90])) + 10


def maska_cepice(sx, vl):
    a = np.array(Image.open(P / f"postava-telo-{sx}.webp"))[..., 3] > 128
    hl = a.copy(); hl[300:] = False
    if vl != "bez":
        hl |= np.array(Image.open(P / f"vlasy-{sx}-{vl}.webp"))[..., 3] > 60
    hl[330:] = False                                       # dlouhé vlasy na zádech čepice neřeší
    y_oci = oci(sx); obocí = y_oci - 16
    m = ndimage.binary_dilation(hl, iterations=55)        # krempa / baret do šířky a nad hlavu
    m[:20] = False; m[obocí + 75:] = False
    # obličej od obočí dolů zamčený (elipsa), po stranách se smí (cípy přes uši)
    row = np.nonzero(a[y_oci])[0]; cx = (row.min() + row.max()) / 2; rx = (row.max() - row.min()) / 2 - 10
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    tvar = (((xx - cx) / rx) ** 2 + ((yy - (obocí + 95)) / 110) ** 2) <= 1
    m &= ~(tvar & (yy > obocí))
    return m


def maska_helmy(sx):
    a = np.array(Image.open(P / f"postava-telo-{sx}.webp"))[..., 3] > 128
    ys = np.nonzero(a.any(1))[0]; top = ys.min()
    hl = a.copy(); hl[top + 210:] = False                  # hlava a krk
    m = ndimage.binary_dilation(hl, iterations=60); m[:15] = False; m[top + 235:] = False
    return m


if __name__ == "__main__":
    o = Path(sys.argv[1]); o.mkdir(parents=True, exist_ok=True)
    for sx in ("muz", "zena"):
        for vl in VLASY:
            zaklad(sx, vl).save(o / f"ksin-hlava-{sx}-{vl}.png")
            Image.fromarray((maska_cepice(sx, vl) * 255).astype(np.uint8)).convert("RGB").save(o / f"ksmask-cepice-{sx}-{vl}.png")
        Image.fromarray((maska_helmy(sx) * 255).astype(np.uint8)).convert("RGB").save(o / f"ksmask-helma-{sx}.png")
        print(sx, "hotovo")
