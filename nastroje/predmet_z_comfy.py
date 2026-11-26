"""1.0-1-6: předmět z ComfyUI (bílé pozadí) → boční obrázek img/predmety/zbrane/<jméno>.webp (+ -m) a ikona img/predmety/ikony/<slug>.webp.
python3 nastroje/predmet_z_comfy.py <png> <jméno souboru bez přípony, např. zbran-rondel> <slug ikony> [otocit] [dia]
  otocit … obrázek je hlavou dolů (jílcem nahoru) — otočí se o 180°, aby hlava / čepel byla nahoře jako u ostatních zbraní
  dia    … ikona šikmo (dlouhé zbraně), jinak rovně"""
import sys
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "img" / "predmety"


def vyrez(src, prah=196):
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).astype(int)
    roh = np.median(np.concatenate([a[:8, :8].reshape(-1, 3), a[:8, -8:].reshape(-1, 3), a[-8:, :8].reshape(-1, 3), a[-8:, -8:].reshape(-1, 3)]), axis=0)
    if roh.min() > 200:                                   # bílé pozadí
        bg = (a.min(axis=2) > prah) & (a.max(axis=2) - a.min(axis=2) < 20)
    else:                                                 # barevné pozadí (zelené u bílých předmětů) — podle barvy v rozích
        bg = np.abs(a - roh).sum(axis=2) < 90
    lab, n = ndimage.label(bg)
    edge = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    edge.discard(0)
    mask = ~np.isin(lab, list(edge))
    lab2, n2 = ndimage.label(mask)                      # jen největší kus (žádné smetí kolem)
    if n2 > 1:
        sizes = ndimage.sum(mask, lab2, range(1, n2 + 1))
        mask = np.isin(lab2, [i + 1 for i, v in enumerate(sizes) if v > sizes.max() * 0.08])
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=4))   # zářezy do světlé čepele zacelit
    mask = ndimage.binary_erosion(mask, iterations=3)               # bez bílého lemu
    alpha = Image.fromarray((mask * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1.2))
    al = np.asarray(alpha).astype(float)
    al = np.clip((al - 60) * 1.5, 0, 255)
    edge = (al > 0) & (al < 250)                                    # světlé okraje ztmavit k barvě vnitřku
    a2 = a.astype(float)
    sv = a2.max(axis=2) > 200
    a2[edge & sv] *= 0.75
    im = Image.fromarray(np.clip(a2, 0, 255).astype("uint8"))
    al = al.astype("uint8")
    rgba = im.copy()
    rgba.putalpha(Image.fromarray(al))
    return rgba.crop(Image.fromarray(al).getbbox())


def main():
    src, jm, slug = sys.argv[1:4]
    opt = set(sys.argv[4:])
    prah = next((int(o[5:]) for o in opt if o.startswith("prah=")), 196)
    im = vyrez(src, prah)
    if "otocit" in opt:
        im = im.rotate(180, expand=True)
    w, h = im.size
    s = min(930 / h, 930 / w)
    im2 = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    out = Image.new("RGBA", (1024, 1024))
    out.alpha_composite(im2, ((1024 - im2.size[0]) // 2, (1024 - im2.size[1]) // 2))
    out.save(R / "zbrane" / f"{jm}.webp", "WEBP", quality=90)
    out.resize((768, 768), Image.LANCZOS).save(R / "zbrane" / f"{jm}-m.webp", "WEBP", quality=88)
    ik = im.rotate(135 if "dia" in opt else 0, expand=True, resample=Image.BICUBIC)
    ik = ik.crop(ik.getchannel("A").point(lambda v: 255 if v > 30 else 0).getbbox())
    S = int(max(ik.size) * 1.08)
    c = Image.new("RGBA", (S, S))
    c.alpha_composite(ik, ((S - ik.size[0]) // 2, (S - ik.size[1]) // 2))
    c.resize((160, 160), Image.LANCZOS).save(R / "ikony" / f"{slug}.webp", "WEBP", quality=88)
    print(jm, "ok")


if __name__ == "__main__":
    main()
