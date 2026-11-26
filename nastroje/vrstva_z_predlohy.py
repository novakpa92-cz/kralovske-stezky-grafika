"""1.0-1-6: nová vrstva postavy z img2img (ComfyUI) podle předlohy — obrys a všechny masky „schová pod sebou“ převezme z předlohy.
python3 nastroje/vrstva_z_predlohy.py <png z ComfyUI> <nová vrstva> <předloha> [odev_tvar]
  - obraz = výstup ComfyUI, alfa = alfa předlohy (vrstva sedí na tělo přesně jako předloha)
  - helmy (výstup 512×512) se vrátí na místo hlavy (výřez 256..768 × 0..512)
  - masky <předloha>-skryva / -skryva-telo / -helma / -vlasy a kombinace <zbroj>__<oděv> se zkopírují pod novým jménem a zapíšou do skryva.json
  - u ženského oděvu i kombinace zbrojí s ním (…__zena-<oděv>) a nohy-zena-<oděv>"""
import json, sys, shutil
import numpy as np
from PIL import Image
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "img" / "postava"
HLAVA = (256, 0)


def uloz(im, name):
    im.save(P / f"{name}.webp", "WEBP", quality=90)
    im.resize((512, 768), Image.LANCZOS).save(P / f"{name}-m.webp", "WEBP", quality=88)


def main():
    src, nova, ref = sys.argv[1:4]
    g = Image.open(src).convert("RGB")
    if g.size != (1024, 1536):
        if g.size[0] == g.size[1]:                       # výřez hlavy
            c = Image.new("RGB", (1024, 1536), "white")
            c.paste(g.resize((512, 512), Image.LANCZOS), HLAVA)
            g = c
        else:
            g = g.resize((1024, 1536), Image.LANCZOS)
    r = Image.open(P / f"{ref}.webp").convert("RGBA")
    a = np.array(r)[..., 3]
    gg = np.array(g).astype(np.int16)
    bila = (gg.min(axis=2) > 225) & (gg.max(axis=2) - gg.min(axis=2) < 20)      # ComfyUI tam nechalo bílé pozadí
    from scipy import ndimage
    bila = ndimage.binary_dilation(bila, iterations=2)
    rr = np.array(r)[..., :3]
    gg = np.where(bila[..., None], rr, gg)                                       # → doplnit z předlohy
    out = np.dstack([gg.astype(np.uint8), a]).astype(np.uint8)
    uloz(Image.fromarray(out, "RGBA"), nova)
    j = json.load(open(P / "skryva.json"))
    kopie = 0
    for f in sorted(P.glob(f"{ref}*.webp")):
        rest = f.name[len(ref):]
        if f.name.endswith("-m.webp") or rest in (".webp",) or rest.startswith("-barva") or not (rest.startswith("-") or rest.startswith("__")):
            continue
        if rest.startswith("-") and not any(rest.startswith(x) for x in ("-skryva", "-helma", "-vlasy")):
            continue                                     # jiná vrstva se stejným začátkem (např. medvedi-kozich)
        shutil.copy(f, P / (nova + rest)); kopie += 1
    for k in ("obleceni", "telo"):
        pridej = {nova + x[len(ref):] for x in j[k] if x == ref or x.startswith(ref + "__")}
        j[k] = sorted(set(j[k]) | pridej)
    if nova.startswith("odev-zena-"):                    # zbroje přes nový oděv: masky jako přes předlohu
        ro, no = ref[len("odev-"):], nova[len("odev-"):]
        for f in sorted(P.glob(f"zbroj-zena-*__{ro}-*.webp")):
            shutil.copy(f, P / f.name.replace(f"__{ro}-", f"__{no}-")); kopie += 1
        for k in ("obleceni", "telo"):
            j[k] = sorted(set(j[k]) | {x.replace(f"__{ro}", f"__{no}") for x in j[k] if x.endswith(f"__{ro}")})
        nh = P / f"nohy-{ro}.webp"
        if nh.exists():
            for suf in ("", "-m"):
                shutil.copy(P / f"nohy-{ro}{suf}.webp", P / f"nohy-{no}{suf}.webp")
    json.dump(j, open(P / "skryva.json", "w"), ensure_ascii=False)
    print(nova, "←", ref, f"({kopie} masek)")


if __name__ == "__main__":
    main()
