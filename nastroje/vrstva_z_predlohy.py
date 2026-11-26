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
    vlastni = "vlastni" in sys.argv[4:]        # 1.0-1-6: obrys z vygenerovaného obrázku (klobouky jiného tvaru než předloha)
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
    if vlastni:                                          # klobouk: vše nebílé nad čelem (pod ním jen mimo obrys těla)
        sx = "zena" if nova.endswith("-zena") else "muz"
        t = np.array(Image.open(P / f"postava-telo-{sx}.webp").convert("RGBA"))
        gg2 = np.array(g).astype(np.int16)                  # původní výstup (bez doplnění z předlohy)
        nebila = ~((gg2.min(axis=2) > 215) & (gg2.max(axis=2) - gg2.min(axis=2) < 25))
        yy = np.arange(1536)[:, None]
        hl = nebila & ((yy < 192) | (t[..., 3] < 20)) & (yy < 250)      # krempa po stranách, ne vlasy na ramenou
        hl = ndimage.binary_opening(hl, iterations=2)
        lab, nn = ndimage.label(hl)
        if nn:
            vel = ndimage.sum(hl, lab, range(1, nn + 1)); hl = lab == (1 + int(np.argmax(vel)))
        hl = ndimage.binary_fill_holes(hl)
        a = (np.asarray(Image.fromarray((hl * 255).astype(np.uint8)).filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(1.0)))).astype(np.uint8)
    elif nova.startswith("helma-"):                        # obličej postavy (kůže) ve výřezu helmy zůstane vidět
        sx = "zena" if nova.endswith("-zena") else "muz"
        t = np.array(Image.open(P / f"postava-telo-{sx}.webp").convert("RGBA")).astype(np.int16)
        skin = (t[..., 3] > 200) & (np.abs(gg - t[..., :3]).sum(axis=2) < 70)
        skin = ndimage.binary_opening(skin, iterations=2)
        skin[a < 10] = False
        a = np.where(ndimage.binary_erosion(~skin, iterations=1) | (a < 10), a, 0).astype(np.uint8)
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
