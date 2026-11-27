"""1.0-1-6 noční běh: z dvojic variant (…-1, …-2) vybere tu, která projde kontrolou s nejmenší odchylkou, a vyřízne vrstvu.
python3 nastroje/zpracuj_noc.py <složka s výsledky> <předpona…>   (např. kz-zbroj-zena-milanska-zbroj)
Zapisuje nastroje/noc_vysledky.json."""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kontrola_kontext import kontrola, zarovnej
from vrstva_kontext import vrstva, uloz

IN = Path("/mnt/user-data/outputs/ks-in")
J = Path(__file__).resolve().parent / "noc_vysledky.json"


def zpracuj_hlavu(slozka, pref):
    """kh-helma-<sx>-<slug> / kh-cepice-<sx>-<slug>-<vlas>: vrstva helma-<slug>(-zena)[-<vlas>]"""
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    from vrstva_kontext import P
    c = pref.split("-"); druh, sx = c[1], c[2]
    if druh == "helma":
        slug, vl = "-".join(c[3:]), "bez"
    else:
        vl = c[-1]; slug = "-".join(c[3:-1])
    zaklad = IN / f"ksin-hlava-{sx}-{vl}.png"
    povol = ("hlava", "oblicej") if druh == "helma" and slug != "zelezna-cepice" else ("hlava",)   # otevřená helma nesmí měnit obličej
    kand = []
    for f in sorted(glob.glob(f"{slozka}/{pref}-[0-9]*_00001_.png")):
        ok, v = kontrola(zaklad, f, sx, povol)
        kand.append((ok, sum(x for k, x in v.items() if k not in povol), f, v))
    vys = json.load(open(J)) if J.exists() else {}
    dobre = sorted([k for k in kand if k[0]], key=lambda k: k[1])
    if not dobre:
        vys[pref] = {"stav": "neproslo", "varianty": [[Path(f).name, v] for _, _, f, v in kand]}
        json.dump(vys, open(J, "w"), ensure_ascii=False, indent=1)
        print(pref, "NEPROŠLO", [v for *_, v in kand]); return None
    _, _, f, v = dobre[0]
    vyl = None
    if druh == "cepice" and vl != "bez":
        vyl = ndimage.binary_erosion(np.array(Image.open(P / f"vlasy-{sx}-{vl}.webp"))[..., 3] > 200, iterations=2)
    im, px = vrstva(zaklad, f, sx, bez=("chodidla", "ruce"), len_ven=False, vylouc=vyl, jen_nad=520)
    nazev = f"helma-{slug}" + ("-zena" if sx == "zena" else "") + (f"-{vl}" if druh == "cepice" else "")
    uloz(im, nazev)
    vys[pref] = {"stav": "ok", "vrstva": nazev, "zdroj": Path(f).name, "odchylky": v, "px": px}
    json.dump(vys, open(J, "w"), ensure_ascii=False, indent=1)
    print(pref, "OK →", nazev, v)
    return nazev


def zpracuj(slozka, pref):
    if pref.startswith("kh-"):
        return zpracuj_hlavu(slozka, pref)
    _, druh, sx, *rest = pref.split("-")
    slug = "-".join(rest)
    zaklad = IN / (f"ksin-telo-{sx}.png" if druh == "odev" else f"ksin-spodni-{sx}.png")
    povol = ("chodidla",) if druh in ("odev", "plast") or slug == "milanska-zbroj" else ()
    kand = []
    for f in sorted(glob.glob(f"{slozka}/{pref}-[0-9]*_00001_.png")):
        ok, v = kontrola(zaklad, f, sx, povol)
        kand.append((ok, sum(x for k, x in v.items() if k not in povol), f, v))
    vys = json.load(open(J)) if J.exists() else {}
    dobre = sorted([k for k in kand if k[0]], key=lambda k: k[1])
    if not dobre:
        vys[pref] = {"stav": "neproslo", "varianty": [[Path(f).name, v] for _, _, f, v in kand]}
        json.dump(vys, open(J, "w"), ensure_ascii=False, indent=1)
        print(pref, "NEPROŠLO", [v for *_, v in kand]); return None
    bez = ("hlava",) if druh == "odev" else ("hlava", "chodidla") if slug != "milanska-zbroj" else ("hlava",)
    minpx = 200000 if slug == "milanska-zbroj" else 100000 if druh == "plast" else 60000 if druh == "zbroj" else 150000
    for _, _, f, v in dobre:                        # kontrola těla nestačí: předmět musí být opravdu nakreslený
        im, px = vrstva(zaklad, f, sx, bez=bez, len_ven=(druh != "odev"))
        if px >= minpx:
            break
    else:
        vys[pref] = {"stav": "nic_nenakresleno", "varianty": [[Path(f).name, v] for _, _, f, v in dobre], "px": px}
        json.dump(vys, open(J, "w"), ensure_ascii=False, indent=1)
        print(pref, "PŘEDMĚT CHYBÍ", px); return None
    nazev = f"{druh}-{sx}-{slug}"
    uloz(im, nazev)
    vys[pref] = {"stav": "ok", "vrstva": nazev, "zdroj": Path(f).name, "odchylky": v, "px": px}
    json.dump(vys, open(J, "w"), ensure_ascii=False, indent=1)
    print(pref, "OK →", nazev, v)
    return nazev


if __name__ == "__main__":
    for p in sys.argv[2:]:
        zpracuj(sys.argv[1], p)
