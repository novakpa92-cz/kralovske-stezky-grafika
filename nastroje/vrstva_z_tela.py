"""1.0-1-6 (Pavel): oblečení namalované v ComfyUI přímo na naši holou postavu (inpaint od krku dolů) → vrstva 1024×1536.
Obrys = to, co se proti holé postavě změnilo (látka přes kůži a vše mimo obrys těla). Sedí tedy na postavu bez napasování.
python3 nastroje/vrstva_z_tela.py <png z ComfyUI> <nová vrstva, např. odev-zena-mestanske-saty> [predloha_kombinaci]
  predloha_kombinaci … ženský oděv s podobným střihem (např. odev-zena-vlnene-saty): zkopírují se masky zbrojí přes tento oděv
  a nohavice pod zbrojí (nohy-zena-…), aby oděv šel nosit pod zbroj."""
import json, shutil, sys
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "img" / "postava"


def uloz(im, name):
    im.save(P / f"{name}.webp", "WEBP", quality=90)
    im.resize((512, 768), Image.LANCZOS).save(P / f"{name}-m.webp", "WEBP", quality=88)


def uloz_masku(m, name):
    a = (m * 255).astype(np.uint8)
    Image.fromarray(np.dstack([np.full_like(a, 255)] * 3 + [a]), "RGBA").save(P / f"{name}.webp", "WEBP", lossless=True)


def main():
    src, nova = sys.argv[1:3]
    volby = sys.argv[3:]
    spodni = "spodni" in volby                     # zbroj: podklad je postava ve spodní vrstvě (košile / košilka)
    ref = next((v for v in volby if v != "spodni"), None)
    sx = "zena" if "-zena-" in nova else "muz"
    g = np.array(Image.open(src).convert("RGB").resize((1024, 1536), Image.LANCZOS)).astype(np.int16)
    tt = Image.open(P / f"postava-telo-{sx}.webp").convert("RGBA")
    if spodni:
        tt.alpha_composite(Image.open(P / ("odev-muz-lnena-kosile.webp" if sx == "muz" else "odev-zena-lnena-kosilka.webp")).convert("RGBA"))
    t = np.array(tt).astype(np.int16)
    yy = np.arange(1536)[:, None]
    telo = np.array(Image.open(P / f"postava-telo-{sx}.webp").convert("RGBA")).astype(np.int16)
    bila = (g.min(axis=2) > 222) & (g.max(axis=2) - g.min(axis=2) < 22)
    # oblast, kterou ComfyUI překreslovalo (stejně jako maska v inpaintu)
    ta = telo[..., 3] > 10
    oblast = (ndimage.binary_dilation(ta, iterations=45) & (yy > 290) & (yy < 1060)) if spodni else (ndimage.binary_dilation(ta, iterations=70) & (yy > 300))
    rg = (telo[..., 0] - telo[..., 1]) / np.maximum(telo[..., 0], 1)
    kuze = (telo[..., 3] > 200) & (rg >= .17)                     # odhalená kůže těla (ne prádlo)
    rozdil = np.abs(g - t[..., :3]).sum(axis=2)
    zustava = (kuze & (rozdil < 70))                               # kůže, kterou ComfyUI nezměnilo (ruce, nohy)
    if spodni:                                                     # košile, kterou zbroj nezakryla
        kosile = (t[..., 3] > 200) & ~kuze
        zustava |= kosile & (rozdil < 45)
    latka = oblast & ~bila & ~zustava
    latka = ndimage.binary_opening(latka, iterations=2)
    lab, n = ndimage.label(latka)
    if n:
        vel = ndimage.sum(latka, lab, range(1, n + 1))
        latka = np.isin(lab, [i + 1 for i, v in enumerate(vel) if v > vel.max() * 0.05])
    latka = ndimage.binary_closing(latka, iterations=3)
    latka = ndimage.binary_fill_holes(latka)
    a = np.asarray(Image.fromarray((latka * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))).astype(np.uint8)
    uloz(Image.fromarray(np.dstack([g.astype(np.uint8), a]), "RGBA"), nova)
    uloz_masku(ndimage.binary_erosion(latka, iterations=4), f"{nova}-skryva-telo")     # tělo pod látkou nevykoukne
    j = json.load(open(P / "skryva.json"))
    j["telo"] = sorted(set(j["telo"]) | {nova})
    if nova.startswith("zbroj-"):                  # zbroj schová i oděv pod sebou (obecná maska, kombinace se zkopírují níž)
        uloz_masku(ndimage.binary_erosion(latka, iterations=4), f"{nova}-skryva")
        j["obleceni"] = sorted(set(j["obleceni"]) | {nova})
    kopie = 0
    if ref and nova.startswith("odev-zena-"):
        ro, no = ref[len("odev-"):], nova[len("odev-"):]
        for f in sorted(P.glob(f"zbroj-zena-*__{ro}-*.webp")):
            shutil.copy(f, P / f.name.replace(f"__{ro}-", f"__{no}-")); kopie += 1
        for k in ("obleceni", "telo"):
            j[k] = sorted(set(j[k]) | {x.replace(f"__{ro}", f"__{no}") for x in j[k] if x.endswith(f"__{ro}")})
        for suf in ("", "-m"):
            if (P / f"nohy-{ro}{suf}.webp").exists():
                shutil.copy(P / f"nohy-{ro}{suf}.webp", P / f"nohy-{no}{suf}.webp")
    json.dump(j, open(P / "skryva.json", "w"), ensure_ascii=False)
    print(nova, "z těla", f"({kopie} masek zbrojí)")


if __name__ == "__main__":
    main()
