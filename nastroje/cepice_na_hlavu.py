"""1.0-1-6 (Pavel): čepice / klobouk vygenerovaný samostatně zepředu (bílé pozadí) → přesně ořezaný tvar nasazený na hlavu postavy.
Šířka podle šířky lebky, spodní okraj naražený do čela (částečně přes vlasy postavy). Zároveň ikona 160 px.
python3 nastroje/cepice_na_hlavu.py <png> <slug> <sirka> [jen_zena] [prah=N] [dolu=N]
  sirka … kolikrát je čepice širší než lebka (baret ~1,45, klobouk s krempou ~1,7, přiléhavá čepice ~1,15)"""
import sys
import numpy as np
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predmet_z_comfy import vyrez, R

P = R.parent / "postava"
CELO = {"muz": 178, "zena": 178}       # řádek, kam sedne spodní okraj čepice (nad obočím, přes vlasy u čela)


def bez_stinu(im):
    """Šedý stín z podložky: odstraní nízce sytné šedé pixely navázané na okraj (zvenku), peří a látka zůstanou."""
    from scipy import ndimage
    a = np.array(im).astype(np.float32)
    f = a[..., :3] / 255; mx, mn = f.max(axis=2), f.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-3), 0)
    stin = (a[..., 3] > 0) & (sat < 0.24) & (mx > 0.08) & (mx < 0.93)
    venku = a[..., 3] < 30
    pryc = np.zeros_like(venku)
    hranice = venku.copy()
    for _ in range(60):                                  # šíření od okraje jen přes stín
        nove = ndimage.binary_dilation(hranice, iterations=1) & stin & ~pryc
        if not nove.any():
            break
        pryc |= nove; hranice = nove
    pryc = ndimage.binary_dilation(pryc, iterations=1) & (stin | (a[..., 3] < 200))
    a[..., 3] = np.where(pryc, 0, a[..., 3])
    a[..., 3] = np.where(ndimage.binary_erosion(a[..., 3] > 30, iterations=1), a[..., 3], 0)   # o pixel užší okraj
    out = Image.fromarray(a.astype(np.uint8), "RGBA")
    return out.crop(out.getchannel("A").point(lambda v: 255 if v > 30 else 0).getbbox())


def uloz(im, name):
    im.save(P / f"{name}.webp", "WEBP", quality=90)
    im.resize((512, 768), Image.LANCZOS).save(P / f"{name}-m.webp", "WEBP", quality=88)


def lebka(sx):
    a = np.array(Image.open(P / f"postava-telo-{sx}.webp"))[..., 3] > 60
    xs = np.where(a[150])[0]
    return xs.min(), xs.max()


def spodni_okraj(im):
    """Řádek spodního okraje čepice uprostřed (medián nejnižších bodů ve střední polovině šířky)."""
    a = np.array(im)[..., 3] > 100
    w = a.shape[1]
    dna = [np.where(a[:, x])[0].max() for x in range(w // 4, 3 * w // 4) if a[:, x].any()]
    return int(np.median(dna)) if dna else a.shape[0] - 1


def main():
    src, slug, k = sys.argv[1], sys.argv[2], float(sys.argv[3])
    opt = sys.argv[4:]
    prah = next((int(o[5:]) for o in opt if o.startswith("prah=")), 205)
    dolu = next((int(o[5:]) for o in opt if o.startswith("dolu=")), 0)
    cap = bez_stinu(vyrez(src, prah))
    for sx in (("zena",) if "jen_zena" in opt else ("muz", "zena")):
        x0, x1 = lebka(sx)
        s = (x1 - x0) * k / cap.size[0]
        c = cap.resize((max(1, round(cap.size[0] * s)), max(1, round(cap.size[1] * s))), Image.LANCZOS)
        dno = spodni_okraj(c)
        out = Image.new("RGBA", (1024, 1536))
        out.alpha_composite(c, (round((x0 + x1) / 2 - c.size[0] / 2), CELO[sx] + dolu - dno))
        uloz(out, f"helma-{slug}" + ("-zena" if sx == "zena" else ""))
    S = int(max(cap.size) * 1.08)
    ik = Image.new("RGBA", (S, S))
    ik.alpha_composite(cap, ((S - cap.size[0]) // 2, (S - cap.size[1]) // 2))
    ik.resize((160, 160), Image.LANCZOS).save(R / "ikony" / f"{slug}.webp", "WEBP", quality=88)
    print(slug, "nasazeno")


if __name__ == "__main__":
    main()
