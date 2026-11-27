"""1.0-1-6 (Pavel): vrstva z obrázku FLUX Kontext. Postava byla převlečená bez změny pózy, takže vrstva = to, co se proti výchozí
postavě změnilo (výchozí = holé tělo nebo tělo ve spodní vrstvě na bílém pozadí 1024×1536). Sedí na pixel.
python3 nastroje/vrstva_kontext.py <vychozi.png> <vysledek.png> <muz|zena> <nova-vrstva> [prah=14] [bez=hlava,ruce,chodidla]
  bez … oblasti, které do vrstvy nikdy nepatří (výchozí: hlava, chodidla; ruce jen když se vrstva rukou netýká)"""
import sys
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kontrola_kontext import oblasti

P = Path(__file__).resolve().parent.parent / "img" / "postava"


def uloz(im, name):
    im.save(P / f"{name}.webp", "WEBP", quality=92)
    im.resize((512, 768), Image.LANCZOS).save(P / f"{name}-m.webp", "WEBP", quality=88)


def rysy(px):
    """Barevné rysy pixelů: chromatičnost (r/s, g/s) a jas — nezávisí moc na stínování látky."""
    px = px.astype(np.float32) + 1
    s = px.sum(-1, keepdims=True)
    return np.concatenate([px[..., :2] / s, s / 765], -1).reshape(-1, 3)


def oblasti_spodni(sx):
    """Maska spodní vrstvy (lněná košile / košilka) na výchozí postavě."""
    f = P / ("odev-muz-lnena-kosile.webp" if sx == "muz" else "odev-zena-lnena-kosilka.webp")
    if not f.exists():
        return None
    m = np.array(Image.open(f))[..., 3] > 200
    return ndimage.binary_erosion(m, iterations=3)


def vrstva(vych, vys, sx, prah=14, bez=("hlava", "chodidla"), len_ven=True, vylouc=None, jen_nad=None):
    a = np.asarray(Image.open(vych).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(vys).convert("RGB").resize((a.shape[1], a.shape[0]), Image.LANCZOS), dtype=np.float32)
    # barevná změna (rozmazané obrazy): nová látka, kov, kůže ano — jen jinak vykreslená textura stejné spodní látky ne
    ab = np.stack([ndimage.gaussian_filter(a[..., i], 3) for i in range(3)], -1)
    bb = np.stack([ndimage.gaussian_filter(b[..., i], 3) for i in range(3)], -1)
    d = np.abs(ab - bb).max(-1)
    zmena = d > 10
    bila = b.min(-1) > 226                                                # pozadí (i mírně změněné) do vrstvy nepatří
    lab, n = ndimage.label(bila)                                          # … ale jen bílá spojená s okrajem; odlesk kovu uvnitř není pozadí
    okraj = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bila = np.isin(lab, list(okraj))
    bila = ndimage.binary_dilation(bila, iterations=1)
    # barva spodní vrstvy (len): model z výchozí postavy; co má barvu lnu, je přemalovaná košile, ne zbroj
    spod = oblasti_spodni(sx) if len_ven else None
    if vylouc is not None:                                   # čepice: barva vlasů / kůže hlavy se do vrstvy nedostane
        spod = vylouc
    if spod is not None:
        X = rysy(a[spod]); mu = X.mean(0); ic = np.linalg.inv(np.cov(X.T) + np.eye(3) * 1e-4)
        Y = rysy(b.reshape(-1, 3)) - mu
        md = np.einsum("ij,jk,ik->i", Y, ic, Y).reshape(b.shape[:2])
        lnene = ndimage.gaussian_filter((md < 9.0).astype(np.float32), 1.5) > 0.5
        zmena &= ~lnene
    zmena &= ~bila
    zmena = ndimage.binary_opening(zmena, iterations=2)
    lab, n = ndimage.label(zmena)
    if n:
        s = ndimage.sum(zmena, lab, range(1, n + 1))
        zmena = np.isin(lab, [i + 1 for i, v in enumerate(s) if v > 400])
    zmena = ndimage.binary_closing(zmena, iterations=4)
    zmena = ndimage.binary_fill_holes(zmena) & ~bila                   # látka podobná spodní vrstvě nesmí mít díry
    # malé díry uvnitř zbroje (odlesk kovu podobný lnu) zalepit — velké otvory (mezi paží a tělem) zůstanou
    for _ in range(2):
        dira = ndimage.binary_closing(zmena, iterations=6) & ~zmena & ~bila
        lab, n = ndimage.label(dira)
        if n:
            s = ndimage.sum(dira, lab, range(1, n + 1))
            zmena |= np.isin(lab, [i + 1 for i, v in enumerate(s) if v < 900])
    if jen_nad is not None:
        zmena[jen_nad:] = False
    ob = oblasti(sx)
    for k in bez:
        zmena &= ~ob[k]
    alfa = Image.fromarray((zmena * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    out = Image.fromarray(np.dstack([b, np.zeros(b.shape[:2])]).astype(np.uint8), "RGBA")
    out.putalpha(alfa)
    return out, int(zmena.sum())


if __name__ == "__main__":
    vych, vys, sx, nova = sys.argv[1:5]
    prah = next((int(x[5:]) for x in sys.argv[5:] if x.startswith("prah=")), 14)
    bez = next((tuple(x[4:].split(",")) for x in sys.argv[5:] if x.startswith("bez=")), ("hlava", "chodidla"))
    im, px = vrstva(vych, vys, sx, prah, bez)
    uloz(im, nova)
    print(nova, "px", px)
