"""1.0-1-6: nastavení čepic z plátna „Čepice na hlavu“ (Pavel) → upeče posun, velikost, natočení a „schovat za hlavu“ do vrstev helma-*.
python3 nastroje/cepice_nastaveni.py 'CEPICE {...}'    (uloží do nastroje/cepice_nastaveni.json a upeče)
Natočení a zvětšení jsou kolem středu spodního okraje čepice (jako v plátně)."""
import json, sys
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

N = Path(__file__).resolve().parent
P = N.parent / "img" / "postava"
J = N / "cepice_nastaveni.json"


def uloz(im, name):
    im.save(P / f"{name}.webp", "WEBP", quality=90)
    im.resize((512, 768), Image.LANCZOS).save(P / f"{name}-m.webp", "WEBP", quality=88)


def upect(k, v):
    sx, cp = k.split("|")
    n = f"helma-{cp}" + ("-zena" if sx == "zena" else "")
    im = Image.open(P / f"{n}.webp").convert("RGBA")
    bb = im.getchannel("A").point(lambda a: 255 if a > 20 else 0).getbbox()
    c = im.crop(bb)
    if v.get("schovat"):                      # štětec „Schovat za hlavu“ / „Vrátit“ (souřadnice v ořezu čepice, plátno 1:1)
        m = Image.new("L", c.size, 255); d = ImageDraw.Draw(m); orig = c.getchannel("A")
        for pol in v["schovat"]:
            if pol and pol[0] == "p":                       # obrys (mnohoúhelník) z vylepšeného plátna
                body = [tuple(b) for b in pol[1]]
                if len(body) >= 3: d.polygon(body, fill=0 if pol[2] else 255)
                continue
            x, y, r, pryc = pol
            d.ellipse((x - r, y - r, x + r, y + r), fill=0 if pryc else 255)
        c.putalpha(Image.fromarray(np.minimum(np.array(orig), np.array(m))))
    s, r = v.get("s", 1), v.get("r", 0)
    c = c.resize((max(1, round(c.size[0] * s)), max(1, round(c.size[1] * s))), Image.LANCZOS)
    pad = int(np.hypot(*c.size)) + 4
    big = Image.new("RGBA", (pad * 2, pad * 2)); big.alpha_composite(c, (pad - c.size[0] // 2, pad - c.size[1]))
    big = big.rotate(-r, resample=Image.BICUBIC, center=(pad, pad))          # v plátně kladné = po směru hodin
    ox = (bb[0] + bb[2]) / 2 + v.get("dx", 0); oy = bb[3] + v.get("dy", 0)
    out = Image.new("RGBA", im.size); out.alpha_composite(big, (round(ox - pad), round(oy - pad)))
    uloz(out, n)
    print(n, "upečeno", v)


def main():
    txt = sys.argv[1].strip()
    nove = json.loads(txt[txt.index("{"):])
    stare = json.load(open(J)) if J.exists() else {}
    stare.update(nove); json.dump(stare, open(J, "w"), ensure_ascii=False, indent=1)
    for k, v in nove.items():
        upect(k, v)


if __name__ == "__main__":
    main()
