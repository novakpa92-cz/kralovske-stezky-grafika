"""Dosazení čepic a helem na hlavu (Pavel 27. 11.: „vše musí vypadat přirozeně, žádné zuby, žádné chyby“).
Čepice a otevřené helmy: spodní okraj na čele kousek nad obočím, temeno nad ním celé zakryté.
Helmy přes obličej: průzor zůstane na očích (kotva = oči), zvětší se jen tolik, aby zakryly lebku.
Převzorkování s předem vynásobenou alfou (žádné tmavé ani světlé lemy, žádné zuby)."""
import sys, os, glob, json, numpy as np
from PIL import Image
D = os.path.join(os.path.dirname(__file__), "..", "img", "postava") + "/"
NECHAT = {"venecek", "batvat", "havirska", "krouzkova-kukla", "kozena-cepice", "velka", "rytirsky-helm", "bascinet"}
OBLICEJ = {"barbuta", "bascinet-s-psi-kuklou"}
OCI = {"zena": 202, "muz": 196}; OBOCI = {"zena": 193, "muz": 187}
UCESY = ("bez", "copanek", "dlouhe", "kratke", "kudrnate", "ofina", "stredni", "tonzura")
def hlava(sx):
    a = np.array(Image.open(D + f"postava-telo-{sx}.webp").convert("RGBA"))[..., 3] > 60
    a[:, :380] = False; a[:, 644:] = False; a[330:] = False
    xs = np.where(a[150])[0]; return a, (xs.min() + xs.max()) / 2
def tf(arr, s, dy, ax, ay, res=Image.BICUBIC):
    im = Image.fromarray(arr)
    return np.array(im.transform(im.size, Image.AFFINE, (1 / s, 0, ax - ax / s, 0, 1 / s, ay - (ay + dy) / s), resample=res))
def okraj(al, cx):
    ys = np.where(al[:, int(cx) - 20:int(cx) + 20].max(1) > 128)[0]; return int(ys.max()) if len(ys) else 0
def dosad(n, sx, slug):
    H, cx = hlava(sx)
    im = np.array(Image.open(D + n + ".webp").convert("RGBA")).astype(np.float32)
    al = im[..., 3].astype(np.uint8)
    if slug in OBLICEJ:
        ay = OCI[sx]; hran = OCI[sx] - 18; lebka = H.copy(); lebka[hran:] = False
        moz = [(s, dy) for s in np.arange(1.0, 1.41, 0.01) for dy in range(-10, 11, 2)]
    else:
        by = okraj(al, cx); ay = by; T = OBOCI[sx] - 14
        lebka = H.copy(); lebka[T - 6:] = False          # temeno celé zakryté, okraj smí být mírně zvlněný
        moz = [(s, dy) for s in np.arange(1.0, 1.51, 0.01) for dy in range(T - 10 - by, T - by + 5)]   # okraj do pásu [T-10, T+4]
        if T - by < -10: moz = [(s, dy) for s in np.arange(1.0, 1.51, 0.01) for dy in range(T - by - 10, 1)]
    # rychlé hledání v polovičním rozlišení; temeno musí být zakryté celé (tvrdá podmínka), pak co nejmenší změna
    h2 = lambda x: np.array(Image.fromarray(x).resize((512, 768), Image.BILINEAR))
    al2 = h2(al); leb2 = h2(lebka.astype(np.uint8) * 255) > 127
    best = None
    for s, dy in moz:
        A = np.array(Image.fromarray(al2).transform((512, 768), Image.AFFINE, (1 / s, 0, cx / 2 - cx / 2 / s, 0, 1 / s, ay / 2 - (ay / 2 + dy / 2) / s), resample=Image.BILINEAR)) > 110
        odkryto = int((leb2 & ~A).sum()) * 4
        cena = (0 if odkryto <= 0 else 100000 + odkryto * 50) + (s - 1) * 2500 + abs(dy) * 2
        if best is None or cena < best[0]: best = (cena, s, dy, odkryto)
    _, s, dy, odk = best
    P = im.copy(); P[..., :3] *= P[..., 3:4] / 255
    M = (1 / s, 0, cx - cx / s, 0, 1 / s, ay - (ay + dy) / s)
    Q = np.dstack([np.array(Image.fromarray(P[..., c]).transform((1024, 1536), Image.AFFINE, M, resample=Image.BICUBIC)) for c in range(4)])
    a = Q[..., 3:4].clip(0, 255); rgb = np.where(a > 0.5, Q[..., :3] * 255 / np.maximum(a, 1e-3), 0)
    out = np.dstack([rgb, a]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out), dict(s=round(float(s), 2), dy=int(dy), odkryto=odk)
if __name__ == "__main__":
    out_dir = sys.argv[1]; vys = {}
    for f in sorted(glob.glob(D + "helma-*.webp")):
        n = os.path.basename(f)[:-5]
        if n.endswith(("-m", "-skryva")) or n.endswith(tuple("-" + u for u in UCESY)): continue
        fem = n.endswith("-zena"); slug = n[6:-5] if fem else n[6:]
        if slug in NECHAT: continue
        o, info = dosad(n, "zena" if fem else "muz", slug)
        o.save(f"{out_dir}/{n}.webp", lossless=True); vys[n] = info; print(n, info, flush=True)
    json.dump(vys, open(f"{out_dir}/posuny.json", "w"), indent=1)
