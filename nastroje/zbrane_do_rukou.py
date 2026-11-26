"""Zbraně, štíty a louč do rukou postavy (r27-4).
Z pěstí od ChatGPT (faze-4-postava/ruce, zelená tyč = osa úchopu) a ikon zbraní (img/predmety/zbrane) upeče
vrstvy 1024×1536 přesně na tělo muže a ženy:
  postava/ruka-pest-<prava|leva>-<sx>.webp, rukavice-pest-<…>-muz.webp   sevřená pěst (zelená vyříznutá)
  postava/ruka-dlan-<prava|leva>-<sx>.webp (+ -skryva, -skryva-telo)       prázdná vrstva, jejíž masky schovají otevřenou dlaň
  postava/zbran-ruka-<slug>-<sx>.webp                                      zbraň v pěsti (pod pěstí)
  postava/stit-ruka-<slug>-<sx>.webp, svetlo-louc-ruka-<sx>.webp
python3 nastroje/zbrane_do_rukou.py <složka s pěstmi PNG>"""
import sys, json
import numpy as np, cv2
from PIL import Image
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "img"
POS, ZBR = R / "postava", R / "predmety" / "zbrane"
SRC = Path(sys.argv[1])
PX = 7.5                      # px na cm (postava ~ 175 cm)
ZEM = 1495                    # řádek země (chodidla)
# druh: visi = rukojeť v pěsti, čepel/hlava dolů · stoji = svisle, patka na zemi · luk = za střed v levé · kuse = za pažbu, luček dolů
ZBRANE = {
  "dlouhy-mec": ("visi", 100), "kratky-mec": ("visi", 75), "jeskuv-mec": ("visi", 95), "cepel-cerneho-rytire": ("visi", 105),
  "bastard": ("visi", 112), "dvourucni-mec": ("visi", 125), "savle": ("visi", 88), "tesak": ("visi", 60), "krysaruv-tesak": ("visi", 62),
  "dyka": ("visi", 40), "palcat": ("visi", 70), "hejtmanuv-palcat": ("visi", 75), "sekyrka": ("visi", 60), "valecne-kladivo": ("visi", 72),
  "dutohlavova-sekera": ("stoji", 120), "prak": ("visi", 75),
  "halapartna": ("stoji", 195), "kopi": ("stoji", 215), "sudlice": ("stoji", 205), "vidle": ("stoji", 165), "kosa": ("stoji", 165),
  "cep": ("stoji", 150), "obourucni-sekera": ("stoji", 120), "obourucni-kladivo": ("stoji", 120),
  "dlouhy-luk": ("luk", 180), "kratky-luk": ("luk", 120), "kompozitni-luk": ("luk", 110),
  # 1.0-1-6
  "rondel": ("visi", 45), "lovecky-tesak": ("visi", 70), "jezdecky-mec": ("visi", 95), "remdih": ("visi", 75), "bojova-sekera": ("visi", 70),
  "cakan": ("visi", 75), "rytirsky-mec": ("visi", 110), "kropac": ("stoji", 140), "partyzana": ("stoji", 205), "dvourucni-cakan": ("stoji", 175),
  "valecny-luk": ("luk", 185), "kuse-s-heverem": ("kuse", 95),
  "lehka-kuse": ("kuse", 78), "tezka-kuse": ("kuse", 92), "arbalest": ("kuse", 88), "hakovnice": ("kuse", 100), "pistala": ("kuse", 62),
}
# 1.0-1-6: polohy podle Pavla — jednoruční zbraně čepelí šikmo nahoru přes tělo, dýky víc šikmo k opasku,
# obouruční meče hrotem na zemi (ruka na jílci), ostatní obouruční svisle patou na zemi (stoji)
DYKY = {"dyka", "rondel", "tesak", "krysaruv-tesak", "lovecky-tesak"}
DVOU_MECE = {"dlouhy-mec", "jeskuv-mec", "cepel-cerneho-rytire", "dvourucni-mec"}
NAKLON_JEDNORUCNI, NAKLON_DYKA, NAKLON_LEVA, NAKLON_KUSE = 34, 50, 36, 35
LEVA = ("dyka", "rondel", "tesak", "lovecky-tesak", "krysaruv-tesak")   # 1.0-1-6: boj dvěma zbraněmi — druhá zbraň v levé ruce (zrcadlově)
STIT_NAKLON = 15               # Pavel 21:31: štít níž před levým bokem, víc nakloněný
STIT_POSUN = (-65, 40)         # střed štítu vůči levé pěsti (px)
NAKLON_SE_STITEM = 47
HLAVICE = {"sekyrka", "palcat", "hejtmanuv-palcat", "valecne-kladivo", "remdih", "bojova-sekera", "cakan"}   # topůrko k opasku
NAKLON_HLAVICE = 52
SEKERY = {"sekyrka", "bojova-sekera", "cakan"}   # Pavel: u nakloněných seker ostří vždy k zemi
def ostri_vlevo(im):
    """Je ostří (větší část hlavice) vlevo od topůrka? Hlavice = horní čtvrtina obrázku."""
    a = np.array(im)[..., 3] > 80; H = a.shape[0]
    osa = stred_x(im, .7) * a.shape[1]
    hl = a[: H // 4]; xs = np.where(hl)[1]
    return len(xs) and (xs < osa).sum() > (xs > osa).sum()          # jednoruční zbraň se štítem: víc šikmo přes tělo (vrstvy zbran-ruka-<slug>-<sx>-stit)                # 1.0-1-6: štíty mírně nakloněné horní hranou k tělu (stupně, proti směru hodin)
STITY = {"stit-pestni": 34, "stit-jezdecky": 64, "stit-malovana-paveza": 105, "stit-mistrovsky": 72, "stit-lehky-dreveny": 58, "stit-okovany": 66, "stit-paveza": 100, "stit-rytirsky": 70, "stit-tezky-dreveny": 66}

def uloz(img, name):
    img.save(POS / f"{name}.webp", "WEBP", quality=90)
    img.resize((512, 768), Image.LANCZOS).save(POS / f"{name}-m.webp", "WEBP", quality=88)
def uloz_masku(m, name):
    a = (m * 255).astype(np.uint8)
    Image.fromarray(np.dstack([np.full_like(a, 255)] * 3 + [a]), "RGBA").save(POS / f"{name}.webp", "WEBP", lossless=True)

def pest(n):
    a = np.array(Image.open(SRC / f"{n}.png").convert("RGBA")).astype(np.float32)
    r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    green = (g > 150) & (g > r * 1.6) & (g > b * 1.6) & (al > 20)
    ys, xs = np.where(green)
    yv = np.unique(ys); cx = [xs[ys == y].mean() for y in yv]
    k, q = np.polyfit(yv, cx, 1)
    gm = cv2.dilate(green.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    a[gm, 3] = 0
    near = cv2.dilate(gm.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    lim = np.maximum(a[..., 0], a[..., 2]); a[..., 1] = np.where(near & (a[..., 1] > lim), lim, a[..., 1])
    F = a[..., 3] > 30; fy = np.where(F.any(1))[0]; top, bot = int(fy.min()), int(fy.max())
    a[..., 3] *= np.clip((np.arange(a.shape[0]) - top) / 34.0, 0, 1)[:, None]        # zápěstí plynule do paže
    c, lab, st, _ = cv2.connectedComponentsWithStats((a[..., 3] > 10).astype(np.uint8))
    big = np.argmax(st[1:, 4]) + 1; a[..., 3] = np.where(lab == big, a[..., 3], 0)
    fx = np.where(F.any(0))[0]
    y = (top + bot) / 2
    return Image.fromarray(a.astype(np.uint8)), {"x": float(q + k * y), "y": float(y), "top": top, "bot": bot, "x0": int(fx.min()), "x1": int(fx.max())}

def obsah(name):
    im = Image.open(ZBR / f"{name}.webp").convert("RGBA"); al = np.array(im)[..., 3]
    ys = np.where(al.max(1) > 40)[0]; xs = np.where(al.max(0) > 40)[0]
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
def stred_x(im, fy):
    a = np.array(im)[..., 3]; y = int(fy * (im.size[1] - 1)); x = np.where(a[y] > 80)[0]
    if not len(x): return .5
    c = np.where(np.diff(x) > 1)[0]; seg = list(zip(np.r_[x[0], x[c + 1]], np.r_[x[c], x[-1]]))
    s = max(seg, key=lambda u: u[1] - u[0]); return (s[0] + s[1]) / 2 / im.size[0]
def jilec(im):
    """Úchop zbraně s čepelí/hlavou nahoře: střed rukojeti pod záštitou (nejširší řádek ve spodní třetině) nebo 85 %."""
    a = np.array(im)[..., 3] > 80; H = a.shape[0]; w = a.sum(1)
    lo = int(H * .55); gy = lo + int(np.argmax(w[lo:int(H * .93)]))
    if w[gy] > np.median(w[lo:]) * 2.2: return min(.97, (gy + (H - 1)) / 2 / H + .01)
    return .85
def poloz(name, H, grip, cil, rot, zrcadlo=False):
    im = obsah(name); s = H / im.size[1]; im = im.resize((max(1, int(im.size[0] * s)), int(H)), Image.LANCZOS)
    if zrcadlo:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    gx, gy = grip[0] * im.size[0], grip[1] * im.size[1]
    pad = int(np.hypot(*im.size)) + 4
    big = Image.new("RGBA", (pad * 2, pad * 2)); big.alpha_composite(im, (int(pad - gx), int(pad - gy)))
    big = big.rotate(rot, resample=Image.BICUBIC, center=(pad, pad))
    out = Image.new("RGBA", (1024, 1536)); out.paste(big, (int(round(cil[0] - pad)), int(round(cil[1] - pad))), big)
    return out

info = {}
import math
def otoc_bod(x, y, c, ang):
    """Bod (x, y) otočený kolem c o ang stupňů jako Image.rotate (proti směru hodin)."""
    a = math.radians(ang); dx, dy = x - c[0], y - c[1]
    return c[0] + dx * math.cos(a) + dy * math.sin(a), c[1] - dx * math.sin(a) + dy * math.cos(a)
def uhel_jm(ang): return ("m" if ang < 0 else "p") + str(abs(int(ang)))
PESTI = {}                      # (strana, sx) -> (pěst, rukavice nebo None, kloub zápěstí)
def pest_natoc(strana, sx, ang):
    """1.0-1-6 (Pavel): pěst natočená v zápěstí stejně jako zbraň — vrstvy ruka-pest-<strana>-<sx>-<m34…>, u muže i kovová rukavice."""
    im, im2, kl = PESTI[(strana, sx)]
    uloz(im.rotate(ang, resample=Image.BICUBIC, center=kl), f"ruka-pest-{strana}-{sx}-{uhel_jm(ang)}")
    if im2 is not None:
        uloz(im2.rotate(ang, resample=Image.BICUBIC, center=kl), f"rukavice-pest-{strana}-muz-{uhel_jm(ang)}")
def v_pesti(n, H, grip, strana, sx, ang, zrcadlo=False, zaklad=0):
    """Zbraň v natočené pěsti: úchop v jejím středu otočeném kolem zápěstí; zaklad = 180 u zbraní držených hlavou dolů."""
    p = info[f"{strana}-{sx}"]; kl = PESTI[(strana, sx)][2]
    cil = otoc_bod(p["x"], p["y"], kl, ang)
    pest_natoc(strana, sx, ang)
    return poloz(n, H, grip, cil, zaklad + ang, zrcadlo=zrcadlo)
for sx in ("muz", "zena"):
    t = np.array(Image.open(POS / f"postava-telo-{sx}.webp").convert("RGBA")).astype(np.float32)
    rg = (t[..., 0] - t[..., 1]) / np.maximum(t[..., 0], 1); ys, xs = np.mgrid[:1536, :1024]
    zb = [np.array(Image.open(f).convert("RGBA"))[..., 3] > 60 for f in sorted(POS.glob(f"zbroj-{sx}-*.webp")) if "skryva" not in f.name and not f.name.endswith("-m.webp")]
    for strana in ("prava", "leva"):
        im, p = pest(f"ruka-pest-{strana}-{sx}"); uloz(im, f"ruka-pest-{strana}-{sx}"); info[f"{strana}-{sx}"] = p
        im2 = None
        if sx == "muz":
            im2, _ = pest(f"rukavice-pest-{strana}-muz"); uloz(im2, f"rukavice-pest-{strana}-muz")
        PESTI[(strana, sx)] = (im, im2, (p["x"], p["top"] + 10))     # kloub zápěstí = horní okraj pěsti na ose úchopu
        box = (ys > p["top"] + 12) & (ys < 1010) & (np.abs(xs - (p["x0"] + p["x1"]) / 2) < (p["x1"] - p["x0"]) / 2 + 40)
        telo = box & (t[..., 3] > 10) & (rg >= .17)
        telo = cv2.dilate(telo.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        obl = np.zeros_like(box)
        for z in zb: obl |= box & z                      # kovové rukavice zbrojí pod pěstí
        uloz_masku(telo & box, f"ruka-dlan-{strana}-{sx}-skryva-telo"); uloz_masku(obl, f"ruka-dlan-{strana}-{sx}-skryva")
        Image.new("RGBA", (64, 96)).save(POS / f"ruka-dlan-{strana}-{sx}.webp", "WEBP", lossless=True)
    P, Lv = info[f"prava-{sx}"], info[f"leva-{sx}"]
    for z, (druh, cm) in ZBRANE.items():
        n = f"zbran-{z}"; H = cm * PX; im = obsah(n)
        if druh == "visi" and z in DVOU_MECE:            # hrotem na zemi: délka tak, aby hrot právě došel na zem
            gy = jilec(im); H = (ZEM - P["y"]) / gy; out = poloz(n, H, (stred_x(im, gy), gy), (P["x"], P["y"]), 180)
        elif druh == "visi" and z != "prak":            # čepel nahoru šikmo přes tělo, pěst natočená se zbraní
            gy = jilec(im); u = -(NAKLON_DYKA if z in DYKY else NAKLON_HLAVICE if z in HLAVICE else NAKLON_JEDNORUCNI)
            zr = z in SEKERY and ostri_vlevo(im)         # po naklonění doprava by ostří mířilo nahoru → zrcadlit
            gx = 1 - stred_x(im, gy) if zr else stred_x(im, gy)
            out = v_pesti(n, H, (gx, gy), "prava", sx, u, zrcadlo=zr)
            if z not in DYKY:                            # se štítem víc šikmo (hlavice stejně)
                u2 = -(NAKLON_HLAVICE if z in HLAVICE else NAKLON_SE_STITEM)
                uloz(v_pesti(n, H, (gx, gy), "prava", sx, u2, zrcadlo=zr), f"zbran-ruka-{z}-{sx}-stit")
        elif druh == "visi":
            gy = jilec(im); out = poloz(n, H, (stred_x(im, gy), gy), (P["x"], P["y"]), 180)
        elif druh == "stoji":
            gy = max(.3, (H - (ZEM - P["y"])) / H); out = poloz(n, H, (stred_x(im, gy), gy), (P["x"], P["y"]), 0)
        elif druh == "luk":
            out = poloz(n, H, (stred_x(im, .5), .5), (Lv["x"], Lv["y"]), 0)
        else:
            gy = .93; out = v_pesti(n, H, (stred_x(im, gy), gy), "prava", sx, NAKLON_KUSE, zaklad=180)   # 1.0-1-6: šikmo dolů přes nohy (Pavel)
        uloz(out, f"zbran-ruka-{z}-{sx}")
    for z in LEVA:                                     # 1.0-1-6: zbraň v levé pěsti (zrcadlo pravé)
        n = f"zbran-{z}"; H = ZBRANE[z][1] * PX; im = obsah(n)
        gy = jilec(im); out = v_pesti(n, H, (1 - stred_x(im, gy), gy), "leva", sx, NAKLON_LEVA, zrcadlo=True, zaklad=180)
        uloz(out, f"zbran-leva-{z}-{sx}")
    for s_, cm in STITY.items():
        H = cm * PX; out = poloz(s_, H, (.5, .45), (Lv["x"] + STIT_POSUN[0], Lv["y"] + STIT_POSUN[1]), STIT_NAKLON); uloz(out, f"stit-ruka-{s_[5:]}-{sx}")   # štít před tělem na levém předloktí, 1.0-1-6: mírně nakloněný (Pavel)
    out = poloz("svetlo-louc", 55 * PX, (stred_x(obsah("svetlo-louc"), .62), .62), (Lv["x"], Lv["y"]), 0); uloz(out, f"svetlo-louc-ruka-{sx}")
json.dump(info, open(POS / "ruce.json", "w"), indent=1)
j = json.load(open(POS / "skryva.json"))
for sx in ("muz", "zena"):
    for strana in ("prava", "leva"):
        n = f"ruka-dlan-{strana}-{sx}"; j["obleceni"] = sorted(set(j["obleceni"]) | {n}); j["telo"] = sorted(set(j["telo"]) | {n})
json.dump(j, open(POS / "skryva.json", "w"), ensure_ascii=False)
print("hotovo", len(ZBRANE), "zbraní,", len(STITY), "štítů")
