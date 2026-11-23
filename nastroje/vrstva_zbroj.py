"""Vrstva 3 (zbroj) pro JEDNU zbroj: spodní vrstvy (oděv i spodní prádlo) se schovají všude, kde by
vyčuhovaly kolem zbroje nad jejím lemem — na bocích, ramenou, u krku mimo průkrčník a na pažích,
které zbroj kryje rukávy. Pod lemem (sukně, nohy) a kůže (ruce, obličej) zůstávají.
python3 nastroje/vrstva_zbroj.py <zbroj> plne|kratke|zadne   (rukávy zbroje: po zápěstí / po loket / žádné)"""
import sys, json
import numpy as np, cv2
from PIL import Image
from pathlib import Path
D = Path("img/postava"); n, ruk = sys.argv[1], sys.argv[2]
OD = sys.argv[3] if len(sys.argv) > 3 else ""; SUKNE = int(sys.argv[4]) if len(sys.argv) > 4 else 0
SUKNE_X = None
if len(sys.argv) > 5 and sys.argv[5] == "kabatek" and OD:   # sukně začíná pod lemem kabátku (světlá kožešina) — po sloupcích
    _k = np.array(Image.open(Path("img/postava") / f"odev-{OD}.webp").convert("RGBA")).astype(float)
    _l = _k[..., :3].mean(2); _a = _k[..., 3] > 100
    SUKNE_X = np.full(1024, SUKNE)
    for _x in range(1024):
        _c = np.where(_a[560:760, _x] & (_l[560:760, _x] > 140))[0]
        if len(_c): SUKNE_X[_x] = _c.max() + 560 + 6
    from scipy.ndimage import maximum_filter1d
    SUKNE_X = maximum_filter1d(SUKNE_X, 61)
sx = "zena" if "-zena-" in n else "muz"
A = np.array(Image.open(D / f"{n}.webp").convert("RGBA"))[..., 3] > 60
t = np.array(Image.open(D / f"postava-telo-{sx}.webp").convert("RGBA")).astype(float) / 255
body = t[..., 3] > .02
H, W = A.shape; ys = np.arange(H)[:, None]; xs = np.arange(W)[None, :]
def useky(row):
    x = np.where(row)[0]
    if not len(x): return []
    c = np.where(np.diff(x) > 1)[0]; return list(zip(np.r_[x[0], x[c + 1]], np.r_[x[c], x[-1]]))
top = np.where(A.any(1))[0].min()
# lem trupu: spodek zbroje ve sloupcích kolem středu
hem = np.full(W, -1)
lc = int(np.where(A[:, 512])[0].max())                 # lem uprostřed
yr = lc - 40                                          # řádek nad lemem: šířka trupu zbroje
seg = [(a_, b_) for a_, b_ in useky(A[yr]) if a_ <= 512 <= b_][0]
for x in range(seg[0], seg[1] + 1):
    c = np.where(A[yr:, x])[0]
    hem[x] = yr + (c.max() if len(c) else 0)
tor = np.arange(seg[0], seg[1] + 1)
lo, hi = tor.min(), tor.max()
for x in range(W):                                   # mimo trup: od rohu lemu šikmo dolů (sukně vychází zpod zbroje)
    if hem[x] < 0:
        e = lo if x < lo else hi
        hem[x] = hem[e] + int(abs(x - e) * .5)
R = np.zeros((H, W), bool); RT = np.zeros((H, W), bool)   # RT: paže pro tělo (jen po konec rukávu zbroje)
if ruk != "zadne":
    R |= (ys <= hem[None, :] - 4) & (ys >= top - 30)
else:
    # vesta: rukávy oděvu zůstávají; schová se jen to, co přečuhuje u trupu do stran a nad ramena
    colTop = np.where(A.any(0), np.argmax(A, axis=0), H)
    for x in np.where(A.any(0))[0]:
        R[max(0, colTop[x] - 40):colTop[x], x] = True
    pp = next((y for y in range(top, 900) if len([u for u in useky(body[y]) if not (u[0] <= 512 <= u[1])]) >= 2), top + 120)
    PODP = np.zeros((H, W), bool)                    # podpaží: tady pryč i cípy spodního prádla (ne kůže)
    # podpaží: vnitřní hranu paže (kde už je od těla oddělená) protáhnu přímkou nahoru — mezi vestou a touto čarou je bok
    li, ri = [], []
    for y in range(pp, pp + 120):
        us = useky(body[y]); l = [u for u in us if u[1] < 512 - 60]; r = [u for u in us if u[0] > 512 + 60]
        if l: li.append((y, max(u[1] for u in l)))
        if r: ri.append((y, min(u[0] for u in r)))
    fl = np.polyfit([a for a, b in li], [b for a, b in li], 1) if len(li) > 10 else None
    fr = np.polyfit([a for a, b in ri], [b for a, b in ri], 1) if len(ri) > 10 else None
    for y in range(top + 60, pp + 12):
        sg = [(a_, b_) for a_, b_ in useky(A[y]) if a_ <= 512 <= b_]
        if not sg: continue
        l_, r_ = sg[0]
        if fl is not None:
            e = int(np.polyval(fl, y)) + 2
            if e < l_: R[y, e:l_] = True
            PODP[y, max(0, e - 18):l_] = True
        if fr is not None:
            e = int(np.polyval(fr, y)) - 2
            if e > r_: R[y, r_ + 1:e] = True
            PODP[y, r_ + 1:e + 18] = True
    for y in range(pp + 12, int(hem.max()) + 1):
        sg = [(a_, b_) for a_, b_ in useky(A[y]) if abs((a_ + b_) / 2 - 512) < 200]     # trup (i se dvěma cípy pod rozparkem)
        if not sg: continue
        l_, r_ = min(u[0] for u in sg), max(u[1] for u in sg)
        z = np.zeros(W, bool); z[max(0, l_ - 60):l_] = True; z[r_ + 1:r_ + 61] = True
        z &= y < hem                                     # pod spodním okrajem vesty už nic
        for a_, b_ in useky(body[y]):
            mg = 35 if y < pp + 40 else 8
            if not (a_ <= 512 <= b_): z[max(0, a_ - mg):b_ + mg + 1] = False
        R[y] |= z
if SUKNE:                                            # sukně oděvu se vedle zbroje smí ukázat od řádku SUKNE dolů (mimo trup zbroje)
    mimo = np.ones(W, bool); mimo[seg[0]:seg[1] + 1] = False          # vedle zbroje (paže se níž vyloučí)
    ruce = np.zeros((H, W), bool)                    # paže a ruce (i s širokými manžetami oděvu) se neuvolňují
    for y in range(top, min(H, 980)):
        for a_, b_ in useky(body[y]):
            if not (a_ <= 512 <= b_ or abs((a_ + b_) / 2 - 512) < 160): ruce[y, max(0, a_ - 25):b_ + 26] = True
    if ruk == "zadne": ruce[:] = False                 # vesta: rukávy oděvu se neskrývají, není co chránit
    SK = (ys >= (SUKNE_X[None, :] if SUKNE_X is not None else SUKNE))
    R &= ~(SK & ~ruce)                     # pod řádkem SUKNE je sukně oděvu vidět všude, kde ji zbroj nekryje (paže ne)
# paže: plné rukávy → na paži nic zespodu (až k zápěstí); krátké → jen do konce rukávu
for y in range(top, min(H, 960)):
    for a, b in useky(body[y]):
        if a <= 512 <= b or abs((a + b) / 2 - 512) < 160: continue
        c = np.where(A[:, (a + b) // 2])[0]; konec = c.max() if len(c) else 0
        if ruk == "plne":
            if y <= konec + 3: RT[y, max(0, a - 40):b + 41] = True
        elif ruk == "kratke":
            R[y, max(0, a - 40):b + 41] = y <= konec + 2
            if y <= konec + 2: RT[y, max(0, a - 40):b + 41] = True
        else:
            R[y, max(0, a - 18):b + 19] = False          # vesta: rukávy oděvu zůstanou
# průkrčník: v otvoru zbroje u krku se límec oděvu smí ukázat
for y in range(top - 30, top + 90):
    row = np.where(A[y])[0]
    s = [(a, b) for a, b in useky(A[y]) if b < 512 or a > 512]
    l = max([b for a, b in s if b < 512], default=None); r = min([a for a, b in s if a > 512], default=None)
    if l is not None and r is not None and r - l < 220: R[y, l:r] = False
    elif y < top + 10: R[y, 512 - 70:512 + 70] = False
R &= ~A
R[: 300] = False                                     # hlava nikdy
def uloz(m, p):
    a = (m * 255).astype(np.uint8)
    Image.fromarray(np.dstack([np.full_like(a, 255)] * 3 + [a]), "RGBA").save(p, "WEBP", lossless=True)
mx = t[..., :3].max(2); mn = t[..., :3].min(2); sat = (mx - mn) / np.maximum(mx, 1e-6)
pradlo = (t[..., 3] > .02) & (sat < .40) & (ys >= 330)                 # spodní prádlo (ne kůže)
paze = np.zeros((H, W), bool)
for y in range(top, min(H, 960)):
    for a, b in useky(body[y]):
        if not (a <= 512 <= b or abs((a + b) / 2 - 512) < 160): paze[y, max(0, a - 40):b + 41] = True
colbot = np.where(A.any(0), H - 1 - np.argmax(A[::-1], axis=0), -1)       # spodek zbroje v každém sloupci
if ruk == "plne":                                      # oděv pod rukávem zbroje: pryč až kousek pod konec rukávu (manžety)
    R |= paze & (ys <= colbot[None, :] + 45) & (colbot[None, :] > 0) & (np.abs(xs - 512) > 150)
RT2 = paze & (ys <= colbot[None, :] + 2) & (colbot[None, :] > 0)
if ruk != "zadne": RT = RT2
RB = (R & ~paze) | (RT & ~A)
if ruk == "zadne":
    RB = R.copy()
    _t = np.array(Image.open(D / f"postava-telo-{sx}.webp").convert("RGBA")).astype(float)
    _mx = _t[..., :3].max(2); _s = (_mx - _t[..., :3].min(2)) / np.maximum(_mx, 1)
    pass                     # vesta: R míří jen k trupu, paže nezasahuje              # paže v rukávu zbroje: pryč celá (i kůže), ruka pod rukávem zůstane
if ruk != "zadne":                                   # těsně kolem rukávů zbroje nic zespodu
    okoli = cv2.dilate(A.astype(np.uint8), np.ones((51, 51), np.uint8)) > 0
    R |= okoli & ~A & (np.abs(xs - 512) > 150) & (ys >= top) & (ys <= colbot[None, :].clip(0) + 10) & ~((ys >= hem[None, :] - 4) & (np.abs(xs - 512) < 260) & (ys > 700))
def cist(m):                                         # bez jednořádkových proužků a drobků
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((4, 1), np.uint8))
    c, lab, st, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, c):
        if st[i, cv2.CC_STAT_AREA] < 300 and st[i, cv2.CC_STAT_HEIGHT] < 4: m[lab == i] = 0
    return m > 0
Ajadro = cv2.erode((np.array(Image.open(D / f"{n}.webp").convert("RGBA"))[..., 3] > 250).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
R = cist(R | (cv2.dilate(R.astype(np.uint8), np.ones((1, 9), np.uint8)) > 0) & ~Ajadro) & ~Ajadro   # i pod měkkým okrajem zbroje
RB = cist(RB | (cv2.dilate(RB.astype(np.uint8), np.ones((1, 9), np.uint8)) > 0) & ~Ajadro) & ~Ajadro
if SUKNE and ruk == "zadne":
    R &= ~(SK & ~ruce); RB &= ~(SK & ~ruce)
if SUKNE_X is not None:                              # kožešina kabátku vedle zbroje (u boku) pryč
    _f = (_a & (_l > 135)).astype(np.uint8)
    _f = cv2.dilate(_f, np.ones((7, 7), np.uint8)) > 0
    _hull = np.zeros((H, W), bool)
    for _y in range(540, 721):
        _sg = [u for u in useky(A[_y]) if abs((u[0] + u[1]) / 2 - 512) < 200]
        if _sg: _hull[_y, max(0, min(u[0] for u in _sg) - 60):min(u[0] for u in _sg)] = True; _hull[_y, max(u[1] for u in _sg) + 1:max(u[1] for u in _sg) + 61] = True
    R |= _f & _hull & ~ruce
if SUKNE and OD:                                     # sukně vedle zbroje nad lemem schovaná – vidět je až pod lemem (vnořená)
    _o = np.array(Image.open(D / f"odev-{OD}.webp").convert("RGBA"))[..., 3] > 100
    _zona = np.abs(xs - 512) < 270
    _cb = colbot[512 - 270:512 + 270]; _hm = int(_cb[_cb > 0].max())
    _hl = int(np.median(colbot[400:624][colbot[400:624] > 0])); _y0 = _hl - 300; _k = 2.5
    _ok = np.zeros((H, W), bool)                      # kde smí být sukně vidět
    _tc = [x for x in range(512 - 270, 512 + 270) if colbot[x] > _hl - 60]   # sloupce trupu zbroje (k lemu)
    _xl, _xr = min(_tc), max(_tc)
    _st0 = colbot.copy().astype(float)                   # pod zbrojí hned pod lemem; vedle ní sukně vylézá šikmo z rohu lemu (kužel), žádná polička
    _bl = int(colbot[_xl:_xl + 25].max()); _br = int(colbot[_xr - 24:_xr + 1].max())
    _st0[:_xl] = _bl + (_xl - np.arange(_xl)) / 0.9
    _st0[_xr + 1:] = _br + (np.arange(_xr + 1, W) - _xr) / 0.9
    for _x in range(_xl, _xr + 1):
        if colbot[_x] < _hl - 60: _st0[_x] = colbot[_x]
    _st0 = _st0.astype(int)
    _ok = (ys > _st0[None, :]) | ((ys >= 600) & ~A & (xs >= _xl) & (xs <= _xr) & (ys > 700) & (np.abs(xs - 512) < 60))   # pod lemem; rozparek
    _v = (_ok & _o).astype(np.uint8)
    _op = cv2.morphologyEx(_v, cv2.MORPH_OPEN, np.ones((1, 9), np.uint8)) > 0
    _ok &= _op | ~_o
    _vz = (_ok & _o & _zona & (ys >= 600) & ~A & ~ruce).astype(np.uint8)   # jen kusy spojené se sukní pod lemem (ne útržky pod paží)
    _c, _lab, _st, _ = cv2.connectedComponentsWithStats(_vz)
    for _i in range(1, _c):
        if _st[_i, cv2.CC_STAT_TOP] + _st[_i, cv2.CC_STAT_HEIGHT] < _hm - 20: _ok[_lab == _i] = False
    _z = (np.abs(xs - 512) < 330) & (ys >= 600) & ~A
    R = np.where(_z, ~_ok, R); RB = np.where(_z & _zona, ~_ok & ~A, RB)
    _vz = (_o & ~R & ~cv2.dilate(A.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).astype(np.uint8)   # drobné útržky oděvu kolem zbroje pryč
    _c, _lab, _st, _ = cv2.connectedComponentsWithStats(_vz)
    for _i in range(1, _c):
        if _st[_i, cv2.CC_STAT_AREA] < 400: R[_lab == _i] = True
jm = f"{n}__{OD}" if OD else n
uloz(R, D / f"{jm}-skryva.webp"); uloz(RB, D / f"{jm}-skryva-telo.webp")
j = json.load(open(D / "skryva.json"))
j["obleceni"] = sorted(set(j["obleceni"]) | {jm}); j["telo"] = sorted(set(j["telo"]) | {jm})
json.dump(j, open(D / "skryva.json", "w"), ensure_ascii=False)
print(n, ruk, "top", top, "lem", int(hem[512]), int(R.sum()))
