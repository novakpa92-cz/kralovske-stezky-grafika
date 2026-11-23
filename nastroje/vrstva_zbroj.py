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
SB = np.full(W, -1)                                    # spodek rukávu zbroje ve sloupci (první souvislý kus shora)
for x in range(W):
    c = np.where(A[:, x])[0]
    if len(c) and c[0] < top + 280 and abs(x - 512) > 140:
        c = c[:np.argmax(np.r_[np.diff(c) > 3, True]) + 1]; SB[x] = c.max()
for _sd in (np.arange(W) < 512, np.arange(W) >= 512):   # rukáv srostlý s bokem: sloupce u trupu nesmí brát spodek trupu
    _o = _sd & (np.abs(np.arange(W) - 512) > 150) & (SB >= 0)
    if _o.any():
        _cap = int(np.percentile(SB[_o], 90)) + 12
        SB[_sd & (SB > _cap)] = _cap
_v = np.where(SB >= 0)[0]
if len(_v):                                          # vně rukávu (oděv trčí do strany): spodní hrana rukávu protažená ve stejném sklonu
    for _e, _d in ((_v.min(), -1), (_v.max(), 1)):
        _q = np.arange(_e, _e - 15 * _d, -_d); _sl = np.polyfit(_q, SB[_q], 1)[0] if len(_q) > 3 else 0
        _xx = np.arange(0, _e) if _d < 0 else np.arange(_e + 1, W)
        SB[_xx] = np.clip(SB[_e] + _sl * (_xx - _e), SB[_e], SB[_e] + 40).astype(int)   # vně rukávu nikdy výš než krajní sloupec
_bsep = next((y for y in range(top + 100, 800) if len([u for u in useky(body[y]) if u[1] < 512 - 60]) >= 1 and len(useky(body[y])) >= 3), top + 190)
PY = int(min(SB.max(), _bsep + 25))                   # výška podpaží (u dlouhých rukávů ne spodek rukávu)
# paže: plné rukávy → na paži nic zespodu (až k zápěstí); krátké → jen do konce rukávu
for y in range(top, min(H, 960)):
    for a, b in useky(body[y]):
        if a <= 512 <= b or abs((a + b) / 2 - 512) < 160: continue
        c = np.where(A[:, (a + b) // 2])[0]
        if len(c): c = c[:np.argmax(np.r_[np.diff(c) > 3, True]) + 1]   # jen první souvislý kus shora (rukáv), ne bok zbroje níž
        konec = c.max() if len(c) else 0
        if ruk == "plne":
            if y <= konec + 3: RT[y, max(0, a - 40):b + 41] = True
        elif ruk == "kratke":
            sl = slice(max(0, a - 40), b + 41)
            R[y, sl] = y <= SB[sl] + 2                    # po spodní hranu rukávu zbroje v každém sloupci (šikmý konec rukávu)
            RT[y, sl] |= y <= SB[sl] + 2
        else:
            R[y, max(0, a - 18):b + 19] = False          # vesta: rukávy oděvu zůstanou
if ruk == "kratke":                                    # pod krátkým rukávem: paže (i s rukávem oděvu) zůstane, i kde v siluetě splývá s bokem
    for y in range(top, min(H, 1000)):
        sg = [u for u in useky(A[y]) if abs((u[0] + u[1]) / 2 - 512) < 200]
        if not sg: continue
        l_, r_ = min(u[0] for u in sg), max(u[1] for u in sg)
        pod = y > SB + 2
        L = np.arange(W) < l_ - 10; P = np.arange(W) > r_ + 10
        R[y, (L | P) & pod] = False; RT[y, (L | P) & pod] = False
_OV = (np.array(Image.open(D / f"odev-{OD}.webp").convert("RGBA"))[..., 3] > 100) if OD else None
if ruk == "kratke":                                    # podpaží: pás u boku zbroje (mimo samotnou paži) schovat – žádné světlé cípy pod rukou
    for y in range(top, int(hem.max())):
        sg = [u for u in useky(A[y]) if abs((u[0] + u[1]) / 2 - 512) < 200]
        if not sg: continue
        l_, r_ = min(u[0] for u in sg), max(u[1] for u in sg)
        _pod = y > SB + 2                                 # pod rukávem zbroje je paže s rukávem oděvu vidět (i kde v siluetě splývá s bokem)
        _pod[l_:r_ + 1] = False
        if y >= 700:                                    # níž jen samotná paže (vedle peplu zbroje nic)
            _pa = np.zeros(W, bool)
            for a_, b_ in useky(body[y]):
                if not (a_ <= 512 <= b_): _pa[max(0, a_ - 40):b_ + 41] = True
            _pod &= _pa
        R[y, _pod] = False; RT[y, _pod] = False
        pas = np.zeros(W, bool)
        if y > PY + 40: pas[max(0, l_ - 26):l_] = True; pas[r_ + 1:r_ + 27] = True   # v podpaží (mezi rukávem a trupem zbroje) látka zůstává
        if PY - 20 < y < 700:                    # pod podpažím: všechno mezi paží (+ tloušťka rukávu) a bokem zbroje pryč
            _ar = [u for u in useky(body[y]) if not (u[0] <= 512 <= u[1])]
            _al = [u for u in _ar if u[1] < l_]; _ap = [u for u in _ar if u[0] > r_]
            if _al:                                    # rukáv oděvu (souvislý kus přes paži) zůstává celý, zbytek u boku pryč
                _e = max(u[1] for u in _al); _k = _e
                if _OV is not None:
                    while _k + 1 < l_ and _OV[y, _k + 1]: _k += 1
                    if _k + 1 >= l_: _k = _e + 30           # rukáv splývá s živůtkem až k boku → rukáv bereme do 30 px za paží
                else: _k = _e + 19
                pas[min(l_, _k + 1):l_] = True
            if _ap:
                _e = min(u[0] for u in _ap); _k = _e
                if _OV is not None:
                    while _k - 1 > r_ and _OV[y, _k - 1]: _k -= 1
                    if _k - 1 <= r_: _k = _e - 30
                else: _k = _e - 18
                pas[r_ + 1:max(r_ + 1, _k)] = True
        for a_, b_ in useky(body[y]):
            if not (a_ <= 512 <= b_): pas[max(0, a_ - 6):b_ + 7] = False
        _kz = (t[y, :, 3] > .3) & ((t[y, :, 0] - t[y, :, 1]) / np.maximum(t[y, :, 0], 1e-3) >= .19)   # kůže paže → rukáv oděvu tam zůstává
        pas &= ~(cv2.dilate(_kz.astype(np.uint8)[None, :], np.ones((1, 13), np.uint8))[0] > 0)
        R[y] |= pas
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
    R |= okoli & ~A & (np.abs(xs - 512) > 150) & (ys >= top) & (ys <= (SB + 2 if ruk == "kratke" else colbot + 10)[None, :].clip(0)) & ~((ys >= hem[None, :] - 4) & (np.abs(xs - 512) < 260) & (ys > 700))
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
    _st0 = colbot.copy()                                  # pod zbrojí hned pod lemem
    for _x in range(_xl, _xr + 1):
        if colbot[_x] < _hl - 60: _st0[_x] = colbot[_x]
    _ok = ((ys > _st0[None, :]) & (xs >= _xl) & (xs <= _xr)) | ((ys >= 600) & ~A & (xs >= _xl) & (xs <= _xr) & (ys > 700) & (np.abs(xs - 512) < 60))   # pod lemem; rozparek
    # vedle zbroje: sukně se z rohu lemu plynule rozšiřuje k vlastní šířce dole (zúžená, bez schodů)
    _bl = int(colbot[_xl:_xl + 25].max()); _br = int(colbot[_xr - 24:_xr + 1].max())
    _rows = np.where(_o[:, 512 - 300:512 + 300].any(1))[0]; _yb = int(_rows.max()) - 25
    def _okraj(_y, lev):
        _r = np.where(_o[_y])[0]
        return (_r.min() if lev else _r.max()) if len(_r) else None
    _sLb = _okraj(_yb, True); _sRb = _okraj(_yb, False)
    for _y in range(min(_bl, _br), H):
        if _y > _yb: _ok[_y] = True; continue
        for lev, _b, _x0, _xe in ((True, _bl, _xl, _sLb), (False, _br, _xr, _sRb)):
            if _y <= _b or _xe is None: continue
            _t = min(1.0, (_y - _b) / 140); _w = 35 * (1 - (1 - _t) ** 2) + max(0, _y - _b - 140) * 0.04   # jako lněná sukně: rovná, jen mírně širší než zbroj
            _tx = _x0 - _w if lev else _x0 + _w
            if lev: _ok[_y, int(round(_tx)):_xl] = True
            else: _ok[_y, _xr + 1:int(round(_tx)) + 1] = True
    _v = (_ok & _o).astype(np.uint8)
    _op = cv2.morphologyEx(_v, cv2.MORPH_OPEN, np.ones((1, 9), np.uint8)) > 0
    _ok &= _op | ~_o
    _vz = (_ok & _o & _zona & (ys >= 600) & ~A & ~ruce).astype(np.uint8)   # jen kusy spojené se sukní pod lemem (ne útržky pod paží)
    _c, _lab, _st, _ = cv2.connectedComponentsWithStats(_vz)
    for _i in range(1, _c):
        if _st[_i, cv2.CC_STAT_TOP] + _st[_i, cv2.CC_STAT_HEIGHT] < _hm - 20: _ok[_lab == _i] = False
    _ruka = np.zeros((H, W), bool)                      # paže (oddělené od trupu) – na ty se sukňové pravidlo nevztahuje
    for _y in range(600, min(H, 1000)):
        for _a, _b in useky(body[_y]):
            if not (_a <= 512 <= _b): _ruka[_y, max(0, _a - 35):_b + 36] = True
    _z = (np.abs(xs - 512) < 330) & (ys >= 600) & ~A & ~_ruka
    R = np.where(_z, ~_ok, R); RB = np.where(_z & _zona, ~_ok & ~A, RB)
    _vz = (_o & ~R & ~cv2.dilate(A.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).astype(np.uint8)   # drobné útržky oděvu kolem zbroje pryč
    _c, _lab, _st, _ = cv2.connectedComponentsWithStats(_vz)
    for _i in range(1, _c):
        if _st[_i, cv2.CC_STAT_AREA] < 400: R[_lab == _i] = True
if ruk == "kratke":                                    # spodní prádlo: kde je oděv schovaný, ani prádlo nesmí vykukovat; bez oděvu jeho rukávek pod rukávem zbroje pryč
    _rg = (t[..., 0] - t[..., 1]) / np.maximum(t[..., 0], 1e-3)      # prádlo (len) má malý rozdíl R-G, kůže velký
    _ch = ((t[..., 3] > .3) & (_rg < .19) & (ys >= 330)).astype(np.uint8)
    _ch = cv2.morphologyEx(_ch, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _c, _lab, _st, _ = cv2.connectedComponentsWithStats(_ch)
    _ch = np.isin(_lab, [i for i in range(1, _c) if _st[i, cv2.CC_STAT_AREA] > 800])
    RB |= R & _ch & ~Ajadro
    if not OD: RB |= _ch & ~A & (np.abs(xs - 512) > 130) & (ys < 700)
    _mimo = np.zeros((H, W), bool)                       # dál než 12 px od trupu zbroje (tam jsou paže)
    for y in range(top, min(H, 1000)):
        sg = [u for u in useky(A[y]) if abs((u[0] + u[1]) / 2 - 512) < 200]
        if sg: _mimo[y, :max(0, min(u[0] for u in sg) - 12)] = True; _mimo[y, max(u[1] for u in sg) + 13:] = True
        else: _mimo[y] = True
    RB &= ~((t[..., 3] > .3) & (_rg >= .19) & ~A & (ys > SB[None, :] + 2) & (ys < 1000))   # kůže paží pod rukávem zůstává celá
    _st = (t[..., 3] > .3) & (_rg < .19) & (ys > 400) & (ys < 620) & (np.abs(xs - 512) > 110)   # rukávek košilky: jeho spodek ve sloupci
    _chB = np.where(_st.any(0), H - 1 - np.argmax(_st[::-1], axis=0), -1)
    RB |= (ys <= _chB[None, :] + 4) & (ys > 400) & (_rg < .27) & (t[..., 3] > .05) & ~A & (np.abs(xs - 512) > 110)
    _lo = ((t[..., 3] > .05) & (_rg < .215)).astype(np.uint8)          # košilka i ve stínu: souvislá s hlavní plochou košilky
    _c, _lab, _st2, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(_lo, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)))
    _big = np.isin(_lab, [i for i in range(1, _c) if _st2[i, cv2.CC_STAT_AREA] > 3000])
    _big = (cv2.dilate(_big.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & (_rg < .27) & (t[..., 3] > .05)
    RB |= _big & ~A & (ys > 400) & (ys < 640) & (np.abs(xs - 512) > 100)
    _vb = ((t[..., 3] > .3) & ~RB & ~cv2.dilate(A.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (ys < 760)).astype(np.uint8)
    _c, _lab, _st, _ = cv2.connectedComponentsWithStats(_vb)   # útržky prádla kolem podpaží (ne paže) pryč
    for _i in range(1, _c):
        if _st[_i, cv2.CC_STAT_AREA] < 1500: RB[_lab == _i] = True
for y in range(650, int(hem.max()) + 1):              # rozparek zbroje: uvnitř je vidět spodní vrstva (oděv i prádlo)
    sg = [u for u in useky(A[y]) if abs((u[0] + u[1]) / 2 - 512) < 200]
    if len(sg) < 2: continue
    sg.sort(); a_ = sg[0][1] + 1; b_ = sg[-1][0]
    if b_ - a_ < 120: R[y, a_:b_] = False; RB[y, a_:b_] = False
if ruk == "kratke" and OD:                             # rukáv oděvu pod krátkým rukávem zbroje: vychází z rohu rukávu a plynule se rozšíří (žádná vodorovná polička do strany)
    _ov = np.array(Image.open(D / f"odev-{OD}.webp").convert("RGBA"))[..., 3] > 100
    for _sd in (-1, 1):
        _cols = np.where(A[top:top + 400].any(0))[0]
        _xo = _cols.min() if _sd < 0 else _cols.max()          # vnější hrana rukávu zbroje
        _y0 = int(np.where(A[:, _xo])[0].max())
        _rows = np.arange(_y0, _y0 + 220)
        _edge = []
        for _y in _rows:
            _b = np.where(body[_y])[0]
            if len(_b): _edge.append((_y, _b.min() if _sd < 0 else _b.max()))
        if len(_edge) < 20: continue
        _f = np.polyfit([a for a, b in _edge], [b for a, b in _edge], 1)
        _off = _xo - np.polyval(_f, _y0)                     # o kolik je roh rukávu vně paže
        for _y in range(_y0 + 1, _y0 + 220):
            _t = min(1.0, (_y - _y0) / 120.0); _w = 22 * (1 - (1 - _t) ** 2)
            _lim = np.polyval(_f, _y) + _off + (-_w if _sd < 0 else _w)
            if _sd < 0: R[_y, :max(0, int(_lim))] |= _ov[_y, :max(0, int(_lim))]
            else: R[_y, int(_lim) + 1:] |= _ov[_y, int(_lim) + 1:]
if ruk == "kratke" and OD:                             # kde zůstává vidět kůže paže, musí na ní zůstat i látka oděvu; kde je vidět jen prádlo, pryč obojí
    _o2 = np.array(Image.open(D / f"odev-{OD}.webp").convert("RGBA"))[..., 3] > 100
    _rg2 = (t[..., 0] - t[..., 1]) / np.maximum(t[..., 0], 1e-3)
    _vis = (t[..., 3] > .3) & ~RB & ~A & (ys > top + 60) & (ys < 760) & (np.abs(xs - 512) > 60)
    _sk = (_vis & (_rg2 >= .19)).astype(np.uint8)          # kůže jen té paže, která pokračuje dolů (předloktí), ne útržky u boku
    _c, _lab, _st3, _ = cv2.connectedComponentsWithStats(_sk)
    _arm = np.isin(_lab, [i for i in range(1, _c) if _st3[i, cv2.CC_STAT_TOP] + _st3[i, cv2.CC_STAT_HEIGHT] > SB.max() + 120])
    _bok = np.zeros((H, W), bool)                         # kůže těsně u boku zbroje (podpaží, bok) – to není paže, pryč
    for y in range(top, 760):
        sg = [u for u in useky(A[y]) if abs((u[0] + u[1]) / 2 - 512) < 200]
        if sg: _bok[y, max(0, min(u[0] for u in sg) - 24):min(u[0] for u in sg)] = True; _bok[y, max(u[1] for u in sg) + 1:max(u[1] for u in sg) + 25] = True
    _arm &= ~(_bok & ~_o2)
    R &= ~(_arm & _o2)
    RB |= _vis & ~_arm & R
    RB |= _vis & _bok & ~_o2
    RB |= _vis & _bok & R
if OD:                                                 # útržky oděvu u boku zbroje pod paží (ne rukáv, ne podpažní okno) pryč
    _o3 = np.array(Image.open(D / f"odev-{OD}.webp").convert("RGBA"))[..., 3] > 100
    _v3 = (_o3 & ~R & ~A & (ys > PY + 4) & (ys < 760)).astype(np.uint8)
    _c, _lab, _st4, _ = cv2.connectedComponentsWithStats(_v3)
    for _i in range(1, _c):
        if _st4[_i, cv2.CC_STAT_AREA] < 2500 and 60 < abs(_st4[_i, cv2.CC_STAT_LEFT] + _st4[_i, cv2.CC_STAT_WIDTH] / 2 - 512) < 190:
            R[_lab == _i] = True; RB[(_lab == _i) & ~A] = True
if ruk == "kratke" and SB.max() >= PY + 40:             # dlouhý rukáv zbroje: oděv kolem rukávu (vně i mezi rukávem a trupem) až po konec rukávu pryč
    for y in range(top, int(SB.max()) + 3):
        bs = [u for u in useky(body[y]) if u[0] <= 512 <= u[1]]
        ax = np.where(A[y])[0]
        if y >= PY - 10 and bs:
            tl, tr = bs[0]
            m_ = ((xs[0] < tl) | (xs[0] > tr)) & ~A[y] & (y <= np.where(SB >= 0, SB, int(SB.max())) + 2)
        elif len(ax):
            m_ = ((xs[0] < ax.min()) | (xs[0] > ax.max())) & (y <= SB + 2)
        else: continue
        R[y] |= m_
        RB[y] |= m_ & ~(((t[y, :, 0] - t[y, :, 1]) / np.maximum(t[y, :, 0], 1e-3)) >= .19)   # prádlo taky, kůže ne
if ruk == "kratke":                                    # oděv vyčuhující nad rameny zbroje (límec, kožešina) u krajů pryč, průkrčník zůstává
    for y in range(max(0, top - 50), top + 70):
        ax = np.where(A[y])[0]
        if not len(ax): continue
        m_ = ((xs[0] < ax.min()) | (xs[0] > ax.max())) & (np.abs(xs[0] - 512) > 70)
        R[y] |= m_
_pp = Path("nastroje/podpazi") / f"{n}.png"            # okno v podpaží zbroje (mezi rukávem a trupem): je vidět látka pod ní
if _pp.exists():
    _P = np.array(Image.open(_pp).convert("L")) > 128
    _P = cv2.dilate(_P.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    R &= ~_P; RB &= ~_P
if ruk == "kratke" and SB.max() < PY + 40:              # každá díra v obrysu zbroje mezi rukávem a trupem (podpaží) ukazuje látku pod ní (jen krátké rukávy)
    _hl2 = cv2.morphologyEx(A.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0
    _d2 = _hl2 & ~A & (np.abs(xs - 512) > 70) & (np.abs(xs - 512) < 190) & (ys > top + 30) & (ys < PY + 12)
    for _y in range(top + 30, PY + 30):        # i řádky, kde je mezi rukávem a trupem otevřeno dolů: až ke spodku rukávu
        _sl = [u for u in useky(A[_y]) if abs((u[0] + u[1]) / 2 - 512) < 200]
        if not _sl: continue
        _l, _r = min(u[0] for u in _sl), max(u[1] for u in _sl)
        _lt = [u for u in useky(A[_y]) if u[1] < _l]; _rt = [u for u in useky(A[_y]) if u[0] > _r]
        if _lt: _d2[_y, max(u[1] for u in _lt) + 1:_l] = True
        if _rt: _d2[_y, _r + 1:min(u[0] for u in _rt)] = True
    _sm = PY; _sg0 = [u for u in useky(A[_sm + 5]) if abs((u[0] + u[1]) / 2 - 512) < 200]
    if _sg0:                                          # pod rukávem u podpaží: látka spojitě až k trupu, spodní hrana obloukem (u trupu výš, k paži níž)
        _l0, _r0 = min(u[0] for u in _sg0), max(u[1] for u in _sg0)
        _dist = np.where(xs < 512, _l0 - xs, xs - _r0).astype(np.float32)
        _bot = _sm - 15 + np.clip(_dist, 0, None) * 1.3
        _d2 |= ~A & (np.abs(xs - 512) > 70) & (np.abs(xs - 512) < 200) & (ys > _sm - 40) & (ys < np.minimum(_bot, _sm + 40))
    R &= ~_d2; RB &= ~_d2
jm = f"{n}__{OD}" if OD else n
uloz(R, D / f"{jm}-skryva.webp"); uloz(RB, D / f"{jm}-skryva-telo.webp")
j = json.load(open(D / "skryva.json"))
j["obleceni"] = sorted(set(j["obleceni"]) | {jm}); j["telo"] = sorted(set(j["telo"]) | {jm})
json.dump(j, open(D / "skryva.json", "w"), ensure_ascii=False)
print(n, ruk, "top", top, "lem", int(hem[512]), int(R.sum()))
