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
R |= (ys <= hem[None, :] - 4) & (ys >= top - 30)
if SUKNE:                                            # sukně oděvu se vedle zbroje smí ukázat od řádku SUKNE dolů (mimo trup zbroje)
    mimo = np.ones(W, bool); mimo[seg[0]:seg[1] + 1] = False          # vedle zbroje (paže se níž vyloučí)
    ruce = np.zeros((H, W), bool)                    # paže a ruce (i s širokými manžetami oděvu) se neuvolňují
    for y in range(top, min(H, 980)):
        for a_, b_ in useky(body[y]):
            if not (a_ <= 512 <= b_ or abs((a_ + b_) / 2 - 512) < 160): ruce[y, max(0, a_ - 70):b_ + 71] = True
    R &= ~((ys >= SUKNE) & mimo[None, :] & ~ruce)
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
RB = (R & ~paze) | (RT & ~A)              # paže v rukávu zbroje: pryč celá (i kůže), ruka pod rukávem zůstane
if ruk != "zadne":                                   # těsně kolem rukávů zbroje nic zespodu
    okoli = cv2.dilate(A.astype(np.uint8), np.ones((51, 51), np.uint8)) > 0
    R |= okoli & ~A & (np.abs(xs - 512) > 150) & (ys >= top) & (ys <= colbot[None, :].clip(0) + 10) & ~((ys >= hem[None, :] - 4) & (np.abs(xs - 512) < 260) & (ys > 700))
def cist(m):                                         # bez jednořádkových proužků a drobků
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((4, 1), np.uint8))
    c, lab, st, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, c):
        if st[i, cv2.CC_STAT_AREA] < 300 and st[i, cv2.CC_STAT_HEIGHT] < 4: m[lab == i] = 0
    return m > 0
R = cist(R) & ~A; RB = cist(RB) & ~A
jm = f"{n}__{OD}" if OD else n
uloz(R, D / f"{jm}-skryva.webp"); uloz(RB, D / f"{jm}-skryva-telo.webp")
j = json.load(open(D / "skryva.json"))
j["obleceni"] = sorted(set(j["obleceni"]) | {jm}); j["telo"] = sorted(set(j["telo"]) | {jm})
json.dump(j, open(D / "skryva.json", "w"), ensure_ascii=False)
print(n, ruk, "top", top, "lem", int(hem[512]), int(R.sum()))
