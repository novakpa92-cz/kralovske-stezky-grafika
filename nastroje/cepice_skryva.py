"""Čepice a otevřené helmy jako samostatná vrstva přes vlasy (Pavel 27. 11.: holá hlava + vlasy podle postavy + čepice navrch).
Pro každou helma-<x>[-zena].webp vyrobí helma-<x>[-zena]-skryva.webp: schová vlasy pod čepicí a nad její spodní hranou
(jen v oblasti, kde vůbec nějaké vlasy jsou — zbraně nad hlavou zůstanou) a zapíše ji do skryva.json („obleceni“)."""
import os, json, glob, numpy as np
from PIL import Image
from scipy import ndimage
D = os.path.join(os.path.dirname(__file__), "..", "img", "postava") + "/"
CELE = {"krouzkova-kukla", "bascinet", "velka", "rytirsky-helm", "barbuta", "bascinet-s-psi-kuklou", "batvat", "havirska"}   # vlasy se nekreslí vůbec
BEZ_MASKY = {"venecek"}                                                                                  # sedí na vlasech
UCESY = ("bez", "copanek", "dlouhe", "kratke", "kudrnate", "ofina", "stredni", "tonzura")
def vlasy(sx):
    u = np.zeros((1536, 1024), bool)
    for f in glob.glob(D + f"vlasy-{sx}-*.webp"):
        if f.endswith("-m.webp") or "kuze" in f: continue
        u |= np.array(Image.open(f).convert("RGBA"))[..., 3] > 10
    return ndimage.binary_dilation(u, iterations=4)
def maska(cap, hair):
    a = np.array(cap)[..., 3] > 40
    H, W = a.shape
    xs = np.where(a.any(0))[0]; x0, x1 = xs.min(), xs.max()
    yb = np.zeros(W, int)
    for x in range(x0, x1 + 1):
        ys = np.where(a[:, x])[0]; yb[x] = ys.max() if len(ys) else 0
    yb = ndimage.maximum_filter1d(yb, 9); yb[:x0] = yb[x0]; yb[x1 + 1:] = yb[x1]
    nad = np.arange(H)[:, None] < (yb[None, :] - 6)
    okno = np.zeros(W, bool); okno[max(0, x0 - 40):x1 + 41] = True
    m = (nad & okno[None, :] & hair) | a
    m = ndimage.gaussian_filter(m.astype(np.float32), 1.0)
    out = np.zeros((H, W, 4), np.uint8); out[..., 3] = (m * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)
if __name__ == "__main__":
    sk = json.load(open(D + "skryva.json")); ob = set(sk["obleceni"])
    V = {"muz": vlasy("muz"), "zena": vlasy("zena")}
    for f in sorted(glob.glob(D + "helma-*.webp")):
        n = os.path.basename(f)[:-5]
        if n.endswith(("-m", "-skryva")) or n.endswith(tuple("-" + u for u in UCESY)): continue
        fem = n.endswith("-zena"); slug = n[6:-5] if fem else n[6:]
        if slug in CELE or slug in BEZ_MASKY: continue
        maska(Image.open(f).convert("RGBA"), V["zena" if fem else "muz"]).save(D + n + "-skryva.webp", lossless=True)
        ob.add(n); print(n)
    sk["obleceni"] = sorted(ob); open(D + "skryva.json", "w").write(json.dumps(sk, ensure_ascii=False))
