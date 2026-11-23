"""Uhladí okraje po gumování: odstraní drobné ostrůvky a zubaté okraje alfy. cleanup.py díl [díl…] (přepíše v editor/img + grafika)."""
import sys, numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
OUTS=["/home/claude/editor/img","/home/claude/kralovske-stezky-grafika/img/postava"]
def clean(im, min_island=400):
    a=np.array(im.getchannel("A")).astype(np.float32)
    m=a>24
    lab,n=ndimage.label(m)
    if n:
        sizes=ndimage.sum(m,lab,range(1,n+1))
        keep=np.isin(lab,1+np.where(sizes>=min_island)[0])
        a[~keep]=0
    # zubatý okraj: median na alfě jen v pásu kolem hrany, pak lehké rozmazání hrany
    A=Image.fromarray(a.astype(np.uint8))
    med=np.array(A.filter(ImageFilter.MedianFilter(5))).astype(np.float32)
    edge=ndimage.binary_dilation(m,iterations=3)&~ndimage.binary_erosion(m,iterations=3)
    a2=np.where(edge,np.minimum(a,med),a)
    bl=np.array(Image.fromarray(a2.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))).astype(np.float32)
    a3=np.where(edge,np.minimum(a2,bl),a2)
    out=im.copy(); out.putalpha(Image.fromarray(a3.clip(0,255).astype(np.uint8))); return out
if __name__=="__main__":
    for n in sys.argv[1:]:
        im=Image.open(f"{OUTS[0]}/{n}.webp").convert("RGBA"); c=clean(im)
        for o in OUTS:
            c.save(f"{o}/{n}.webp","WEBP",quality=90,method=6)
        c.resize((512,768),Image.LANCZOS).save(f"{OUTS[1]}/{n}-m.webp","WEBP",quality=88,method=6)
        print("uhlazeno",n)
