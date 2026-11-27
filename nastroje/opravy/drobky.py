import numpy as np
from PIL import Image
from scipy import ndimage
D="/home/claude/kralovske-stezky-grafika/img/postava/"
def drobky(n, min_px=400):
    L=np.array(Image.open(D+n+".webp").convert("RGBA")); a=L[...,3]>40
    lab,k=ndimage.label(a); vel=ndimage.sum(a,lab,range(1,k+1))
    mal=np.isin(lab,[i+1 for i,v in enumerate(vel) if v<min_px]); L[mal,3]=0
    im=Image.fromarray(L); im.save(D+n+".webp","WEBP",quality=92); im.resize((512,768)).save(D+n+"-m.webp","WEBP",quality=88)
    return int(mal.sum())
