import numpy as np
from PIL import Image
from scipy import ndimage
D="/home/claude/kralovske-stezky-grafika/img/postava/"
def vnitrek(n):
    L=np.array(Image.open(D+n+".webp").convert("RGBA")); a=L[...,3]>40
    inn=ndimage.binary_erosion(ndimage.binary_fill_holes(a),iterations=3)&a
    L[inn,3]=255
    Image.fromarray(L).save(D+n+".webp","WEBP",quality=92); Image.fromarray(L).resize((512,768)).save(D+n+"-m.webp","WEBP",quality=88)
