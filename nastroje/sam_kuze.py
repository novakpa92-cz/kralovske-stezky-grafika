import numpy as np, sys
from PIL import Image, ImageFilter
from skimage.color import rgb2lab
from scipy import ndimage
R='/home/claude/kralovske-stezky-grafika/img/postava/'
def kuze_px(rgb):
    lab=rgb2lab(rgb/255.)
    L,a,b=lab[...,0],lab[...,1],lab[...,2]
    return (L>30)&(L<88)&(a>6)&(a<30)&(b>12)&(b<38)&(b>a*0.75)
def vycisti(n, sam):
    Lr=np.array(Image.open(R+n+'.webp').convert('RGBA'))
    s=np.array(Image.open(sam).convert('L').resize((1024,1536)))>100
    s=ndimage.binary_dilation(s,iterations=3)
    k=kuze_px(Lr[...,:3].astype(float))
    cut=s&k&(Lr[...,3]>0)
    cut=ndimage.binary_opening(cut,iterations=1); cut=ndimage.binary_dilation(cut,iterations=1)
    lab,nn=ndimage.label(cut); vel=ndimage.sum(cut,lab,range(1,nn+1)); cut=np.isin(lab,[i+1 for i,v in enumerate(vel) if v>150])
    out=Lr.copy(); a=out[...,3].astype(float)
    soft=ndimage.gaussian_filter(cut.astype(float),1.0)
    out[...,3]=(a*(1-soft)).clip(0,255).astype('uint8')
    return Lr,out,cut
