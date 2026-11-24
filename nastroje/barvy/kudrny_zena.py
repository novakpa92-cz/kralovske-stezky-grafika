"""Ženské kudrnaté: dole u tváří užší (vlasy přiléhají k obličeji), nahoře podle hlavy."""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
Z='/home/claude/zaloha-postava/vlasy/'; P='/home/claude/kralovske-stezky-grafika/img/postava/'
k_top,k_bot,y0,y1,ky=[float(v) for v in sys.argv[1:6]]
im=np.asarray(Image.open(Z+'vlasy-zena-kudrnate.webp').convert('RGBA')).astype(float)
cx,cy=512,225
H,W=im.shape[:2]
yy,xx=np.mgrid[0:H,0:W].astype(float)
ys=cy+(yy-cy)/ky                                 # svislé zmenšení kolem čela
t=np.clip((ys-y0)/(y1-y0),0,1); k=k_top+(k_bot-k_top)*t
xs=cx+(xx-cx)/k
out=np.dstack([ndimage.map_coordinates(im[...,c],[ys,xs],order=1,mode='constant',cval=0) for c in range(4)])
# r27-6: tvar podle Pavlova náčrtu — vlasy přiléhají k hlavě a končí u ramen
from PIL import ImageDraw, ImageFilter
POLY=[(452,128),(512,118),(572,128),(612,166),(624,235),(614,315),(602,388),(588,412),(436,412),(422,388),(410,315),(400,235),(412,166)]
m=Image.new('L',(W,H),0); ImageDraw.Draw(m).polygon(POLY,fill=255); m=np.asarray(m.filter(ImageFilter.GaussianBlur(5))).astype(float)/255
out[...,3]*=m
telo=np.asarray(Image.open(P+'postava-telo-zena.webp').convert('RGBA')).astype(float)
hair=np.load('/tmp/postava-telo-zena-vlasy.npy')
lab,nl=ndimage.label(hair); sz=ndimage.sum(hair,lab,range(1,nl+1)); hair=lab==(np.argmax(sz)+1)
hair=ndimage.binary_erosion(hair,iterations=2); w=np.clip(ndimage.gaussian_filter(hair.astype(float),0.8),0,1)
a=out[...,3:]/255; ww=w[...,None]*(1-a)
rgb=out[...,:3]*a+telo[...,:3]*ww; al=a+ww
o=Image.fromarray(np.clip(np.dstack([rgb/np.maximum(al,1e-6),np.clip(al,0,1)*255]),0,255).astype(np.uint8),'RGBA')
o.save(P+'vlasy-zena-kudrnate.webp','WEBP',quality=92,method=6); o.resize((512,768),Image.LANCZOS).save(P+'vlasy-zena-kudrnate-m.webp','WEBP',quality=90,method=6)
