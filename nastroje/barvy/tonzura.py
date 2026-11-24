"""Tonzura podle Pavlova náčrtu: lysé temeno (kůže), věnec vlasů nad čelem a po stranách k uším — z vlastních vlasů těla."""
import numpy as np
from PIL import Image
from scipy import ndimage
P='/home/claude/kralovske-stezky-grafika/img/postava/'
telo=np.asarray(Image.open(P+'postava-telo-muz.webp').convert('RGBA')).astype(float)
hair=np.load('/tmp/postava-telo-muz-vlasy.npy')
lab,nl=ndimage.label(hair); sz=ndimage.sum(hair,lab,range(1,nl+1)); hair=lab==(np.argmax(sz)+1)
hair=ndimage.binary_dilation(hair,iterations=2)&(telo[...,3]>40)
yy,xx=np.mgrid[0:1536,0:1024].astype(float)
HR=146                                                   # hranice lysiny (temeno nad ní)
t=np.clip((yy-(HR-6))/12,0,1)                            # měkký přechod
hm=hair.astype(float)
# věnec vlasů = vlasy těla pod hranicí
v=telo.copy(); v[...,3]=255*hm*t
# lysina = kůže z čela (řádky pod vlasovou čárou) natažená nahoru, temeno lehce tmavší do stran
ys,xs=np.nonzero(hair); top=ys.min()
celo_y0=176
src_y=np.clip(celo_y0+((yy-top)*0.35),0,1535).astype(int)
k=telo[src_y,xx.astype(int)].copy()
cx=512; sx=np.clip(np.abs(xx-cx)/70,0,1)
shade=1-0.18*sx**2-0.10*np.clip((HR-yy)/40,0,1)
k[...,:3]*=shade[...,None]
# temeno s hladkým obrysem (řádek po řádku vyhlazené okraje vlasů = tvar lebky)
dm=np.zeros_like(hm)
rows=range(top-4,HR+8)
L=[];R=[]
for y in rows:
    xs_=np.nonzero(hair[max(y,top)])[0]
    L.append(xs_.min() if len(xs_) else 512); R.append(xs_.max() if len(xs_) else 512)
L=ndimage.uniform_filter1d(np.array(L,float),9); R=ndimage.uniform_filter1d(np.array(R,float),9)
for i,y in enumerate(rows):
    dm[y,int(L[i])-1:int(R[i])+2]=1
dm=ndimage.gaussian_filter(dm,1.5)
hl=np.exp(-(((xx-500)/45)**2+((yy-(top+14))/14)**2))*0.12       # lesk temene
k[...,:3]=np.clip(k[...,:3]*(1+hl[...,None]),0,255)
k[...,3]=255*dm*(1-t)
for nm,arr in (('vlasy-muz-tonzura',v),('vlasy-muz-tonzura-kuze',k)):
    o=Image.fromarray(np.clip(arr,0,255).astype(np.uint8),'RGBA')
    o.save(P+nm+'.webp','WEBP',quality=92,method=6); o.resize((512,768),Image.LANCZOS).save(P+nm+'-m.webp','WEBP',quality=90,method=6)
print('top',top)
