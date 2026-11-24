"""Plátový kyrys: kroužkové rukávy přes celou šířku paže (řádek po řádku) a neprůhledné (pod kroužky tmavá vycpávka)."""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
P='/home/claude/kralovske-stezky-grafika/img/postava/'
sx=sys.argv[1]; y0,y1=int(sys.argv[2]),int(sys.argv[3])
A=np.asarray(Image.open(f'/home/claude/zaloha-postava/zbroj-{sx}-platovy-kyrys.webp').convert('RGBA')).copy()
names=(['postava-telo-muz','odev-muz-vlneny-kabatec','odev-muz-lnena-kosile'] if sx=='muz'
       else ['postava-telo-zena','odev-zena-lnena-kosilka'])
ruk=np.zeros(A.shape[:2],bool)
for n in names:
    ruk|=np.asarray(Image.open(P+n+'.webp').getchannel('A'))>60
al=A[...,3]>30
out=A.copy()
def runs(row):
    r=[];s=None
    for x,v in enumerate(row):
        if v and s is None: s=x
        if not v and s is not None: r.append((s,x-1)); s=None
    if s is not None: r.append((s,len(row)-1))
    return r
for y in range(y0,y1):
    for strana in (0,1):
        kr=[r for r in runs(al[y]) if (r[1]<430 if strana==0 else r[0]>594)]
        ar=[r for r in runs(ruk[y]) if (r[1]<440 if strana==0 else r[0]>584)]
        if not kr or not ar: continue
        l,r=(kr[0] if strana==0 else kr[-1]); l=min(k[0] for k in kr); r=max(k[1] for k in kr)
        L,R=(min(a[0] for a in ar),max(a[1] for a in ar))
        L,R=L-2,R+2
        if R-L<=r-l: continue
        xs=np.arange(L,R+1); src=np.clip(np.round(l+(xs-L)*(r-l)/max(1,R-L)).astype(int),0,1023)
        out[y,L:R+1]=A[y,src]
# pod kroužky tmavá vycpávka (rukávy neprůhledné)
rows=np.zeros(A.shape[:2],bool); rows[y0:y1]=True
side=np.zeros_like(rows); side[:, :430]=True; side[:,594:]=True
reg=rows&side&(out[...,3]>20)
reg=ndimage.binary_closing(reg,iterations=3)
a=out[...,3:4].astype(float)/255
pad=np.array([40,38,36],float)
rgb=out[...,:3]*a+pad*(1-a)
out[reg,:3]=rgb[reg].astype(np.uint8); out[reg,3]=255
o=Image.fromarray(out,'RGBA')
o.save(P+f'zbroj-{sx}-platovy-kyrys.webp','WEBP',quality=92,method=6); o.resize((512,768),Image.LANCZOS).save(P+f'zbroj-{sx}-platovy-kyrys-m.webp','WEBP',quality=90,method=6)
print('ok')
