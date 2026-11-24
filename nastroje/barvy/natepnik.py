"""Nátepníky zbroje z medvědí kůže: každý řádek nátepníku roztáhne na celou šířku předloktí (podle nejširšího rukávu)."""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
P='/home/claude/kralovske-stezky-grafika/img/postava/'
sx=sys.argv[1]                     # muz / zena
src=Image.open(f'/home/claude/zaloha-postava/zbroj-{sx}-medvedi.webp').convert('RGBA')
A=np.asarray(src).copy()
al=A[...,3]>40
lab,k=ndimage.label(al)
sl=ndimage.find_objects(lab)
velke=sorted(range(k),key=lambda i:-(lab[sl[i]]==i+1).sum())
hlavni=velke[0]+1
rukavy=[n for n in (('odev-muz-vlneny-kabatec','odev-muz-lnena-kosile','postava-telo-muz') if sx=='muz' else ('odev-zena-vlnene-saty','odev-zena-lnena-kosilka','postava-telo-zena'))]
ruk=np.zeros_like(al)
for n in rukavy:
    ruk|=np.asarray(Image.open(P+n+'.webp').getchannel('A'))>60
out=A.copy()
for i in velke[1:3]:
    s=sl[i]; m=lab==i+1
    if m.sum()<1500: continue
    ys,xs=np.nonzero(m); cx=xs.mean()
    out[m]=0
    for y in range(ys.min(),ys.max()+1):
        row=np.nonzero(m[y])[0]
        if len(row)<2: continue
        l,r=row.min(),row.max()
        arm=np.nonzero(ruk[y])[0]
        arm=arm[np.abs(arm-cx)<70]
        if len(arm)<4: continue
        # souvislý úsek ruky nejblíž nátepníku
        L,R=arm.min()-2,arm.max()+2
        if R-L<r-l: L,R=l,r
        xs_t=np.arange(L,R+1)
        xs_s=np.clip(np.round(l+(xs_t-L)*(r-l)/max(1,R-L)).astype(int),0,1023)
        out[y,xs_t]=A[y,xs_s]
Image.fromarray(out,'RGBA').save(P+f'zbroj-{sx}-medvedi.webp','WEBP',quality=92,method=6)
Image.fromarray(out,'RGBA').resize((512,768),Image.LANCZOS).save(P+f'zbroj-{sx}-medvedi-m.webp','WEBP',quality=90,method=6)
print('ok',[ (lab==i+1).sum() for i in velke[:4]])
