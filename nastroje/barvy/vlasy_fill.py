"""Vlasy navazují na hlavu: krátké vlasy těla, které by vykukovaly pod účesem, se přenesou do vrstvy účesu —
obarví se pak spolu s účesem (dřív zůstaly hnědé pod blond či zrzavým účesem)."""
import glob, os
import numpy as np
from PIL import Image
from scipy import ndimage
P='/home/claude/kralovske-stezky-grafika/img/postava/'
Z='/home/claude/zaloha-postava/vlasy/'
for sx in ('muz','zena'):
    telo=np.asarray(Image.open(P+f'postava-telo-{sx}.webp').convert('RGBA')).astype(float)
    hair=np.load(f'/tmp/postava-telo-{sx}-vlasy.npy')
    lab,nl=ndimage.label(hair); sz=ndimage.sum(hair,lab,range(1,nl+1)); hair=lab==(np.argmax(sz)+1)
    hair=ndimage.binary_erosion(hair,iterations=2)
    w=np.clip(ndimage.gaussian_filter(hair.astype(float),0.8),0,1)        # měkký okraj, ne přes kůži
    for f in sorted(glob.glob(Z+f'vlasy-{sx}-*.webp')):
        n=os.path.basename(f)[:-5]
        if n.endswith('-m'): continue
        im=np.asarray(Image.open(f).convert('RGBA')).astype(float)
        a=im[...,3:]/255
        ww=w[...,None]*(1-a)                                    # jen kde účes nekryje
        rgb=im[...,:3]*a+telo[...,:3]*ww
        al=a+ww
        out=np.dstack([rgb/np.maximum(al,1e-6),np.clip(al,0,1)*255])
        o=Image.fromarray(np.clip(out,0,255).astype(np.uint8),'RGBA')
        o.save(P+n+'.webp','WEBP',quality=92,method=6); o.resize((512,768),Image.LANCZOS).save(P+n+'-m.webp','WEBP',quality=90,method=6)
        print(n)
