import numpy as np
from PIL import Image
AX=1016
def mirror_patch(im, x0,y0,x1,y1, feather=24, ax=AX):
    """Oblast vlevo (x0..x1) nahradí zrcadlem pravé strany (osa těla x=508), s měkkým přechodem na okraji x1 a y0/y1."""
    a=np.array(im.convert("RGBA")).astype(np.float32); m=a[:, ::-1].copy()
    # m[:,x] = a[:,1023-x]; chceme a[:,AX-x] -> posun o (AX-1023)
    sh=ax-1023; m=np.roll(m, -sh, axis=1) if sh else m
    w=np.zeros(a.shape[:2],np.float32); w[y0:y1,x0:x1]=1
    for i in range(feather):
        t=(i+1)/(feather+1)
        if x1+i<1024: w[y0:y1,x1+i]=np.maximum(w[y0:y1,x1+i],1-t)
        if y0-1-i>=0: w[y0-1-i,x0:x1]=np.maximum(w[y0-1-i,x0:x1],1-t)
        if y1+i<1536: w[y1+i,x0:x1]=np.maximum(w[y1+i,x0:x1],1-t)
    w=w[...,None]
    out=a*(1-w)+m*w
    return Image.fromarray(out.clip(0,255).astype(np.uint8),"RGBA")
def mirror_right(im, x0,y0,x1,y1, feather=24, ax=AX):
    """Pravou oblast (x0..x1) nahradí zrcadlem levé strany."""
    f=im.transpose(Image.FLIP_LEFT_RIGHT)
    # oblast vpravo po překlopení obrazu leží vlevo: x' = 1023-x
    r=mirror_patch(f, max(0,1023-x1), y0, 1023-x0, y1, feather, 2046-ax)
    return r.transpose(Image.FLIP_LEFT_RIGHT)
