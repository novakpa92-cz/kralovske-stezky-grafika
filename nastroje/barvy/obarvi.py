import numpy as np, json
from PIL import Image
P="/home/claude/kralovske-stezky-grafika/img/postava/"
PAL=json.load(open("/home/claude/barvy/paleta.json")); ST=json.load(open("/home/claude/barvy/stats.json"))
def hexrgb(h): return np.array([int(h[i:i+2],16) for i in (1,3,5)],float)/255
def obarvi(n,latka=None,nohavice=None):
    im=np.array(Image.open(P+n+".webp").convert("RGBA")).astype(float)/255
    m=np.array(Image.open(f"/home/claude/barvy/{n}-barva.png")).astype(float)/255
    rgb=im[...,:3]; y=rgb@np.array([.299,.587,.114])
    out=rgb.copy()
    for ch,key,dye in ((0,"latka",latka),(1,"nohavice",nohavice)):
        if not dye or key not in ST[n]: continue
        D=hexrgb(PAL[dye][1]); J=ST[n][key]["jas"]; t=(y/J)[...,None]
        tgt=np.where(t<=1,D*t,D+(1-D)*np.clip(t-1,0,1)*0.35)
        w=m[...,ch:ch+1]; out=out*(1-w)+np.clip(tgt,0,1)*w
    return np.concatenate([out,im[...,3:]],-1)
