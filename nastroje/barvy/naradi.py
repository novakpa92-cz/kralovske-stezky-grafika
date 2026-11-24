"""Vrstvy nářadí na postavě (1024 × 1536): lopata a krumpáč křížem na zádech, lovecký nůž a sekera za opaskem."""
import sys, json
from PIL import Image, ImageFilter
I='/home/claude/kralovske-stezky-grafika/img/predmety/ikony/'
P='/home/claude/kralovske-stezky-grafika/img/postava/'
C=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
def nastroj(n, delka, uhel, stred, zrcadlo=False):
    im=Image.open(I+n+'.webp').convert('RGBA')
    if zrcadlo: im=im.transpose(Image.FLIP_LEFT_RIGHT)
    im=im.crop(im.getchannel('A').getbbox())
    k=delka/max(im.size); im=im.resize((round(im.width*k),round(im.height*k)),Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2,60,2))
    im=im.rotate(uhel,resample=Image.BICUBIC,expand=True)
    return im,(round(stred[0]-im.width/2),round(stred[1]-im.height/2))
import numpy as np
OPASEK=np.asarray(Image.open(P+'opasek-kozeny.webp').getchannel('A'))>90
def vrstva(nazev, polozky, za_opaskem_od=None):
    c=Image.new('RGBA',(1024,1536))
    for n,d,u,s,*z in polozky:
        im,pos=nastroj(n,d,u,s,bool(z and z[0])); c.alpha_composite(im,pos)
    if za_opaskem_od is not None:                    # část pod daným řádkem je schovaná za opaskem (vrstva jde přes opasek)
        a=np.asarray(c).copy(); yy=np.arange(1536)[:,None]
        a[OPASEK&(yy>=za_opaskem_od),3]=0
        c=Image.fromarray(a,'RGBA')
    c.save(P+nazev+'.webp','WEBP',quality=92,method=6); c.resize((512,768),Image.LANCZOS).save(P+nazev+'-m.webp','WEBP',quality=90,method=6)
Z=C.get('zada',{"d":640,"u":33,"x":512,"y":560})
kr=("krumpac",Z["d"],Z["u"],(Z["x"],Z["y"])); lo=("lopata",Z["d"],-Z["u"],(Z["x"],Z["y"]))
vrstva('naradi-zada-krumpac-lopata',[lo,kr])
vrstva('naradi-zada-krumpac',[("krumpac",Z["d"],Z["u"],(Z["x"],Z["y"]))])
vrstva('naradi-zada-lopata',[("lopata",Z["d"],-Z["u"],(Z["x"],Z["y"]))])
N=C.get('nuz',{"d":200,"u":-140,"x":580,"y":690}); S=C.get('sekera',{"d":270,"u":160,"x":672,"y":700})
vrstva('naradi-pas-nuz',[("lovecky-nuz",N["d"],N["u"],(N["x"],N["y"]),N.get("m"))],N.get("za"))
vrstva('naradi-pas-sekera',[("sekera",S["d"],S["u"],(S["x"],S["y"]),S.get("m"))],S.get("za"))
# hlavice sekery zvlášť (kreslí se přes opasek a malou brašnu, topůrko zůstane za nimi)
from scipy import ndimage
a=np.asarray(Image.open(P+'naradi-pas-sekera.webp').convert('RGBA')).copy()
r,g,b=[a[...,i].astype(int) for i in range(3)]
kov=(a[...,3]>40)&((r-b)<22)
lab,n=ndimage.label(ndimage.binary_closing(kov,iterations=3))
if n:
    sz=ndimage.sum(kov,lab,range(1,n+1)); kov=lab==(np.argmax(sz)+1)
kov=ndimage.binary_dilation(kov,iterations=4)
h=a.copy(); h[~kov,3]=0
Image.fromarray(h,'RGBA').save(P+'naradi-pas-sekera-hlava.webp','WEBP',quality=92,method=6)
Image.fromarray(h,'RGBA').resize((512,768),Image.LANCZOS).save(P+'naradi-pas-sekera-hlava-m.webp','WEBP',quality=90,method=6)
print('ok')
