"""Vrstvy nářadí na postavě (1024 × 1536): lopata a krumpáč křížem na zádech, lovecký nůž a sekera za opaskem."""
import sys, json
from PIL import Image, ImageFilter
I='/home/claude/kralovske-stezky-grafika/img/predmety/ikony/'
P='/home/claude/kralovske-stezky-grafika/img/postava/'
C=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
def nastroj(n, delka, uhel, stred):
    im=Image.open(I+n+'.webp').convert('RGBA')
    im=im.crop(im.getchannel('A').getbbox())
    k=delka/max(im.size); im=im.resize((round(im.width*k),round(im.height*k)),Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2,60,2))
    im=im.rotate(uhel,resample=Image.BICUBIC,expand=True)
    return im,(round(stred[0]-im.width/2),round(stred[1]-im.height/2))
def vrstva(nazev, polozky):
    c=Image.new('RGBA',(1024,1536))
    for n,d,u,s in polozky:
        im,pos=nastroj(n,d,u,s); c.alpha_composite(im,pos)
    c.save(P+nazev+'.webp','WEBP',quality=92,method=6); c.resize((512,768),Image.LANCZOS).save(P+nazev+'-m.webp','WEBP',quality=90,method=6)
Z=C.get('zada',{"d":640,"u":33,"x":512,"y":560})
kr=("krumpac",Z["d"],Z["u"],(Z["x"],Z["y"])); lo=("lopata",Z["d"],-Z["u"],(Z["x"],Z["y"]))
vrstva('naradi-zada-krumpac-lopata',[lo,kr])
vrstva('naradi-zada-krumpac',[("krumpac",Z["d"],Z["u"],(Z["x"],Z["y"]))])
vrstva('naradi-zada-lopata',[("lopata",Z["d"],-Z["u"],(Z["x"],Z["y"]))])
N=C.get('nuz',{"d":200,"u":-140,"x":580,"y":690}); S=C.get('sekera',{"d":270,"u":160,"x":672,"y":700})
vrstva('naradi-pas-nuz',[("lovecky-nuz",N["d"],N["u"],(N["x"],N["y"]))])
vrstva('naradi-pas-sekera',[("sekera",S["d"],S["u"],(S["x"],S["y"]))])
print('ok')
