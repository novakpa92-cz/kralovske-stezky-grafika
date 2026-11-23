import sys, os
from PIL import Image
folder=sys.argv[1]; sub=sys.argv[2]; names=sys.argv[3:]
dst=f'/home/claude/kralovske-stezky-grafika/img/{sub}'.rstrip('/'); os.makedirs(dst,exist_ok=True)
for n in names:
    im=Image.open(f'{folder}/{n}.png')
    alpha = 'A' in im.mode or 'transparency' in im.info
    im=im.convert('RGBA' if alpha else 'RGB')
    if alpha and im.getextrema()[3][0]==255:      # alfa je všude plná → ve skutečnosti neprůhledný
        print('POZOR: neprůhledné pozadí:', n); im=im.convert('RGB')
    if im.mode=='RGBA':
        r,g,b,a=im.split(); im=Image.merge('RGBA',(r,g,b,a.point(lambda v:255 if v>=240 else v)))
    im.save(f'{dst}/{n}.webp','WEBP',quality=78,method=6)
    w=768 if im.size[0]>=im.size[1] else 512
    im.resize((w,round(im.size[1]*w/im.size[0])),Image.LANCZOS).save(f'{dst}/{n}-m.webp','WEBP',quality=76,method=6)
print('published',len(names))
