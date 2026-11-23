import sys,glob,os
from PIL import Image,ImageDraw,ImageFont
src,pat,box,cols,out=sys.argv[1],sys.argv[2],tuple(map(int,sys.argv[3].split(','))),int(sys.argv[4]),sys.argv[5]
sc=float(sys.argv[6]) if len(sys.argv)>6 else 1
fs=sorted(glob.glob(f"{src}/{pat}.png"))
w,h=int((box[2]-box[0])*sc),int((box[3]-box[1])*sc);rows=(len(fs)+cols-1)//cols
S=Image.new("RGB",(cols*w,rows*(h+22)),(30,24,18));d=ImageDraw.Draw(S)
try:F=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
except:F=None
for i,f in enumerate(fs):
    im=Image.open(f).convert("RGBA").crop(box).resize((w,h));bg=Image.new("RGBA",im.size,(52,40,30,255));bg.alpha_composite(im)
    x,y=(i%cols)*w,(i//cols)*(h+22);S.paste(bg.convert("RGB"),(x,y+22));d.text((x+4,y+3),os.path.basename(f)[:-4],fill=(240,220,160),font=F)
S.save(out);print(out,S.size)
