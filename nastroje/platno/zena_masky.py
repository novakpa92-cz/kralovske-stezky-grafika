import numpy as np, json
from PIL import Image, ImageFilter
from scipy import ndimage
ED="/home/claude/editor/img"
Y,X=np.mgrid[0:1536,0:1024]
body=np.array(Image.open(f"{ED}/postava-telo-zena.webp"))[:,:,3]>30
low=body&(Y>=520); lab,_=ndimage.label(low); torso=lab==lab[700,506]
torso|= body&(Y<520)&(np.abs(X-506.5)<=125)
arm=body&~torso&(Y>=380)&(Y<1000)
dT=ndimage.distance_transform_edt(~torso); dA=ndimage.distance_transform_edt(~arm)
nearArm=dA<dT
def save(m,path):
    im=Image.new("RGBA",(1024,1536),(255,255,255,0)); im.putalpha(Image.fromarray((m*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))); im.save(path,"WEBP",lossless=True)
full=['goticky-plat','platova']
for n in ['krouzkova','kapitanova-brigantina','purkrabiho-brigantina','prosivana','platovy-kyrys','cerny-plat']+full:
    a=np.array(Image.open(f"{ED}/zbroj-zena-{n}.webp"))[:,:,3]>60
    ae=ndimage.binary_erosion(a,iterations=1)
    m=np.zeros_like(a); ruce=np.zeros_like(a)
    for side in (X<506.5, X>=506.5):
        armp=a&side&ndimage.binary_dilation(arm,iterations=6)&(Y>=520)
        if not armp.any(): continue
        endY=1000 if n in ('cerny-plat','platova') else np.where(armp.sum(1)>8)[0].max()-6
        zone=ndimage.binary_dilation(arm&side&(Y<endY),iterations=55)&nearArm&side&(Y<endY)
        m|=zone
        ruce|=ndimage.binary_dilation(arm&side&(Y>=endY),iterations=4)
    if n in full: m|=(Y>=880)
    m&=~ae
    dno=min(720,np.where((a&(np.abs(X-506.5)<90)).any(1))[0].max()-8)
    mo=m.copy()|((Y>=380)&(Y<dno)&~ae&~ndimage.binary_dilation(a&(Y<400),iterations=0 or 1))   # mezera mezi paží a trupem
    mt=m.copy()|((Y>=380)&(Y<dno)&~ae&~arm&torso)
    if n in full:
        mo|=(Y>=380)&~ae                 # celoplátová: šaty ani spodní košile těla nikde nevykukují
        mt|=(Y>=380)&~ae&~ruce
    save(mo,f"{ED}/zbroj-zena-{n}-skryva.webp"); save(mt,f"{ED}/zbroj-zena-{n}-skryva-telo.webp")
