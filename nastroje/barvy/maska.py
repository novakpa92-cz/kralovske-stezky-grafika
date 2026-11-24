import numpy as np, json, sys
from PIL import Image
from skimage import color as skc
from sklearn.neighbors import KNeighborsClassifier
from scipy import ndimage as nd
P="/home/claude/kralovske-stezky-grafika/img/postava/"
SH=[(330,1360,430,1440),(580,1360,690,1440)]
# třídy: 0 = jiné (kůže, kov, opasek, boty), 1 = látka, 2 = nohavice. hem = hranice látka/nohavice, boty = odkud jsou jen boty
CFG={
 "odev-muz-lnena-kosile":dict(hem=830,boty=1330,
   L=[(320,380,460,620),(560,380,700,620),(220,420,290,620),(740,420,800,620),(360,700,450,820),(560,700,660,820)],
   N=[(350,880,430,1250),(570,880,650,1250)],
   X=[(370,642,470,666),(560,642,650,666),(484,690,512,770),(190,710,235,790),(790,710,830,790)]+SH),
 "odev-muz-suknice":dict(hem=1040,boty=1360,
   L=[(330,400,460,600),(560,400,690,600),(340,700,500,1000),(580,700,700,1000),(200,500,260,650)],
   N=[(350,1080,430,1300),(570,1080,650,1300)],
   X=[(432,642,468,654),(600,642,636,654),(532,680,548,840)]+SH),
 "odev-muz-vlneny-kabatec":dict(hem=1050,boty=1240,
   L=[(320,400,470,600),(540,400,680,600),(330,720,480,1000),(560,720,700,1000),(220,500,270,700)],
   N=[(340,1080,420,1220),(600,1080,680,1220)],
   X=[(380,644,470,668),(560,644,640,668),(522,690,552,820),(470,305,550,345),(330,1260,420,1430),(590,1260,690,1430)]),
 "odev-zena-lnena-kosilka":dict(hem=None,boty=1360,
   L=[(330,380,470,700),(560,380,690,700),(320,800,480,1200),(560,800,700,1200),(220,450,280,650)],N=[],X=SH),
 "odev-zena-saty":dict(hem=None,boty=1360,
   L=[(330,380,470,520),(560,380,690,520),(330,700,480,1200),(570,700,700,1200),(200,450,260,650)],N=[],
   X=[(400,557,480,578),(560,557,640,578),(502,595,522,750),(425,327,600,342)]+SH),
 "odev-zena-vlnene-saty":dict(hem=None,boty=1360,
   L=[(330,380,470,520),(560,380,690,520),(330,700,480,1200),(570,700,700,1200),(200,450,260,650)],N=[],
   X=[(392,562,470,583),(560,562,630,583),(517,595,538,740),(432,327,588,342)]+SH),
 "odev-zena-kozisek":dict(hem=None,boty=1360,
   L=[(330,800,480,1250),(560,800,700,1250),(400,700,600,760)],N=[],
   X=[(320,400,460,600),(560,400,700,600),(200,550,260,700),(330,330,400,400),(220,700,280,760),(503,580,527,780)]+SH),
 "plast-muz-cestovni":dict(hem=None,boty=None,L=[(150,500,350,1300),(650,500,850,1300),(380,320,460,400),(560,320,640,400)],N=[],X=[(495,332,530,362)]),
 "plast-muz-lehky":dict(hem=None,boty=None,L=[(250,450,400,780),(620,450,780,780),(390,330,470,400),(560,330,640,400)],N=[],X=[(495,335,530,362)]),
 "plast-muz-vlneny-s-kapi":dict(hem=None,boty=None,L=[(150,500,350,1150),(650,500,850,1150),(380,100,620,300)],N=[],X=[(493,340,528,368)]),
}
def feat(rgb):
    lab=skc.rgb2lab(rgb/255.0); lab[...,0]*=0.45; return lab
def run(n,c):
    im=np.array(Image.open(P+n+".webp").convert("RGBA")).astype(float)
    a=im[...,3]/255; F=feat(im[...,:3])
    Xs,ys=[],[]
    for cls,key in ((1,"L"),(2,"N"),(0,"X")):
        for (x0,y0,x1,y1) in c[key]:
            m=a[y0:y1,x0:x1]>0.9; f=F[y0:y1,x0:x1][m]
            if len(f)>4000: f=f[np.random.default_rng(0).choice(len(f),4000,replace=False)]
            Xs.append(f); ys.append(np.full(len(f),cls))
    X=np.concatenate(Xs); y=np.concatenate(ys)
    knn=KNeighborsClassifier(15).fit(X,y)
    msk=a>0.02; lab=np.zeros(a.shape,int); lab[msk]=knn.predict(F[msk])
    yy=np.arange(a.shape[0])[:,None]*np.ones((1,a.shape[1]))
    if c["hem"] is None: lab[lab==2]=1
    else:
        lab[(lab==2)&(yy<c["hem"]-20)]=1; lab[(lab==1)&(yy>c["hem"]+8)]=2
    if c["boty"]: lab[yy>=c["boty"]]=0
    # úklid: malé ostrůvky přeřadit
    for k in (0,1,2):
        m=lab==k
        m2=nd.binary_opening(m,iterations=1)
        lab[m&~m2&msk]=-1
    lab[~msk]=0
    # drobné šedé tečky v látce (šum vlny) — ostrůvky „jiné“ menší než 150 px patří k okolí
    cc,nc=nd.label(lab==0)
    if nc:
        sz=nd.sum(np.ones_like(cc),cc,index=np.arange(1,nc+1))
        small=np.isin(cc,np.nonzero(sz<150)[0]+1)
        lab[small]=-1
    # díry (-1) doplnit nejbližší třídou
    idx=nd.distance_transform_edt(lab<0,return_distances=False,return_indices=True)
    lab=lab[idx[0],idx[1]]
    R=nd.gaussian_filter((lab==1).astype(float),1.0)*a; G=nd.gaussian_filter((lab==2).astype(float),1.0)*a
    out=np.zeros(a.shape+(4,),np.uint8); out[...,0]=(R*255).round(); out[...,1]=(G*255).round(); out[...,3]=255
    Image.fromarray(out[...,:3]).save(f"/home/claude/barvy/{n}-barva.png")
    # průměrný jas látky pro normalizaci
    L=(im[...,:3]@np.array([0.299,0.587,0.114]))/2.55
    st={}
    for k,m in (("latka",lab==1),("nohavice",lab==2)):
        mm=m&(a>0.9)
        if mm.sum()>500: st[k]=dict(jas=round(float(np.median(L[mm]))/100,3))
    return st
names=sys.argv[1:] or list(CFG)
stats={}
for n in names: stats[n]=run(n,CFG[n]); print(n,stats[n])
json.dump(stats,open("/home/claude/barvy/stats.json","w"),ensure_ascii=False,indent=1)
