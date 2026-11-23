"""Sestaví platno-postavy.zip: index.html + obrazky.js (všechny vrstvy a masky jako data URL) + NAVOD.txt."""
import base64, json, os, glob, zipfile
ED="/home/claude/editor"; Z="/tmp/zipdir/platno-postavy"
os.makedirs(Z,exist_ok=True)
data={os.path.basename(f)[:-5]:"data:image/webp;base64,"+base64.b64encode(open(f,"rb").read()).decode() for f in sorted(glob.glob(ED+"/img/*.webp"))}
sk=json.load(open(ED+"/skryva.json")) if os.path.exists(ED+"/skryva.json") else {"obleceni":[],"telo":[]}
import glob as _g
orig="/home/claude/editor/img-puvodni"; import filecmp
dily=sorted(os.path.basename(f)[:-5] for f in _g.glob(ED+"/img/*.webp") if os.path.exists(orig+"/"+os.path.basename(f)) and not filecmp.cmp(f,orig+"/"+os.path.basename(f),shallow=False))
zap={"id":"sada-"+str(len(dily))+"-"+str(int(os.path.getmtime(ED+"/skryva.json"))),"dily":dily}
open(Z+"/obrazky.js","w").write("window.OBRAZKY="+json.dumps(data)+";\nwindow.SKRYVA="+json.dumps(sk)+";\nwindow.ZAPECENO="+json.dumps(zap)+";\n")
print("zapečené díly:",len(dily))
import shutil; shutil.copy(ED+"/index.html",Z+"/index.html")
out="/home/claude/platno-postavy.zip"
with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
    for f in ("index.html","obrazky.js","NAVOD.txt"): z.write(f"{Z}/{f}",f"platno-postavy/{f}")
print(out, os.path.getsize(out)//1024,"kB", len(data),"obrázků")
