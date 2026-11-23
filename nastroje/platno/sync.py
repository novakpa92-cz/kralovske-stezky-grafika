"""Zkopíruje upravené vrstvy a masky z editor/img do repa grafiky (+ -m verze) a skryva.json."""
import glob, os, filecmp, shutil, json
from PIL import Image
ED="/home/claude/editor/img"; OUT="/home/claude/kralovske-stezky-grafika/img/postava"; ORIG="/home/claude/editor/img-puvodni"
n=0
for f in sorted(glob.glob(ED+"/*.webp")):
    b=os.path.basename(f); o=f"{OUT}/{b}"
    if os.path.exists(o) and filecmp.cmp(f,o,shallow=False): continue
    shutil.copy(f,o); n+=1
    if not b.endswith(("-skryva.webp","-skryva-telo.webp")):
        Image.open(f).convert("RGBA").resize((512,768),Image.LANCZOS).save(o[:-5]+"-m.webp","WEBP",quality=88,method=6)
    print("→",b)
shutil.copy("/home/claude/editor/skryva.json",OUT+"/skryva.json")
print(n,"souborů")
