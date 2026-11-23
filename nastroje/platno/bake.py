"""Zapeče nastavení z Plátna postavy do vrstev v repu grafiky.
Použití: python3 bake.py nastaveni.json [--dry]
Vykreslí každý upravený díl stejným kódem jako editor (headless Chrome), uloží
img/postava/<díl>.webp (1024×1536) a <díl>-m.webp (512×768)."""
import json, sys, base64, io, threading, http.server, functools, os
from PIL import Image
from playwright.sync_api import sync_playwright
ED="/home/claude/editor"; OUT="/home/claude/kralovske-stezky-grafika/img/postava"
cfg=json.load(open(sys.argv[1])); dry="--dry" in sys.argv
if "T" in cfg: cfg=cfg["T"]
H=functools.partial(http.server.SimpleHTTPRequestHandler,directory=ED)
H.log_message=lambda *a:None
srv=http.server.ThreadingHTTPServer(("127.0.0.1",8799),H); threading.Thread(target=srv.serve_forever,daemon=True).start()
with sync_playwright() as p:
    b=p.chromium.launch(channel="chrome"); pg=b.new_page()
    pg.on("pageerror",lambda e:print("JS:",e))
    man0=json.load(open(f"{OUT}/skryva.json")) if os.path.exists(f"{OUT}/skryva.json") else {"obleceni":[],"telo":[]}
    pg.add_init_script("localStorage.clear();window.SKRYVA="+json.dumps(man0))
    pg.goto("http://127.0.0.1:8799/index.html"); pg.wait_for_timeout(500); pg.evaluate("draw=()=>{}")
    res=pg.evaluate("""async cfg=>{T=cfg;for(const k in mcache)delete mcache[k];const out={};
      for(const n of Object.keys(cfg)){const i=img(n);await new Promise(r=>{if(i.complete&&i.naturalWidth)r();else{i.addEventListener('load',r);i.addEventListener('error',r)}});
        if(!i.naturalWidth){out[n]=null;continue}bbox(n);
        cx.setTransform(1,0,0,1,0,0);cx.globalAlpha=1;cx.clearRect(0,0,1024,1536);cx.save();xf(n);drawWarped(n,i);cx.restore();
        out[n]=cv.toDataURL('image/png');
        for(const [b,suf] of [[0,'-skryva'],[1,'-skryva-telo']]){const hm=hideMask(n,0,b);if(hm){cx.setTransform(1,0,0,1,0,0);cx.clearRect(0,0,1024,1536);cx.save();xf(n);drawWarped(n,hm,true);cx.restore();out[n+suf]=cv.toDataURL('image/png')}else out[n+suf]=null}}
      return out}""",cfg)
    b.close()
srv.shutdown()
man_p=f"{OUT}/skryva.json"
man={k:set(v) for k,v in man0.items()}
for n,d in res.items():
    if n.endswith("-skryva") or n.endswith("-skryva-telo"):
        telo=n.endswith("-telo"); dil=n[:-12] if telo else n[:-7]; M=man["telo" if telo else "obleceni"]
        if d is None:
            if dil in M and not dry: M.discard(dil); os.path.exists(f"{OUT}/{n}.webp") and os.remove(f"{OUT}/{n}.webp"); print("maska odebrána",dil)
            continue
        im=Image.open(io.BytesIO(base64.b64decode(d.split(",",1)[1]))).convert("RGBA")
        a=im.getchannel("A"); mask=Image.new("RGBA",im.size,(255,255,255,0)); mask.putalpha(a)
        if dry: mask.save(f"/tmp/bake-{n}.png"); print("náhled",n); continue
        mask.save(f"{OUT}/{n}.webp","WEBP",lossless=True); M.add(dil)
        import shutil; shutil.copy(f"{OUT}/{n}.webp",f"{ED}/img/{n}.webp"); print("maska",dil); continue
    if not d: print("chybí obrázek:",n); continue
    im=Image.open(io.BytesIO(base64.b64decode(d.split(",",1)[1]))).convert("RGBA")
    if dry: im.save(f"/tmp/bake-{n}.png"); print("náhled",n); continue
    im.save(f"{OUT}/{n}.webp","WEBP",quality=90,method=6)
    import shutil; shutil.copy(f"{OUT}/{n}.webp",f"{ED}/img/{n}.webp")
    im.resize((512,768),Image.LANCZOS).save(f"{OUT}/{n}-m.webp","WEBP",quality=88,method=6)
    print("zapečeno",n)
if not dry:
    mo={k:sorted(v) for k,v in man.items()}
    json.dump(mo,open(man_p,"w"),ensure_ascii=False); print("skryva.json:",mo)
    json.dump(mo,open(f"{ED}/skryva.json","w"))
    print("Pozor: zapečené díly mají v Plátně nový základ — Pavlovi poslat nový zip (make_zip.py) a u těchto dílů vrátit nastavení na nulu.")
