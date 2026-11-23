"""Vykreslí kombinace postavy s nastavením z Plátna (stejný kód jako editor). render.py cfg.json combos.json outdir"""
import json, sys, base64, threading, http.server, functools, os
from playwright.sync_api import sync_playwright
ED="/home/claude/editor"
cfg=json.load(open(sys.argv[1])); T=cfg.get("T",cfg); combos=json.load(open(sys.argv[2])); out=sys.argv[3]; os.makedirs(out,exist_ok=True)
H=functools.partial(http.server.SimpleHTTPRequestHandler,directory=ED); H.log_message=lambda *a:None
srv=http.server.ThreadingHTTPServer(("127.0.0.1",8798),H); threading.Thread(target=srv.serve_forever,daemon=True).start()
sk=json.load(open(ED+"/skryva.json")) if os.path.exists(ED+"/skryva.json") else {"obleceni":[],"telo":[]}
with sync_playwright() as p:
    b=p.chromium.launch(channel="chrome"); pg=b.new_page(viewport={"width":1400,"height":1000})
    pg.on("pageerror",lambda e:print("JS:",e))
    pg.add_init_script("localStorage.clear();window.SKRYVA="+json.dumps(sk))
    pg.goto("http://127.0.0.1:8798/index.html"); pg.wait_for_timeout(800)
    for name,c in combos.items():
        d=pg.evaluate("""async ([T0,c])=>{T=T0;for(const k in mcache)delete mcache[k];for(const k in hcache)delete hcache[k];sex=c.sex;pick[sex]=Object.assign({odev:'',zbroj:'',opasek:'',plast:'',batoh:'',vlasy:'',helma:'',vousy:'',svetlo:''},c.p);active='__';tool='move';
          for(const n of layers()){const i=img(n);await new Promise(r=>{if(i.complete&&i.naturalWidth)r();else{i.addEventListener('load',r);i.addEventListener('error',r)}})}
          for(const n of layers())if(hasBaseHide(n,0)||hasBaseHide(n,1)){for(const s of ['-skryva','-skryva-telo']){const i=img(n+s);await new Promise(r=>{if(i.complete)r();else{i.addEventListener('load',r);i.addEventListener('error',r)}})}}
          VIEW={k:1,e:0,f:0};await new Promise(r=>setTimeout(r,250));for(const k in hcache)delete hcache[k];draw();return cv.toDataURL('image/png')}""",[T,c])
        open(f"{out}/{name}.png","wb").write(base64.b64decode(d.split(",",1)[1]))
    b.close()
srv.shutdown()
