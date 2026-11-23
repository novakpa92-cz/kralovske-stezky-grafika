# Nástroje (Claude)

- `publish.py` — převod obrázků z ChatGPT/ComfyUI (PNG) do `img/…` jako `.webp` + `-m.webp` (mobil).
- `platno/` — **Plátno postavy** (editor vrstev, `index.html`) a skripty:
  - `bake.py nastaveni.json [--dry]` — zapeče nastavení z Plátna do `img/postava/` (vykreslí stejným kódem jako editor, headless Chrome + playwright).
  - `render.py cfg.json combos.json outdir` + `sheet.py` — náhledy kombinací postavy.
  - `fix.py` (zrcadlení částí), `cleanup.py` (uhlazení okrajů), `zena_masky.py` (automatické masky „schová pod sebou“ pro ženské zbroje).
  - `sync.py` (editor/img → img/postava), `make_zip.py` (zip Plátna pro Pavla s obrázky v `obrazky.js`).
  - Cesty ve skriptech počítají s `/home/claude/editor` (index.html + img/ = kopie img/postava + masky) a `/home/claude/kralovske-stezky-grafika`. Při novém sezení: `mkdir -p /home/claude/editor && cp platno/index.html /home/claude/editor/ && cp -r ../img/postava /home/claude/editor/img && cp ../img/postava/skryva.json /home/claude/editor/`.
