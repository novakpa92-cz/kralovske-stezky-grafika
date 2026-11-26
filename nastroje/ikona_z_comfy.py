"""1.0-1-6: ikona předmětu (160 px, průhledná) z obrázku z ComfyUI na bílém pozadí. python3 nastroje/ikona_z_comfy.py <png> <slug> [prah=N]"""
import sys
from PIL import Image
from predmet_z_comfy import vyrez, R
src, slug = sys.argv[1:3]
prah = next((int(o[5:]) for o in sys.argv[3:] if o.startswith("prah=")), 196)
ik = vyrez(src, prah)
S = int(max(ik.size) * 1.08)
c = Image.new("RGBA", (S, S))
c.alpha_composite(ik, ((S - ik.size[0]) // 2, (S - ik.size[1]) // 2))
c.resize((160, 160), Image.LANCZOS).save(R / "ikony" / f"{slug}.webp", "WEBP", quality=88)
print(slug, "ok")
