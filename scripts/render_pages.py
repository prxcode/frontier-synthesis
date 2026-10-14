"""Render PDF pages to PNG for visual inspection: python scripts/render_pages.py paper/main.pdf out_dir [dpi]"""
import sys
from pathlib import Path

import pymupdf

pdf, out = sys.argv[1], Path(sys.argv[2])
dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 110
out.mkdir(parents=True, exist_ok=True)
doc = pymupdf.open(pdf)
for i, page in enumerate(doc, 1):
    page.get_pixmap(dpi=dpi).save(out / f"page{i}.png")
print(f"{len(doc)} pages -> {out}")
