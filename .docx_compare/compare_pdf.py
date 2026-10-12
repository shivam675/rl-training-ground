from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageChops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=144)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    da, db = pdfium.PdfDocument(args.a), pdfium.PdfDocument(args.b)
    pages = []
    scale = args.dpi / 72
    for i in range(max(len(da), len(db))):
        row = {"page": i + 1, "in_a": i < len(da), "in_b": i < len(db)}
        if i >= len(da) or i >= len(db):
            pages.append(row)
            continue
        pa, pb = da[i], db[i]
        ta = pa.get_textpage().get_text_range()
        tb = pb.get_textpage().get_text_range()
        row["text_equal"] = ta == tb
        row["text_chars_a"] = len(ta)
        row["text_chars_b"] = len(tb)
        if ta != tb:
            diff = "\n".join(difflib.unified_diff(ta.splitlines(), tb.splitlines(), fromfile=f"A/page-{i+1}", tofile=f"B/page-{i+1}", lineterm=""))
            (args.out / f"text-page-{i+1}.diff.txt").write_text(diff, encoding="utf-8")
        ia = pa.render(scale=scale).to_pil().convert("RGB")
        ib = pb.render(scale=scale).to_pil().convert("RGB")
        row["same_dimensions"] = ia.size == ib.size
        if ia.size == ib.size:
            diff_img = ImageChops.difference(ia, ib)
            bbox = diff_img.getbbox()
            row["visual_equal"] = bbox is None
            row["diff_bbox"] = bbox
            if bbox is not None:
                ia.save(args.out / f"a-page-{i+1}.png")
                ib.save(args.out / f"b-page-{i+1}.png")
                # Amplify differences for inspection.
                diff_img.point(lambda x: min(255, x * 6)).save(args.out / f"diff-page-{i+1}.png")
        pages.append(row)
    result = {
        "page_count_a": len(da),
        "page_count_b": len(db),
        "changed_text_pages": [p["page"] for p in pages if p.get("text_equal") is False],
        "changed_visual_pages": [p["page"] for p in pages if p.get("visual_equal") is False],
        "pages": pages,
    }
    (args.out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "pages"}, indent=2))


if __name__ == "__main__":
    main()
