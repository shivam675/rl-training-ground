from __future__ import annotations

import argparse
import difflib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
NS = {"w": W, "w14": W14}


def load_doc(path: Path):
    with zipfile.ZipFile(path) as zf:
        return etree.fromstring(zf.read("word/document.xml"))


def normalized_subtrees(root, xpath):
    items = []
    for node in root.xpath(xpath, namespaces=NS):
        clone = etree.fromstring(etree.tostring(node))
        for sub in clone.iter():
            for attr in list(sub.attrib):
                local = etree.QName(attr).localname
                if local.startswith("rsid") or local in {"paraId", "textId"}:
                    del sub.attrib[attr]
        items.append(etree.tostring(clone, method="c14n"))
    return items


def subtree_summaries(root, xpath):
    out = []
    for i, node in enumerate(root.xpath(xpath, namespaces=NS), 1):
        text = "".join(node.xpath(".//w:t/text()", namespaces=NS))
        prev = node.xpath("preceding::w:p[w:pPr/w:pStyle[@w:val='Heading1' or @w:val='Heading2' or @w:val='Heading3']][1]", namespaces=NS)
        section = "".join(prev[0].xpath(".//w:t/text()", namespaces=NS)) if prev else "Front matter"
        out.append({"index": i, "section": section, "text": text[:500]})
    return out


def para_info(root):
    result = []
    heading = "Front matter"
    for idx, p in enumerate(root.xpath(".//w:body//w:p", namespaces=NS)):
        text = "".join(p.xpath(".//w:t/text() | .//w:delText/text()", namespaces=NS))
        style = p.xpath("string(w:pPr/w:pStyle/@w:val)", namespaces=NS) or "Normal"
        if style in {"Heading1", "Heading2", "Heading3", "FrontMatterHeading"} and text.strip():
            heading = text.strip()
        pid = p.get(f"{{{W14}}}paraId")
        ppr = p.find(f"{{{W}}}pPr")
        ppr_c14n = etree.tostring(ppr, method="c14n") if ppr is not None else b""
        run_props = []
        for r in p.xpath(".//w:r", namespaces=NS):
            rpr = r.find(f"{{{W}}}rPr")
            run_props.append(etree.tostring(rpr, method="c14n") if rpr is not None else b"")
        result.append({
            "index": idx,
            "para_id": pid,
            "style": style,
            "text": text,
            "heading": heading,
            "ppr": ppr_c14n,
            "run_props": run_props,
        })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    aroot, broot = load_doc(args.a), load_doc(args.b)
    aa = para_info(aroot)
    bb = para_info(broot)
    detail_dir = args.out.parent / "semantic_subtree_diffs"
    detail_dir.mkdir(parents=True, exist_ok=True)
    for label, xpath in (("table", ".//w:tbl"), ("drawing", ".//w:drawing")):
        xs, ys = normalized_subtrees(aroot, xpath), normalized_subtrees(broot, xpath)
        for i, (x, y) in enumerate(zip(xs, ys), 1):
            if x != y:
                xd = etree.tostring(etree.fromstring(x), pretty_print=True, encoding="unicode").splitlines()
                yd = etree.tostring(etree.fromstring(y), pretty_print=True, encoding="unicode").splitlines()
                (detail_dir / f"{label}-{i}.diff.txt").write_text("\n".join(difflib.unified_diff(xd, yd, fromfile=f"A/{label}-{i}", tofile=f"B/{label}-{i}", lineterm="")), encoding="utf-8")
    amap = {p["para_id"]: p for p in aa if p["para_id"]}
    bmap = {p["para_id"]: p for p in bb if p["para_id"]}
    common = set(amap) & set(bmap)
    changed_text = []
    changed_style = []
    changed_ppr = []
    changed_run_props = []
    for pid in sorted(common, key=lambda x: amap[x]["index"]):
        a, b = amap[pid], bmap[pid]
        base = {"para_id": pid, "a_index": a["index"], "b_index": b["index"], "section": a["heading"]}
        if a["text"] != b["text"]:
            changed_text.append(base | {"a_style": a["style"], "b_style": b["style"], "a_text": a["text"], "b_text": b["text"]})
        if a["style"] != b["style"]:
            changed_style.append(base | {"a_style": a["style"], "b_style": b["style"], "a_text": a["text"], "b_text": b["text"]})
        if a["ppr"] != b["ppr"]:
            changed_ppr.append(base)
        if a["run_props"] != b["run_props"]:
            changed_run_props.append(base)

    # If Word regenerated paragraph IDs, provide position/sequence statistics too.
    sm = difflib.SequenceMatcher(a=[(p["style"], p["text"]) for p in aa], b=[(p["style"], p["text"]) for p in bb], autojunk=False)
    opcodes = []
    changed_position_sections = Counter()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            for p in aa[i1:i2]:
                if p["text"].strip():
                    changed_position_sections[p["heading"]] += 1
            opcodes.append({
                "tag": tag, "a_range": [i1, i2], "b_range": [j1, j2],
                "a_count": i2-i1, "b_count": j2-j1,
                "a_sections": sorted(set(p["heading"] for p in aa[i1:i2])),
                "b_sections": sorted(set(p["heading"] for p in bb[j1:j2])),
                "a_styles": dict(Counter(p["style"] for p in aa[i1:i2])),
                "b_styles": dict(Counter(p["style"] for p in bb[j1:j2])),
            })

    by_section = Counter(x["section"] for x in changed_text)
    position_style_changes = []
    for a, b in zip(aa, bb):
        if a["style"] != b["style"]:
            position_style_changes.append({
                "index": a["index"], "section": a["heading"],
                "a_style": a["style"], "b_style": b["style"],
                "a_text": a["text"], "b_text": b["text"],
            })
    result = {
        "paragraph_counts": {"a": len(aa), "b": len(bb)},
        "paragraph_ids": {"a": len(amap), "b": len(bmap), "common": len(common), "only_a": len(set(amap)-set(bmap)), "only_b": len(set(bmap)-set(amap))},
        "subtree_equality_ignoring_edit_ids": {
            "tables": normalized_subtrees(aroot, ".//w:tbl") == normalized_subtrees(broot, ".//w:tbl"),
            "drawings": normalized_subtrees(aroot, ".//w:drawing") == normalized_subtrees(broot, ".//w:drawing"),
            "section_properties": normalized_subtrees(aroot, ".//w:sectPr") == normalized_subtrees(broot, ".//w:sectPr"),
        },
        "changed_subtree_indexes_ignoring_edit_ids": {
            "tables": [i for i, (x, y) in enumerate(zip(normalized_subtrees(aroot, ".//w:tbl"), normalized_subtrees(broot, ".//w:tbl")), 1) if x != y],
            "drawings": [i for i, (x, y) in enumerate(zip(normalized_subtrees(aroot, ".//w:drawing"), normalized_subtrees(broot, ".//w:drawing")), 1) if x != y],
        },
        "table_summaries_a": subtree_summaries(aroot, ".//w:tbl"),
        "table_summaries_b": subtree_summaries(broot, ".//w:tbl"),
        "drawing_summaries_a": subtree_summaries(aroot, ".//w:drawing"),
        "drawing_summaries_b": subtree_summaries(broot, ".//w:drawing"),
        "changed_text_count_common_ids": len(changed_text),
        "changed_style_count_common_ids": len(changed_style),
        "changed_paragraph_properties_count_common_ids": len(changed_ppr),
        "changed_run_properties_count_common_ids": len(changed_run_props),
        "changed_text_by_section": dict(by_section),
        "changed_paragraphs_by_section_sequence": dict(changed_position_sections),
        "style_changes": changed_style,
        "position_style_changes": position_style_changes,
        "changed_text": changed_text,
        "sequence_opcodes": opcodes,
        "only_a_para_ids": sorted(set(amap)-set(bmap)),
        "only_b_para_ids": sorted(set(bmap)-set(amap)),
    }
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"changed_text", "sequence_opcodes"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
