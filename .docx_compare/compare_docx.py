from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
VOLATILE_ATTRS = {
    f"{{{W}}}rsidR",
    f"{{{W}}}rsidRPr",
    f"{{{W}}}rsidDel",
    f"{{{W}}}rsidP",
    f"{{{W}}}rsidSect",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_zip(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist() if not name.endswith("/")}


def canonical_xml(data: bytes, strip_volatile: bool = False) -> bytes:
    parser = etree.XMLParser(remove_blank_text=True, recover=False)
    root = etree.fromstring(data, parser)
    if strip_volatile:
        for node in root.iter():
            for attr in list(node.attrib):
                if attr in VOLATILE_ATTRS or etree.QName(attr).localname.startswith("rsid"):
                    del node.attrib[attr]
    return etree.tostring(root, method="c14n", with_comments=True)


def xml_text_lines(data: bytes) -> list[str]:
    root = etree.fromstring(data)
    lines: list[str] = []
    for p in root.xpath(".//w:p", namespaces=NS):
        text = "".join(p.xpath(".//w:t/text() | .//w:delText/text()", namespaces=NS))
        style = p.xpath("string(w:pPr/w:pStyle/@w:val)", namespaces=NS)
        lines.append(f"[{style or 'Normal'}] {text}")
    return lines


def document_stats(files: dict[str, bytes]) -> dict:
    out: dict = {}
    doc = files.get("word/document.xml")
    if doc:
        root = etree.fromstring(doc)
        out["paragraphs"] = len(root.xpath(".//w:p", namespaces=NS))
        out["tables"] = len(root.xpath(".//w:tbl", namespaces=NS))
        out["rows"] = len(root.xpath(".//w:tr", namespaces=NS))
        out["cells"] = len(root.xpath(".//w:tc", namespaces=NS))
        out["drawings"] = len(root.xpath(".//w:drawing", namespaces=NS))
        out["insertions"] = len(root.xpath(".//w:ins", namespaces=NS))
        out["deletions"] = len(root.xpath(".//w:del", namespaces=NS))
        out["comments_anchors"] = len(root.xpath(".//w:commentReference", namespaces=NS))
        out["sections"] = len(root.xpath(".//w:sectPr", namespaces=NS))
        out["visible_text"] = "\n".join(xml_text_lines(doc))
    out["media"] = {
        name: {"size": len(data), "sha256": sha256(data)}
        for name, data in files.items()
        if name.startswith("word/media/")
    }
    out["headers"] = sorted(n for n in files if re.fullmatch(r"word/header\d+\.xml", n))
    out["footers"] = sorted(n for n in files if re.fullmatch(r"word/footer\d+\.xml", n))
    out["comments_parts"] = sorted(n for n in files if "comment" in n.lower())
    out["footnotes"] = "word/footnotes.xml" in files
    out["endnotes"] = "word/endnotes.xml" in files
    return out


def unified(a: bytes, b: bytes, name: str) -> str:
    try:
        aa = etree.tostring(etree.fromstring(a), pretty_print=True, encoding="unicode").splitlines()
        bb = etree.tostring(etree.fromstring(b), pretty_print=True, encoding="unicode").splitlines()
    except Exception:
        aa = a.decode("utf-8", "replace").splitlines()
        bb = b.decode("utf-8", "replace").splitlines()
    return "\n".join(difflib.unified_diff(aa, bb, fromfile=f"A/{name}", tofile=f"B/{name}", lineterm=""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    az = read_zip(args.a)
    bz = read_zip(args.b)
    all_names = sorted(set(az) | set(bz))
    only_a = [n for n in all_names if n not in bz]
    only_b = [n for n in all_names if n not in az]
    raw_changed: list[str] = []
    canonical_changed: list[str] = []
    substantive_changed: list[str] = []
    raw_only_xml_changes: list[str] = []
    diff_files: list[str] = []

    for name in all_names:
        if name not in az or name not in bz:
            continue
        if az[name] == bz[name]:
            continue
        raw_changed.append(name)
        if name.endswith((".xml", ".rels")):
            ca = canonical_xml(az[name], strip_volatile=False)
            cb = canonical_xml(bz[name], strip_volatile=False)
            if ca != cb:
                canonical_changed.append(name)
                sa = canonical_xml(az[name], strip_volatile=True)
                sb = canonical_xml(bz[name], strip_volatile=True)
                if sa != sb:
                    substantive_changed.append(name)
                diff_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".diff.txt"
                (args.out / diff_name).write_text(unified(az[name], bz[name], name), encoding="utf-8")
                diff_files.append(diff_name)
            else:
                raw_only_xml_changes.append(name)
        else:
            substantive_changed.append(name)

    text_a = xml_text_lines(az["word/document.xml"])
    text_b = xml_text_lines(bz["word/document.xml"])
    text_diff = "\n".join(difflib.unified_diff(text_a, text_b, fromfile="A/text", tofile="B/text", lineterm=""))
    (args.out / "visible_text.diff.txt").write_text(text_diff, encoding="utf-8")

    report = {
        "a": {"path": str(args.a), "size": args.a.stat().st_size, "sha256": sha256(args.a.read_bytes()), "stats": document_stats(az)},
        "b": {"path": str(args.b), "size": args.b.stat().st_size, "sha256": sha256(args.b.read_bytes()), "stats": document_stats(bz)},
        "entry_counts": {"a": len(az), "b": len(bz)},
        "only_a": only_a,
        "only_b": only_b,
        "raw_changed": raw_changed,
        "canonical_changed": canonical_changed,
        "substantive_changed_ignoring_rsid": substantive_changed,
        "raw_only_xml_changes": raw_only_xml_changes,
        "visible_text_equal": text_a == text_b,
        "visible_text_diff_file": "visible_text.diff.txt",
        "xml_diff_files": diff_files,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
