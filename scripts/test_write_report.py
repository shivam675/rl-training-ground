"""Self-check for the report Markdown parser (no Word, no COM).

    .venv/Scripts/python.exe scripts/test_write_report.py
    .venv/Scripts/python.exe -m pytest scripts/test_write_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_report import (  # noqa: E402
    CITATION_RE,
    Autosave,
    WordWriter,
    citation_code,
    inline_runs,
    load_sources,
    parse,
)

SAMPLE = """---
title: Demo
subtitle: A Subtitle
affiliation: School | University | Month Year
---

@@sect front

@@front Abstract

First paragraph.

@@pagebreak

@@front Table of Contents

@@toc

## List of Figures

@@lof

## List of Tables

@@lot

@@sect body

# 1  Introduction

Body text with an escaped \\| pipe.

- bullet one
- bullet two

@@figure figure01.png 6.2x1.55 :: A caption.

Table: A table caption.
| Head A | Head B |
|---|---|
| a1 | b1 |
| a2 | b2 |
"""


def test_metadata():
    meta, _ = parse(SAMPLE)
    assert meta["title"] == "Demo"
    assert meta["subtitle"] == "A Subtitle"
    assert meta["affiliation"].split(" | ") == ["School", "University", "Month Year"]


def test_block_order():
    _, blocks = parse(SAMPLE)
    assert [b.kind for b in blocks] == [
        "sect", "front", "para", "pagebreak", "front", "toc",
        "h2", "lof", "h2", "lot",
        "sect", "h1", "para", "bullet", "bullet", "figure", "table",
    ]
    assert blocks[0].text == "front"
    assert blocks[10].text == "body"
    assert blocks[11].text == "1  Introduction"  # double space in the number preserved


def test_figure():
    _, blocks = parse(SAMPLE)
    fig = next(b for b in blocks if b.kind == "figure")
    assert (fig.image, fig.width, fig.height) == ("figure01.png", 6.2, 1.55)
    assert fig.caption == "A caption."


def test_table():
    _, blocks = parse(SAMPLE)
    tbl = next(b for b in blocks if b.kind == "table")
    assert tbl.caption == "A table caption."
    assert tbl.rows == [["Head A", "Head B"], ["a1", "b1"], ["a2", "b2"]]  # divider row dropped


def test_escaped_pipe_stays_in_prose():
    _, blocks = parse(SAMPLE)
    para = [b for b in blocks if b.kind == "para"][1]
    assert para.text == "Body text with an escaped \\| pipe."


def test_inline_bold_is_split_into_runs():
    assert inline_runs("A **bold** term.") == [
        ("A ", False, False),
        ("bold", True, False),
        (" term.", False, False),
    ]
    assert inline_runs("**First.** Then **second**.") == [
        ("First.", True, False),
        (" Then ", False, False),
        ("second", True, False),
        (".", False, False),
    ]


def test_inline_code_is_split_into_runs():
    assert inline_runs("Run `flutter test` now.") == [
        ("Run ", False, False),
        ("flutter test", False, True),
        (" now.", False, False),
    ]
    # A code span owns its contents: asterisks inside it stay literal.
    assert inline_runs("Glob `a**b` here") == [
        ("Glob ", False, False),
        ("a**b", False, True),
        (" here", False, False),
    ]


def test_inline_markup_preserves_unmatched_markers():
    assert inline_runs("An **unfinished marker") == [("An **unfinished marker", False, False)]
    assert inline_runs("An `unfinished span") == [("An `unfinished span", False, False)]
    assert inline_runs("ordinary text") == [("ordinary text", False, False)]


def test_code_fence_is_kept_verbatim():
    _, blocks = parse("Intro:\n\n```\ncd app; flutter test\nsecond --line\n```\n\nAfter.\n")
    assert [b.kind for b in blocks] == ["para", "code", "para"]
    assert blocks[1].text == "cd app; flutter test\nsecond --line"


def test_citation_field_codes():
    assert citation_code([3]) == "CITATION src3 \\l 1033"
    assert citation_code([13, 14]) == "CITATION src13 \\m src14 \\l 1033"


def test_ragged_table_is_padded():
    _, blocks = parse("| a | b | c |\n|---|---|---|\n| 1 |\n")
    assert blocks[0].rows == [["a", "b", "c"], ["1", "", ""]]


def test_bad_input_is_rejected():
    for bad in ("@@nope\n", "@@sect sideways\n", "@@figure\n", "---\ntitle: x\n"):
        try:
            parse(bad)
        except ValueError:
            continue
        raise AssertionError("parse accepted bad input: %r" % bad)


def test_real_report_if_present():
    source = Path(__file__).resolve().parents[1] / "report" / "report.md"
    if not source.exists():
        return
    meta, blocks = parse(source.read_text(encoding="utf-8"))
    for key in ("title", "author", "student_id", "supervisor", "programme", "affiliation"):
        assert meta.get(key), "cover page is missing %s" % key
    kinds = [b.kind for b in blocks]
    # Structural invariants, not content counts: the report grows, the shape does not.
    assert kinds.count("sect") == 2, "front matter, then body"
    assert kinds.count("toc") == kinds.count("lof") == kinds.count("lot") == 1
    assert kinds.count("h1") >= 8, "eight numbered chapters at minimum"
    assert kinds.count("figure") >= 8 and kinds.count("table") >= 5

    media = source.parent / "media"
    for block in blocks:
        if block.kind == "figure":
            assert (media / block.image).exists(), "missing image %s" % block.image
            assert block.width and block.height, "figure %s has no size" % block.image
        if block.kind == "table":
            widths = {len(row) for row in block.rows}
            assert len(widths) == 1, "ragged table near %r" % block.caption[:40]

    # The reference list is a Word BIBLIOGRAPHY field over report/sources.json, so
    # the heading carries the field and nothing else — no typed entries.
    assert kinds.count("bibliography") == 1, "expected one BIBLIOGRAPHY field"
    inside = False
    for block in blocks:
        if block.kind == "h1":
            inside = block.text.strip().lower() == "bibliography"
        elif inside:
            assert block.kind == "bibliography", (
                "typed text under the Bibliography heading: %r" % block.text[:40]
            )
            inside = False

    # Every [n] the prose cites must resolve, or Word renders an error in its place.
    sources = load_sources()
    assert len(sources) >= 20, "expected a full reference list, found %d" % len(sources)
    cited = set()
    for block in blocks:
        if block.kind in ("para", "bullet"):
            for match in CITATION_RE.finditer(block.text):
                cited.update(int(n) for n in match.group(1).split(","))
    missing = sorted(cited - set(sources))
    assert not missing, "cited but not in sources.json: %s" % missing
    # Word's BIBLIOGRAPHY field only prints sources something cites.
    orphans = sorted(set(sources) - cited)
    assert not orphans, "in sources.json but never cited: %s" % orphans


class FakeDoc:
    """Stands in for a Word document so autosave can be tested without Word."""

    def __init__(self, fail=False):
        self.fail = fail
        self.save_as_calls = 0
        self.save_calls = 0

    def SaveAs2(self, path, FileFormat=None):
        self.save_as_calls += 1
        if self.fail:
            raise RuntimeError("Word cannot save this file because it is already open elsewhere.")

    def Save(self):
        self.save_calls += 1
        if self.fail:
            raise RuntimeError("Word cannot save this file because it is already open elsewhere.")


def test_autosave_first_save_then_incremental(tmp_path=None):
    target = Path(tmp_path or ".") / "out.docx"
    doc = FakeDoc()
    a = Autosave(doc, target, min_seconds=0.001, max_seconds=0.001)
    assert a.save() is True
    assert (doc.save_as_calls, doc.save_calls) == (1, 0), "first save must name the file"
    assert a.save() is True
    assert (doc.save_as_calls, doc.save_calls) == (1, 1), "later saves reuse the path"
    assert a.saves == 2 and a.failures == 0


def test_autosave_survives_a_locked_file(tmp_path=None):
    doc = FakeDoc(fail=True)
    a = Autosave(doc, Path(tmp_path or ".") / "locked.docx",
                 min_seconds=0.001, max_seconds=0.001)
    assert a.save() is False, "a failed save must not raise"
    assert a.failures == 1 and a.saves == 0 and a.saved_once is False
    a.next_at = 0  # force the periodic path rather than racing the wall clock
    a.maybe()  # must also swallow the failure there
    assert a.failures == 2
    assert a.next_at > 0, "a failed save must still schedule the next attempt"


def test_autosave_respects_its_interval():
    doc = FakeDoc()
    a = Autosave(doc, Path(".") / "out.docx", min_seconds=999, max_seconds=999)
    a.maybe()
    assert doc.save_as_calls == 0, "must not save before the interval elapses"
    disabled = Autosave(doc, Path(".") / "out.docx", enabled=False,
                        min_seconds=0.001, max_seconds=0.001)
    disabled.maybe()
    assert doc.save_as_calls == 0, "disabled autosave must never save"
    # A zero upper bound is the other way to turn it off, and must not mean
    # "save on every block".
    zeroed = Autosave(doc, Path(".") / "out.docx", min_seconds=0, max_seconds=0)
    zeroed.maybe()
    assert doc.save_as_calls == 0, "max_seconds=0 must disable autosave"


class FakeApp:
    """WordWriter only touches app.Selection during construction."""

    Selection = None


def writer(**kw):
    """A WordWriter with no Word behind it — enough to exercise the pause logic."""
    kw.setdefault("pause_seconds", (0, 0))  # keep tests instant
    return WordWriter(FakeApp(), doc=None, **kw)


def test_pause_gap_stays_in_range():
    w = writer(pause_chars=(10, 20))
    for _ in range(200):
        gap = w.next_pause_gap()
        assert 10 <= gap <= 20, gap
    fixed = writer(pause_chars=(7, 7))
    assert fixed.next_pause_gap() == 7, "a zero-width range is still valid"


def test_pauses_can_be_disabled():
    for w in (writer(pauses=False), writer(pause_chars=(0, 0))):
        assert w.next_pause_gap() == float("inf")
        assert w.chars_until_pause == float("inf"), "must never reach a pause"


def test_pause_waits_for_a_word_boundary():
    w = writer(pause_at_boundary=True)
    assert w.may_pause_after("x", 5, 50) is False, "must not break mid-word"
    assert w.may_pause_after(" ", 5, 50) is True
    assert w.may_pause_after(".", 5, 50) is True
    assert w.may_pause_after(",", 5, 50) is True
    assert w.may_pause_after("x", 50, 50) is True, "end of text is a boundary"

    anywhere = writer(pause_at_boundary=False)
    assert anywhere.may_pause_after("x", 5, 50) is True


def test_think_rearms_and_counts():
    w = writer(pause_chars=(4, 4), pause_seconds=(0, 0))
    assert w.chars_until_pause == 4
    w.chars_until_pause = 0
    w.think()
    assert w.n_pauses == 1
    assert w.chars_until_pause == 4, "next pause must be armed again"
    w.think()
    assert w.n_pauses == 2 and w.paused_seconds == 0.0


def test_pause_seconds_stay_in_range():
    w = writer(pause_chars=(1, 1), pause_seconds=(0.01, 0.02))
    w.think()
    assert 0.01 <= w.paused_seconds <= 0.02, w.paused_seconds


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok  %s" % test.__name__)
    print("%d checks passed" % len(tests))
