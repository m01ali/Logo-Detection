#!/usr/bin/env python3
"""T9 (07 MM1 Step 4.4): renders the frozen row schema from
``logo_pipeline/schema.py`` so the document sent to the client is generated
from code and cannot drift. Prints to stdout by default; ``--out`` writes a
file (e.g. back onto ``spec docs/05_row_schema.md``'s table section, or to
``docs/`` as the artefact actually handed to the client).

Usage::

    python scripts/export_schema_doc.py [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logo_pipeline.schema import CSV_COLUMNS, Row, SECTION_TITLES  # noqa: E402


def render() -> str:
    lines = [
        "# Frozen Row Schema (T9, T56)",
        "",
        "Generated from `logo_pipeline/schema.py` by `scripts/export_schema_doc.py`; "
        "do not hand-edit — edit the dataclass and regenerate.",
        "",
        "One row per detection that survives the merge. Column order in `rows.csv` is exactly "
        "the order below. Columns marked ★ exist in M2 today with the same name and meaning "
        "(`app/review_ui.py` reads them).",
        "",
    ]
    by_section: dict[str, list] = {}
    for f in fields(Row):
        by_section.setdefault(f.metadata.get("section", "?"), []).append(f)

    for section, title in SECTION_TITLES.items():
        flds = by_section.get(section, [])
        if not flds:
            continue
        lines.append(f"## {section}. {title}")
        lines.append("")
        lines.append("| Field | Type | M2 | Notes |")
        lines.append("|-------|------|----|-------|")
        for f in flds:
            m2 = "★" if f.metadata.get("m2") else ""
            note = f.metadata.get("note", "")
            lines.append(f"| `{f.name}` | {f.type} | {m2} | {note} |")
        lines.append("")

    lines.append(f"**Total fields:** {len(CSV_COLUMNS)}")
    lines.append("")
    lines.append("## CSV column order")
    lines.append("")
    lines.append("```")
    lines.extend(CSV_COLUMNS)
    lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    text = render()
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(CSV_COLUMNS)} fields)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
