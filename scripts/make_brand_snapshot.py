#!/usr/bin/env python3
"""T5, one-off (07 MM0 Step 1): reads ``~/Downloads/logo2.yolov11/data.yaml``
(the Roboflow export) and writes ``configs/brands/roboflow_2026-04-28.yaml``
in the ``06`` format: empty alias lists except the seeds from ``06`` §2,
``channel: true`` for the broadcaster names listed in ``07`` MM0 T5
(**[ASSUMPTION]**, confirm with the client; ``tim`` and ``sky wifi`` are
advertisers as often as channels).

Usage::

    python scripts/make_brand_snapshot.py \
        --data-yaml ~/Downloads/logo2.yolov11/data.yaml \
        --out configs/brands/roboflow_2026-04-28.yaml \
        --version 2026-04-28

Re-running is safe: aliases and the ``channel`` flag are re-applied from the
tables below on every run, so hand edits to *other* brands (adding an alias
learned later) survive only if this script is not re-run over them; the
06 §4 procedure ("add a brand") is the normal way to extend the file after
this first snapshot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from logo_pipeline.brands import BrandEntry, write_brand_list  # noqa: E402

# 07 MM0 T5.1 — broadcaster names marked channel: true. [ASSUMPTION]: confirm
# with the client; `tim` and `sky wifi` are advertisers as often as channels.
CHANNEL_NAMES = {
    "boing", "discovery plus", "infinity plus", "m6", "mediaset", "nove",
    "paramount plus", "rai", "rai fiction", "rete 4", "sky", "sky wifi",
    "sport italia", "tim", "x factor",
}

# 06 §2 — initial alias seeds from the audit.
ALIAS_SEEDS: dict[str, list[str]] = {
    "verisure": ["very sure", "verishure", "verishore", "berichur"],
    "rai sport": ["rai sport hd", "raisport"],
    "rai": ["rai 1", "rai 2", "rai 3", "rai uno", "rai due", "rai tre", "rai news 24", "rai news", "raiplay"],
    "wolkswagen": ["volkswagen", "vw"],
    "wind tre": ["windtre", "wind 3", "wind"],
    "mcdonald": ["mcdonald's", "mcdonalds"],
    "loreal": ["l'oreal", "l oreal"],
    "hotelleap.com": ["hotelleap"],
    "immobiliare.it": ["immobiliare"],
    "noicompriamoauto.it": ["noicompriamoauto"],
    "frosta": ["frost"],
    "8 per mille chiesa cattolica": ["8xmille", "8 x mille", "otto per mille"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-yaml", default=str(Path.home() / "Downloads" / "logo2.yolov11" / "data.yaml"))
    ap.add_argument("--out", default="configs/brands/roboflow_2026-04-28.yaml")
    ap.add_argument("--version", default="2026-04-28")
    args = ap.parse_args()

    data_yaml_path = Path(args.data_yaml)
    data = yaml.safe_load(data_yaml_path.read_text())
    names: list[str] = data["names"]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"data.yaml has duplicate class names: {dupes}")

    roboflow = data.get("roboflow", {})
    source = (f"Roboflow project {roboflow.get('workspace', '?')}, "
              f"dataset export (see {data_yaml_path}), data.yaml names[]")

    entries = [
        BrandEntry(canonical=name, aliases=tuple(ALIAS_SEEDS.get(name, [])), channel=name in CHANNEL_NAMES)
        for name in names
    ]

    unseeded_channel_names = CHANNEL_NAMES - set(names)
    if unseeded_channel_names:
        raise SystemExit(f"CHANNEL_NAMES contains names absent from data.yaml: {sorted(unseeded_channel_names)}")

    content_hash = write_brand_list(args.out, args.version, source, entries)
    print(f"wrote {args.out}: {len(entries)} brands, "
          f"{sum(1 for e in entries if e.channel)} channel brands, "
          f"{sum(1 for e in entries if e.aliases)} with alias seeds")
    print(f"content_hash: {content_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
