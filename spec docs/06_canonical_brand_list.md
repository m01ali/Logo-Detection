# 06 — Canonical Brand List Snapshot (T5)

Decision locked: the 321-name Roboflow class list is the canonical list for M3, committed as a YAML snapshot keyed by export date with a content hash, aliases in the same file, version recorded in every run output. Adding a brand is a documented file edit.

## 1. File format `configs/brands/roboflow_2026-04-28.yaml`

```yaml
version: "2026-04-28"                 # the Roboflow export date; becomes the date of the last edit after any change
source: "Roboflow project antos-workspace-cttem, dataset export 28 Apr 2026, data.yaml names[]"
content_hash: "sha256:…"              # recomputed by the loader; see §3
normalisation: "lowercase; unicode NFKD; strip accents; keep [a-z0-9 ]; collapse spaces"
brands:
  - canonical: "rai"
    aliases: ["rai 1", "rai 2", "rai 3", "rai uno", "rai due", "rai tre", "rai news 24", "rai news", "raiplay"]
    channel: true
  - canonical: "rai sport"
    aliases: ["rai sport hd", "raisport"]
    channel: true
  - canonical: "verisure"
    aliases: ["very sure", "verishure", "verishore", "berichur"]
  - canonical: "wolkswagen"           # the client's spelling is canonical (it is their list)
    aliases: ["volkswagen", "vw"]
  - canonical: "8 per mille chiesa cattolica"
    aliases: ["8xmille", "8 x mille", "otto per mille"]
  # … one entry per class, 321 entries, in data.yaml order
```

Rules:

- `canonical` is the exact string from `data.yaml`, already lowercase. It is the value written to `canonical_brand` and used for dedup grouping.
- `aliases` are optional, lowercase, and matched after normalisation. An alias may not equal another entry's canonical name or alias; the validator rejects collisions.
- `channel: true` marks broadcaster brands; this set is `known_channel_brands` for Rule 5 unless the per-video config narrows it (T47, `configs/watermark_regions.yaml` `channel_brands`).
- Entries are never deleted in M3; a brand that should stop matching gets `deprecated: true` and the loader ignores it for matching but keeps it for reporting.

## 2. Initial alias seeds (from the audit)

| Canonical | Alias candidates | Evidence |
|-----------|------------------|----------|
| `verisure` | `very sure`, `verishure`, `verishore`, `berichur` | `predicted_brands.json`, M2 transcript |
| `rai sport` | `rai sport hd`, `raisport` | M2 CSV shows `Rai Sport HD` |
| `rai` | `rai 2`, `rai 1`, `rai 3`, `rai news 24`, … | audio brands, frames |
| `orogel` | — | exact |
| `frosta` | `frost` | audio: `frost` |
| (not in list) | `neo borocillina`, `neoborocillina` | The video_1 advertiser NeoBorocillina is absent from the 321 names. It will route to UNKNOWN_LOGO via Rule 16 and is the first live test of the add-a-brand procedure. |
| `wolkswagen` | `volkswagen`, `vw` | misspelling in the client list |
| `wind tre` | `windtre`, `wind 3`, `wind` | frames |
| `mcdonald` | `mcdonald's`, `mcdonalds` | list spelling |
| `loreal` | `l'oreal`, `l oreal` | normalisation drops the apostrophe anyway |
| `hotelleap.com`, `immobiliare.it`, `noicompriamoauto.it` | `hotelleap`, `immobiliare`, `noicompriamoauto` | Normalisation strips the dot, so the canonical forms become `hotelleapcom` etc.; the TLD-less aliases are what OCR reads. |

Seed the file with these and let the UNKNOWN_LOGO queue and the T37 `unmatched` log grow it.

## 3. Content hash

`content_hash = sha256(canonical_json)` where `canonical_json` is `json.dumps({"version": …, "brands": [{canonical, aliases(sorted), channel, deprecated} …sorted by canonical]}, sort_keys=True, ensure_ascii=False)`. The stored value is informational; the loader recomputes it on every load and writes the recomputed value into the run output. With `brands.require_hash_match: true` the loader refuses to run when they differ (default false so a forgotten `brands hash` does not block a run; the mismatch is logged as a warning).

## 4. Procedure: adding a brand (the text that goes into the schema document)

1. Open `configs/brands/<current version>.yaml`.
2. Append an entry: `canonical` (lowercase, the spelling to appear on every row), `aliases` (any spellings the VLM, OCR or audio have produced), `channel: true` only for broadcasters.
3. Set `version` at the top to today's date (`YYYY-MM-DD`).
4. Run `python -m logo_pipeline brands hash --list configs/brands/<file>` — it validates (no collisions, no duplicates, valid characters) and rewrites `content_hash`.
5. Commit. The next run records the new version and hash on every row. Earlier runs keep the version they were produced with, so results remain comparable.

There is no code change and no re-run required; a re-run of the canonicalisation and router stages on an existing run directory (`--from-stage canonicalise --to-stage router`) applies the new list to old VLM output in minutes.

The intended source of additions is the UNKNOWN_LOGO queue in the review screen (T73): a reviewer who labels a queued row with a new brand name adds it here, then optionally enrols the crop through T75.

## 5. Loader API (`logo_pipeline/brands.py`)

```python
@dataclass(frozen=True)
class BrandList:
    version: str
    content_hash: str
    canonical: tuple[str, ...]
    alias_to_canonical: Mapping[str, str]     # normalised alias -> canonical
    channel_brands: frozenset[str]

    def normalise(self, s: str) -> str
    def map_exact(self, s: str) -> str | None          # alias table first, then canonical set
    def map_fuzzy(self, s: str, threshold: int, short_cutoff: int) -> tuple[str | None, int, str]  # (canonical, score, mode)
```

`map_fuzzy` implements the length-dependent rule (T27): if `len(normalised) <= short_cutoff` or the candidate canonical name is that short, only exact matches count; otherwise rapidfuzz `WRatio` over canonical names and aliases with the given threshold. This same function serves T26 (OCR) and T37 (canonicalisation) so the two never disagree.

## 6. Short names in the list

Exactly 37 canonical names are four characters or fewer (verified against `data.yaml`): `aia`, `ail`, `aws`, `bmw`, `chef`, `coop`, `dash`, `dkv`, `doc`, `eni`, `fage`, `fiat`, `hera`, `hp`, `ihc`, `ikea`, `iwc`, `kia`, `kiko`, `lego`, `lidl`, `m6`, `mbe`, `msc`, `nike`, `nove`, `oasy`, `oreo`, `puma`, `q8`, `rai`, `sela`, `sky`, `tim`, `uber`, `ubs`, `x`. Under T27 they match only on an exact normalised OCR read. `x` (the platform) is a one-character canonical name and will match only a lone `x`, which is the intended conservative behaviour and is worth stating to the client.
