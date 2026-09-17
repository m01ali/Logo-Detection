"""Canonical brand list (T5): loader, normalisation, aliases, hash (06).

The list is a versioned YAML snapshot of the Roboflow class names. The loader
recomputes the content hash on every load and the same ``map_fuzzy`` serves
OCR matching (T26/T27) and canonicalisation (T37) so the two never disagree.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import yaml

NORMALISATION_RULE = "lowercase; unicode NFKD; strip accents; keep [a-z0-9 ]; collapse spaces"
_TOP_KEYS = {"version", "source", "content_hash", "normalisation", "brands"}
_ENTRY_KEYS = {"canonical", "aliases", "channel", "deprecated"}
_VERSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class BrandListError(ValueError):
    """Validation failure of the snapshot file."""


def normalise(s: str | None) -> str:
    """``NORMALISATION_RULE`` applied to any string; ``None`` -> ``""``."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass(frozen=True)
class BrandEntry:
    canonical: str
    aliases: tuple[str, ...] = ()
    channel: bool = False
    deprecated: bool = False


def _effective_aliases(entry: BrandEntry) -> list[str]:
    """Aliases after dropping self-collisions and normalised duplicates —
    the same rule ``_parse_entries`` applies when reading a file back. Used
    by the hash so it is stable whether ``entries`` came from a fresh
    construction (the writer) or from re-parsing the YAML (the loader):
    without this, a raw, undeduplicated alias list (e.g. both "l'oreal" and
    "l oreal" for "loreal" — 06 §2 lists both, and normalisation collapses
    "l'oreal" onto the canonical name itself) would hash differently before
    and after its first save/reload round trip."""
    own_key = normalise(entry.canonical)
    seen: set[str] = set()
    out: list[str] = []
    for a in entry.aliases:
        k = normalise(a)
        if not k or k == own_key or k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out


def compute_content_hash(version: str, entries: Iterable[BrandEntry]) -> str:
    doc = {
        "version": version,
        "brands": sorted(
            (
                {
                    "canonical": e.canonical,
                    "aliases": sorted(_effective_aliases(e)),
                    "channel": bool(e.channel),
                    "deprecated": bool(e.deprecated),
                }
                for e in entries
            ),
            key=lambda d: d["canonical"],
        ),
    }
    digest = hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class BrandList:
    version: str
    content_hash: str
    canonical: tuple[str, ...]                     # active (non-deprecated) canonical names, file order
    alias_to_canonical: Mapping[str, str]          # normalised alias -> canonical
    channel_brands: frozenset[str]
    entries: tuple[BrandEntry, ...] = ()
    source: str = ""
    path: str | None = None
    stored_hash: str | None = None
    warnings: tuple[str, ...] = ()
    _canonical_keys: Mapping[str, str] = field(default_factory=dict, repr=False)   # normalised canonical -> canonical

    # -- derived ---------------------------------------------------------
    @property
    def hash_matches(self) -> bool | None:
        return None if self.stored_hash is None else self.stored_hash == self.content_hash

    @property
    def short_names(self) -> tuple[str, ...]:
        return tuple(c for c in self.canonical if len(normalise(c)) <= 4)

    def normalise(self, s: str) -> str:
        return normalise(s)

    # -- matching --------------------------------------------------------
    def map_exact_with_mode(self, s: str) -> tuple[str | None, str | None]:
        key = normalise(s)
        if not key:
            return None, None
        if key in self.alias_to_canonical:
            return self.alias_to_canonical[key], "alias"
        if key in self._canonical_keys:
            return self._canonical_keys[key], "exact"
        return None, None

    def map_exact(self, s: str) -> str | None:
        """Alias table first, then the canonical set. ``None`` when unmatched."""
        return self.map_exact_with_mode(s)[0]

    def _fuzzy_candidates(self, short_cutoff: int) -> tuple[list[str], list[str]]:
        keys: list[str] = []
        canon: list[str] = []
        for c in self.canonical:
            k = normalise(c)
            if len(k) > short_cutoff:
                keys.append(k)
                canon.append(c)
        for k, c in self.alias_to_canonical.items():
            if len(k) > short_cutoff:
                keys.append(k)
                canon.append(c)
        return keys, canon

    def map_fuzzy(self, s: str, threshold: int = 90, short_cutoff: int = 4) -> tuple[str | None, int, str]:
        """Length-dependent matching (T27).

        Returns ``(canonical | None, score, mode)`` where ``mode`` names the
        rule applied: ``alias``/``exact`` (exact match on a long string),
        ``exact_short`` (the string or candidate is short: only an exact
        normalised match counts, score 100 or 0), ``fuzzy`` (rapidfuzz
        ``WRatio`` over long canonical names and aliases; ``canonical`` is
        ``None`` when the best score is below ``threshold``) or
        ``unmatched`` (empty string).
        """
        key = normalise(s)
        if not key:
            return None, 0, "unmatched"
        exact, mode = self.map_exact_with_mode(key)
        if exact is not None:
            return exact, 100, "exact_short" if len(key) <= short_cutoff else str(mode)
        if len(key) <= short_cutoff:
            return None, 0, "exact_short"
        keys, canon = self._fuzzy_candidates(short_cutoff)
        if not keys:
            return None, 0, "fuzzy"
        from rapidfuzz import fuzz, process
        best = process.extractOne(key, keys, scorer=fuzz.WRatio, processor=None)
        if best is None:
            return None, 0, "fuzzy"
        _, score, idx = best
        score_i = int(round(float(score)))
        if score_i >= threshold:
            return canon[idx], score_i, "fuzzy"
        return None, score_i, "fuzzy"

    def is_channel(self, canonical: str | None) -> bool:
        return canonical is not None and canonical in self.channel_brands

    def entry(self, canonical: str) -> BrandEntry | None:
        for e in self.entries:
            if e.canonical == canonical:
                return e
        return None


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def _parse_entries(raw_brands: object, strict: bool = True) -> tuple[list[BrandEntry], list[str]]:
    if not isinstance(raw_brands, list):
        raise BrandListError("'brands' must be a list")
    entries: list[BrandEntry] = []
    warnings: list[str] = []
    for i, item in enumerate(raw_brands):
        if not isinstance(item, dict):
            raise BrandListError(f"brands[{i}]: each entry must be a mapping")
        unknown = sorted(set(item) - _ENTRY_KEYS)
        if unknown:
            raise BrandListError(f"brands[{i}]: unknown key(s) {unknown}")
        canonical = item.get("canonical")
        if not isinstance(canonical, str) or not canonical.strip():
            raise BrandListError(f"brands[{i}]: 'canonical' must be a non-empty string")
        if canonical != canonical.strip() or canonical != canonical.lower():
            raise BrandListError(f"brands[{i}]: canonical {canonical!r} must be lowercase without surrounding spaces")
        if not normalise(canonical):
            raise BrandListError(f"brands[{i}]: canonical {canonical!r} is empty after normalisation")
        aliases_raw = item.get("aliases", []) or []
        if not isinstance(aliases_raw, list) or not all(isinstance(a, str) for a in aliases_raw):
            raise BrandListError(f"brands[{i}] ({canonical}): 'aliases' must be a list of strings")
        channel = item.get("channel", False)
        deprecated = item.get("deprecated", False)
        if not isinstance(channel, bool) or not isinstance(deprecated, bool):
            raise BrandListError(f"brands[{i}] ({canonical}): 'channel' and 'deprecated' must be booleans")
        own_key = normalise(canonical)
        seen: set[str] = set()
        aliases: list[str] = []
        for a in aliases_raw:
            if a != a.lower():
                raise BrandListError(f"brands[{i}] ({canonical}): alias {a!r} must be lowercase")
            k = normalise(a)
            if not k:
                raise BrandListError(f"brands[{i}] ({canonical}): alias {a!r} is empty after normalisation")
            if k == own_key:
                warnings.append(f"{canonical}: alias {a!r} equals its own canonical name after normalisation; dropped")
                continue
            if k in seen:
                warnings.append(f"{canonical}: duplicate alias {a!r} after normalisation; dropped")
                continue
            seen.add(k)
            aliases.append(a)
        entries.append(BrandEntry(canonical=canonical, aliases=tuple(aliases), channel=channel, deprecated=deprecated))
    return entries, warnings


def build_brand_list(
    version: str,
    entries: Iterable[BrandEntry],
    *,
    source: str = "",
    path: str | None = None,
    stored_hash: str | None = None,
    warnings: Iterable[str] = (),
) -> BrandList:
    entries = tuple(entries)
    if not _VERSION_RE.match(str(version)):
        raise BrandListError(f"version must be YYYY-MM-DD, got {version!r}")
    canonical_keys: dict[str, str] = {}
    for e in entries:
        k = normalise(e.canonical)
        if k in canonical_keys:
            raise BrandListError(f"duplicate canonical name {e.canonical!r} (collides with {canonical_keys[k]!r})")
        canonical_keys[k] = e.canonical
    alias_map: dict[str, str] = {}
    active_keys = {normalise(e.canonical): e.canonical for e in entries if not e.deprecated}
    for e in entries:
        if e.deprecated:
            continue
        for a in e.aliases:
            k = normalise(a)
            if k in active_keys:
                raise BrandListError(
                    f"{e.canonical}: alias {a!r} collides with canonical name {active_keys[k]!r}")
            if k in alias_map and alias_map[k] != e.canonical:
                raise BrandListError(
                    f"{e.canonical}: alias {a!r} collides with alias of {alias_map[k]!r}")
            alias_map[k] = e.canonical
    return BrandList(
        version=str(version),
        content_hash=compute_content_hash(str(version), entries),
        canonical=tuple(e.canonical for e in entries if not e.deprecated),
        alias_to_canonical=alias_map,
        channel_brands=frozenset(e.canonical for e in entries if e.channel and not e.deprecated),
        entries=entries,
        source=source,
        path=path,
        stored_hash=stored_hash,
        warnings=tuple(warnings),
        _canonical_keys=active_keys,
    )


def parse_brand_yaml(data: dict, path: str | None = None) -> BrandList:
    if not isinstance(data, dict):
        raise BrandListError("top level must be a mapping")
    unknown = sorted(set(data) - _TOP_KEYS)
    if unknown:
        raise BrandListError(f"unknown top-level key(s) {unknown}")
    if "version" not in data or "brands" not in data:
        raise BrandListError("'version' and 'brands' are required")
    entries, warnings = _parse_entries(data["brands"])
    return build_brand_list(
        str(data["version"]), entries,
        source=str(data.get("source", "") or ""),
        path=path,
        stored_hash=data.get("content_hash"),
        warnings=warnings,
    )


def load_brand_list(path: str | Path, require_hash_match: bool = False) -> BrandList:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    bl = parse_brand_yaml(data, path=str(path))
    if require_hash_match and bl.hash_matches is False:
        raise BrandListError(
            f"{path}: stored content_hash {bl.stored_hash} != recomputed {bl.content_hash} "
            "(run `python -m logo_pipeline brands hash`)")
    return bl


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _yq(s: str) -> str:
    """Double-quoted YAML scalar."""
    return json.dumps(s, ensure_ascii=False)


def render_brand_yaml(version: str, source: str, entries: Iterable[BrandEntry], content_hash: str | None = None) -> str:
    entries = list(entries)
    if content_hash is None:
        content_hash = compute_content_hash(version, entries)
    lines = [
        f"version: {_yq(version)}",
        f"source: {_yq(source)}",
        f"content_hash: {_yq(content_hash)}",
        f"normalisation: {_yq(NORMALISATION_RULE)}",
        "brands:",
    ]
    for e in entries:
        lines.append(f"  - canonical: {_yq(e.canonical)}")
        if e.aliases:
            lines.append("    aliases: [" + ", ".join(_yq(a) for a in e.aliases) + "]")
        if e.channel:
            lines.append("    channel: true")
        if e.deprecated:
            lines.append("    deprecated: true")
    return "\n".join(lines) + "\n"


def write_brand_list(path: str | Path, version: str, source: str, entries: Iterable[BrandEntry]) -> str:
    entries = list(entries)
    text = render_brand_yaml(version, source, entries)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return compute_content_hash(version, entries)


def rewrite_hash(path: str | Path) -> BrandList:
    """``brands hash``: validate, recompute and rewrite ``content_hash``."""
    bl = load_brand_list(path)
    write_brand_list(path, bl.version, bl.source, bl.entries)
    return load_brand_list(path)


def add_brand(path: str | Path, canonical: str, aliases: Iterable[str] = (), channel: bool = False,
              version: str | None = None) -> BrandList:
    """``brands add``: the 06 §4 procedure as one command."""
    bl = load_brand_list(path)
    entries = list(bl.entries)
    entries.append(BrandEntry(canonical=canonical.strip().lower(), aliases=tuple(a.strip().lower() for a in aliases),
                              channel=channel))
    new_version = version or bl.version
    build_brand_list(new_version, entries)   # validates collisions before touching the file
    write_brand_list(path, new_version, bl.source, entries)
    return load_brand_list(path)
