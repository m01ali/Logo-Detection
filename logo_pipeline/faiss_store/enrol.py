"""The one instrumented enrolment path into the FAISS index (T55, ``spec docs/16``;
first caller T75, ``spec docs/13``).

``enrol`` is the only function that calls ``db.add_logos``. It reads
``db.index.ntotal`` before and after, so every new vector gets a provenance
record keyed by its vector index; augmentation copies point back at their
original through ``augmented_of``.

Storage [ASSUMPTION, ``19`` B6]: ``metadata.json`` stays a flat list of brand
strings (every existing reader keeps working) and provenance goes to
``enrolments.jsonl`` next to it. Records are held on the database object until
``db.save()`` writes the index, then appended, so the file never names a
vector the saved index does not have. Vectors without a record are legacy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

ORIGINS = (
    "faiss_manager.add_logo", "faiss_manager.video_crops", "faiss_manager.unknown_single",
    "faiss_manager.unknown_bulk", "faiss_manager.known_update", "faiss_manager.known_false_logo",
    "faiss_manager.known_bulk_correction", "faiss_manager.known_bulk_confirm", "review_ui.accept",
)
LEGACY = "legacy"


@dataclass
class EnrolmentRecord:
    brand: str
    source_id: str
    enrolled_at: str
    origin: str
    enrolled_by: str = "unknown"
    video_id: str | None = None
    run_id: str | None = None
    frame_idx: int | None = None
    det_id: int | None = None
    crop_path: str | None = None
    review_label: str | None = None
    augmented_of: int | None = None      # vector index of the original when this is an augmentation copy
    vector_index: int | None = None      # set by enrol(); None only when the index could not be resolved

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EnrolmentRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def enrolments_path(db) -> Path:
    """``FAISS/enrolments.jsonl`` for ``metadata.json``; ``enrolments_siglip2.jsonl`` for ``metadata_siglip2.json``."""
    meta = Path(db.metadata_path)
    stem = meta.stem.replace("metadata", "enrolments") if "metadata" in meta.stem else meta.stem + "_enrolments"
    return meta.with_name(stem + ".jsonl")


def _pending(db) -> list[EnrolmentRecord]:
    if not hasattr(db, "_pending_enrolments"):
        db._pending_enrolments = []
        db._after_save = flush            # LogoDatabaseNew.save() calls this once the index is written
    return db._pending_enrolments


def enrol(db, images: list, records: list[EnrolmentRecord], augmentations=None, num_augments: int = 0) -> list[int]:
    """Adds ``images`` (plus ``num_augments`` augmentations each when ``augmentations`` is given) and
    queues one record per new vector. Returns the vector indices of the originals."""
    if len(images) != len(records):
        raise ValueError("images and records must have the same length")
    bad = [r.origin for r in records if r.origin not in ORIGINS]
    if bad:
        raise ValueError(f"unknown enrolment origin(s) {sorted(set(bad))}; allowed: {list(ORIGINS)}")
    if not images:
        return []
    n_aug = num_augments if augmentations else 0
    before = int(db.index.ntotal)
    db.add_logos(images, [r.brand for r in records], augmentations=augmentations, num_augments=n_aug)
    added = int(db.index.ntotal) - before
    resolved = added == len(images) * (1 + n_aug)

    queue = _pending(db)
    originals: list[int] = []
    for i, rec in enumerate(records):
        base = before + i * (1 + n_aug) if resolved else None
        originals.append(base)
        queue.append(EnrolmentRecord(**{**rec.to_dict(), "vector_index": base, "augmented_of": None}))
        for j in range(n_aug):
            queue.append(EnrolmentRecord(**{**rec.to_dict(), "vector_index": base + 1 + j if resolved else None,
                                            "augmented_of": base}))
    if not resolved:
        import warnings
        warnings.warn(f"enrol: expected {len(images) * (1 + n_aug)} new vectors, index grew by {added} "
                      "(an embedding batch failed); these records carry vector_index null")
    return originals


def flush(db) -> int:
    """Appends the queued records to ``enrolments.jsonl``; returns how many were written."""
    queue = getattr(db, "_pending_enrolments", None)
    if not queue:
        return 0
    path = enrolments_path(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for rec in queue:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    n = len(queue)
    queue.clear()
    return n


def save(db) -> int:
    """``db.save()`` plus the record flush, for databases whose ``save`` has no hook."""
    db.save()
    return flush(db)


def read_enrolments(path: str | Path) -> dict[int, EnrolmentRecord]:
    out: dict[int, EnrolmentRecord] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = EnrolmentRecord.from_dict(json.loads(line))
            if rec.vector_index is not None:
                out[rec.vector_index] = rec
    return out


def record_for(vector_index: int, metadata: list[str], enrolments: dict[int, EnrolmentRecord]) -> EnrolmentRecord:
    """The legacy-list adapter: an enrolled vector's record, or a legacy record built from ``metadata.json``."""
    if vector_index in enrolments:
        return enrolments[vector_index]
    return EnrolmentRecord(brand=metadata[vector_index], source_id=f"legacy#{vector_index}", enrolled_at="",
                           origin=LEGACY, vector_index=vector_index)
