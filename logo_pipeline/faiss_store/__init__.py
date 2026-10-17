"""FAISS enrolment with provenance (T55, T75). Kept outside ``eval/`` because
both the review UI and the FAISS manager write through it (03 §2)."""

from .enrol import EnrolmentRecord, enrol, enrolments_path, flush, read_enrolments, record_for, save, utcnow

__all__ = ["EnrolmentRecord", "enrol", "enrolments_path", "flush", "read_enrolments", "record_for", "save", "utcnow"]
