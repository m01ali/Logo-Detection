"""Shared pytest fixtures (21_test_plan.md). Ensures the repo root (and
therefore ``logo_pipeline``, ``generate_hints``, ``models``, ``utils``) is
importable regardless of the working directory pytest is invoked from."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# faiss-cpu and torch each ship an OpenMP runtime; on macOS loading both in one
# process aborts (OMP Error #15; app/streamlit_faiss_app.py sets the same switch),
# and with both loaded a multithreaded faiss search can segfault. One OpenMP thread
# in the test process avoids both; the tiny test indexes do not need more.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def brands_small_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "brands_small.yaml"
