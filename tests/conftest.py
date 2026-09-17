"""Shared pytest fixtures (21_test_plan.md). Ensures the repo root (and
therefore ``logo_pipeline``, ``generate_hints``, ``models``, ``utils``) is
importable regardless of the working directory pytest is invoked from."""

from __future__ import annotations

import sys
from pathlib import Path

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
