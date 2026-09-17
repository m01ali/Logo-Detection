"""Stage progress logging (03 §1 goal 4, §3 rule 5).

``StageLogger`` writes ``stage=<name> frame=i/N det=j/M elapsed=…`` lines at
the configured cadence to stdout and to ``<run_dir>/logs/pipeline.log``.
Stage timings are collected and written by the runner into
``stage_status.json``.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

LOGGER_NAME = "logo_pipeline"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


class StageLogger:
    def __init__(self, run_dir: Path | None, log_every_n: int = 10, level: int = logging.INFO):
        self.log_every_n = max(1, int(log_every_n))
        self.logger = get_logger()
        self.logger.setLevel(level)
        self.logger.propagate = False
        self._stage: str | None = None
        self._t0: float = 0.0
        self._file_handler: logging.Handler | None = None

        fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
        if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_lp_stream", False)
                   for h in self.logger.handlers):
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            sh._lp_stream = True  # type: ignore[attr-defined]
            self.logger.addHandler(sh)
        if run_dir is not None:
            log_dir = Path(run_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
            self._file_handler = fh

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        if self._file_handler is not None:
            self.logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    def stage_start(self, name: str, device: str) -> None:
        self._stage = name
        self._t0 = time.perf_counter()
        self.logger.info("stage=%s status=start device=%s", name, device)

    def stage_end(self, name: str, counts: dict | None = None) -> float:
        elapsed = time.perf_counter() - self._t0
        extra = " ".join(f"{k}={v}" for k, v in (counts or {}).items() if not isinstance(v, (dict, list)))
        self.logger.info("stage=%s status=done elapsed=%.1fs %s", name, elapsed, extra)
        self._stage = None
        return elapsed

    def stage_skipped(self, name: str, reason: str) -> None:
        self.logger.info("stage=%s status=skipped reason=%s", name, reason)

    # -- progress ------------------------------------------------------------
    def progress(self, i: int, n: int, unit: str = "frame", det: tuple[int, int] | None = None) -> None:
        """Log ``i/N`` when ``i`` hits the cadence or the end."""
        if i % self.log_every_n != 0 and i != n:
            return
        elapsed = time.perf_counter() - self._t0
        det_part = f" det={det[0]}/{det[1]}" if det else ""
        self.logger.info("stage=%s %s=%d/%d%s elapsed=%.1fs", self._stage, unit, i, n, det_part, elapsed)

    def info(self, msg: str, *args) -> None:
        self.logger.info(msg, *args)

    def warning(self, msg: str, *args) -> None:
        self.logger.warning(msg, *args)

    def error(self, msg: str, *args) -> None:
        self.logger.error(msg, *args)
