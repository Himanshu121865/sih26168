"""Run-scoped structured logging (loguru) with run IDs.

CLI scripts call :func:`init_runlog` once at startup; every record then carries
``run_id`` so Colab outputs from different runs stay attributable. Optional
``--log-dir`` tees the same records to ``<name>_<run_id>.log``.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def init_runlog(name: str, log_dir: str | Path | None = None, run_id: str | None = None) -> str:
    """Configure loguru for a CLI run (idempotent per process).

    Args:
        name: Script short name used in the log line and file stem.
        log_dir: Optional directory for a ``<name>_<run_id>.log`` tee.
        run_id: Optional fixed id (defaults to 8 random hex chars).

    Returns:
        The run id bound to every subsequent record.
    """
    global _CONFIGURED
    rid = run_id or uuid.uuid4().hex[:8]
    if not _CONFIGURED:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            + name
            + " {extra[run_id]} | {message}",
        )
        _CONFIGURED = True
    logger.configure(extra={"run_id": rid})
    if log_dir is not None:
        dest = Path(log_dir)
        dest.mkdir(parents=True, exist_ok=True)
        logger.add(dest / f"{name}_{rid}.log", format="{time} | {level} | {extra[run_id]} | {message}")
    logger.info("started")
    return rid
