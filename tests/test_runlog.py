"""Tests for run-scoped logging (need loguru — run on Colab/CI)."""

from __future__ import annotations

from python.core.runlog import init_runlog


def test_init_runlog_returns_stable_id_for_explicit_input() -> None:
    """Explicit run id is returned verbatim (reproducible logs)."""
    assert init_runlog("test", run_id="abc123") == "abc123"


def test_init_runlog_generates_short_id() -> None:
    """Generated ids are 8 hex chars."""
    rid = init_runlog("test")
    assert len(rid) == 8
    int(rid, 16)
