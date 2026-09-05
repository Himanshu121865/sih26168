"""Stdlib-only tests for the IO-VNBD downloader (run anywhere)."""

from __future__ import annotations

from pathlib import Path

import pytest

from python.download_iovnbd import SYNC_SIZE, SYNC_URL, download


def test_download_skips_when_size_matches(tmp_path: Path) -> None:
    """Existing file with matching size is not re-downloaded."""
    dest = tmp_path / "sync.zip"
    dest.write_bytes(b"x" * 16)
    assert download("http://example.invalid/sync.zip", dest, 16) == dest
    assert dest.read_bytes() == b"x" * 16


def test_download_warns_on_size_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Size mismatch keeps the file but warns on stderr."""
    import urllib.request

    dest = tmp_path / "sync.zip"

    def fake_retrieve(url: str, filename: object, reporthook: object = None) -> tuple[object, object]:
        Path(str(filename)).write_bytes(b"short")
        return filename, {}

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    download("http://example.invalid/sync.zip", dest, 999)
    assert dest.read_bytes() == b"short"
    assert "size mismatch" in capsys.readouterr().err


def test_sync_constants_sane() -> None:
    """Pin the documented dataset URL/size used by Colab."""
    assert SYNC_URL.startswith("https://")
    assert "Synchronised" in SYNC_URL
    assert SYNC_SIZE == 203606286
