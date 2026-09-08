"""Tests for the window-spec fingerprinting module (P2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from python.core.spec import (
    SPEC_VERSION,
    attach_spec,
    file_sha256,
    spec_sha256,
    verify_model_manifest,
    verify_scaler,
    write_model_manifest,
)


@pytest.fixture
def scaler_file(tmp_path: Path) -> Path:
    """A minimal scaler.json without any spec fields."""
    p = tmp_path / "scaler.json"
    p.write_text(json.dumps({"mean": [0.0] * 6, "std": [1.0] * 6, "hz": 100}))
    return p


def test_spec_constants() -> None:
    """Current spec is v2 and the canonical hash is stable."""
    assert SPEC_VERSION == 2
    assert len(spec_sha256()) == 64
    # pinned so any accidental SPEC_TEXT edit fails loudly
    assert spec_sha256() == "063703d0fd897b5280b2c64257dad28fea6743dc03ac05b9ab62f175a6ed42bd"


def test_attach_then_verify_roundtrip(scaler_file: Path) -> None:
    """Stamping adds the fingerprint; verify accepts a matching one."""
    d = attach_spec(scaler_file)
    assert d["spec_version"] == SPEC_VERSION
    assert d["spec_sha256"] == spec_sha256()
    assert d["gravity_alpha"] == 0.02
    got = verify_scaler(scaler_file)
    assert got["spec_version"] == SPEC_VERSION


def test_verify_refuses_unversioned(scaler_file: Path) -> None:
    """No spec fields => loud refusal, not a log line."""
    with pytest.raises(ValueError, match="NO spec fingerprint"):
        verify_scaler(scaler_file)


def test_verify_refuses_version_mismatch(scaler_file: Path) -> None:
    """A v1 scaler refuses pairing with a v2 expectation."""
    attach_spec(scaler_file, spec_version=SPEC_VERSION)
    d = json.loads(scaler_file.read_text())
    d["spec_version"] = 1
    scaler_file.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="spec_version 1 != expected"):
        verify_scaler(scaler_file)


def test_verify_refuses_hash_drift(scaler_file: Path) -> None:
    """Correct version but a stale hash => refusal (window path drifted)."""
    attach_spec(scaler_file)
    d = json.loads(scaler_file.read_text())
    d["spec_sha256"] = "0" * 64
    scaler_file.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="spec_sha256 mismatch"):
        verify_scaler(scaler_file)


def test_manifest_bind_and_verify(tmp_path: Path, scaler_file: Path) -> None:
    """Manifest binds model+scaler by hash; a post-export edit is caught."""
    attach_spec(scaler_file)
    model = tmp_path / "model.tflite"
    model.write_bytes(b"fake-model-bytes")
    man = write_model_manifest(tmp_path / "model_manifest.json", model, scaler_file)
    assert man["model_sha256"] == file_sha256(model)

    # tamper with the model -> verification refuses
    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="model_sha256 mismatch"):
        verify_model_manifest(tmp_path / "model_manifest.json", model, scaler_file)
