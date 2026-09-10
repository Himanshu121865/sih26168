"""Core primitives: signal, scaler, spec, logging, training, contracts."""

from python.core.scaler import TrainOnlyScaler
from python.core.spec import SPEC_VERSION, attach_spec, spec_sha256, verify_scaler

__all__ = ["SPEC_VERSION", "TrainOnlyScaler", "attach_spec", "spec_sha256", "verify_scaler"]
