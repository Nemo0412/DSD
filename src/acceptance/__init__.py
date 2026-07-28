"""Acceptance model helpers."""

from __future__ import annotations

from typing import Any

from .export_acceptance_seq import (
    details_record_to_trace,
    flatten_acceptance_seq,
    merge_acceptance_into_workload,
)
from .replay import AcceptanceSeqCursor, normalize_acceptance_seq

__all__ = [
    "AcceptanceRegressor",
    "AcceptanceSeqCursor",
    "normalize_acceptance_seq",
    "flatten_acceptance_seq",
    "details_record_to_trace",
    "merge_acceptance_into_workload",
]


def __getattr__(name: str) -> Any:
    if name == "AcceptanceRegressor":
        from .regressor import AcceptanceRegressor

        return AcceptanceRegressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
