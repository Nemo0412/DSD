"""Trace utilities."""

from .types import TraceRecord, TraceRecordDict, TraceParseError
from .trace_loader import iter_trace_records, load_trace
from .acceptance_seq import AcceptanceSeqCursor, normalize_acceptance_seq
from .synthetic_trace import (
    SyntheticTraceConfig,
    DeviceClassWeight,
    LengthDistribution,
    SyntheticTraceGenerator,
    build_device_mix_from_specs,
)

__all__ = [
    "TraceRecord",
    "TraceRecordDict",
    "TraceParseError",
    "AcceptanceSeqCursor",
    "normalize_acceptance_seq",
    "iter_trace_records",
    "load_trace",
    "SyntheticTraceConfig",
    "DeviceClassWeight",
    "LengthDistribution",
    "SyntheticTraceGenerator",
    "build_device_mix_from_specs",
]
