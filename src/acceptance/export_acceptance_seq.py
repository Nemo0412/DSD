"""Convert speculative_profiler details logs into acceptance_seq fields."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .replay import normalize_acceptance_seq


def accepted_flags_to_bits(flags: Sequence[Any]) -> List[int]:
    """Convert a list of bool / 0/1 flags into acceptance bits."""
    bits: List[int] = []
    for flag in flags:
        if isinstance(flag, bool):
            bits.append(1 if flag else 0)
        else:
            bits.append(1 if int(flag) else 0)
    return bits


def flatten_acceptance_seq(iterations: Sequence[Mapping[str, Any]]) -> Tuple[int, ...]:
    """Flatten per-iteration accepted_flags into a draft-token acceptance_seq."""
    bits: List[int] = []
    for iteration in iterations:
        flags = iteration.get("accepted_flags") or []
        bits.extend(accepted_flags_to_bits(flags))
    return normalize_acceptance_seq(bits)


def details_record_to_trace(
    record: Mapping[str, Any],
    *,
    request_id: Optional[str] = None,
    arrival_ms: float = 0.0,
    device_tier: str = "default",
    prompt_tokens: Optional[int] = None,
    target_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Map one speculative_profiler details JSON object to a workload trace row."""
    iterations = list(record.get("iterations") or [])
    acceptance_seq = flatten_acceptance_seq(iterations)

    if prompt_tokens is None:
        if iterations:
            prompt_tokens = int(iterations[0].get("context_length_before") or 0)
        else:
            prompt_tokens = 0
    prompt_tokens = max(1, int(prompt_tokens))

    if target_tokens is None:
        target_tokens = int(record.get("total_generated_tokens") or 0)
        if target_tokens <= 0:
            # Count committed accepts; reject bits are not committed draft tokens.
            target_tokens = sum(1 for bit in acceptance_seq if bit == 1)
    target_tokens = max(1, int(target_tokens))

    metadata = dict(record.get("metadata") or {})
    metadata["acceptance_source"] = "hardware_profile"
    if "prompt" in record:
        metadata.setdefault("prompt_preview", str(record["prompt"])[:120])
    if "prompt_index" in record:
        metadata.setdefault("prompt_index", record["prompt_index"])

    rid = request_id
    if rid is None:
        idx = record.get("prompt_index", 0)
        rid = f"profile_{idx:05d}"

    return {
        "request_id": rid,
        "arrival_ms": float(arrival_ms),
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
        "device_tier": device_tier,
        "acceptance_seq": list(acceptance_seq),
        "metadata": metadata,
    }


def merge_acceptance_into_workload(
    workload_row: Mapping[str, Any],
    details_record: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach a hardware acceptance_seq onto an existing workload JSONL row."""
    merged = dict(workload_row)
    acceptance_seq = flatten_acceptance_seq(list(details_record.get("iterations") or []))
    merged["acceptance_seq"] = list(acceptance_seq)

    metadata = dict(merged.get("metadata") or {})
    details_meta = dict(details_record.get("metadata") or {})
    metadata.update({k: v for k, v in details_meta.items() if k not in metadata})
    metadata["acceptance_source"] = "hardware_profile"
    if "prompt_index" in details_record:
        metadata["prompt_index"] = details_record["prompt_index"]
    merged["metadata"] = metadata

    # Prefer profiled generation length when workload target is missing/zero.
    if not merged.get("target_tokens"):
        generated = int(details_record.get("total_generated_tokens") or 0)
        if generated > 0:
            merged["target_tokens"] = generated
    return merged


def iter_details_records(lines: Iterable[str]) -> Iterable[Dict[str, Any]]:
    import json

    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        yield json.loads(text)
