"""Acceptance-sequence normalization and replay cursor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


def normalize_acceptance_seq(raw: object) -> Tuple[int, ...]:
    """Convert a JSON-friendly acceptance sequence into a tuple of 0/1 ints."""
    if raw is None:
        return tuple()
    if isinstance(raw, (str, bytes)):
        raise TypeError("acceptance_seq must be a list/tuple of 0/1 values")
    if not isinstance(raw, Iterable):
        raise TypeError("acceptance_seq must be iterable")

    values: List[int] = []
    for item in raw:
        if isinstance(item, bool):
            values.append(1 if item else 0)
            continue
        try:
            number = int(item)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"invalid acceptance_seq entry: {item!r}") from exc
        if number not in (0, 1):
            raise ValueError(f"acceptance_seq entries must be 0 or 1, got {number}")
        values.append(number)
    return tuple(values)


@dataclass
class AcceptanceSeqCursor:
    """Consume a per-request acceptance sequence across speculation rounds.

    The sequence is interpreted in draft-token order. For a speculation window of
    size ``gamma``, we accept the longest leading run of ones (at most ``gamma``).
    If a zero appears inside the window, that reject bit is also consumed and the
    remaining speculative tokens are discarded, matching standard SD semantics.
    """

    sequence: Tuple[int, ...]
    index: int = 0

    @classmethod
    def from_optional(cls, raw: object) -> Optional["AcceptanceSeqCursor"]:
        seq = normalize_acceptance_seq(raw)
        if not seq:
            return None
        return cls(sequence=seq)

    def remaining(self) -> int:
        return max(0, len(self.sequence) - self.index)

    def has_data(self) -> bool:
        return self.remaining() > 0

    def verify(self, gamma: int) -> Tuple[int, int]:
        """Return ``(accepted_tokens, rejected_tokens)`` for one speculation round."""
        gamma = max(0, int(gamma))
        if gamma == 0:
            return 0, 0

        accepted = 0
        rejected = 0
        for _ in range(gamma):
            if self.index >= len(self.sequence):
                rejected = gamma - accepted
                break
            bit = self.sequence[self.index]
            self.index += 1
            if bit == 1:
                accepted += 1
            else:
                rejected = gamma - accepted
                break
        return accepted, rejected
