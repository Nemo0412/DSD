"""Tree-level acceptance replay for tree speculative decoding (EAGLE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TreeRound:
    """One measured verification round of a candidate tree.

    ``accepted`` is the accepted draft path length, excluding the bonus /
    correction token the target appends. ``tree_nodes`` is the number of
    candidate nodes the target scored, or 0 when the log does not say.
    """

    accepted: int
    depth: int
    tree_nodes: int


def _round_from_entry(entry: Mapping[str, Any]) -> TreeRound:
    if "accepted_path_len" not in entry:
        raise ValueError("tree_accept entry needs accepted_path_len")
    accepted = int(entry["accepted_path_len"])
    if accepted < 0:
        raise ValueError(f"accepted_path_len must be >= 0, got {accepted}")
    depth = max(0, int(entry.get("depth", 0) or 0))
    # tree_nodes is the forward width. Older logs only carry verified or
    # candidates; profile_eagle_tree_accept.py's `verified` counts
    # paths x depth, so it over-states the width when tree_nodes is present.
    nodes = 0
    for key in ("tree_nodes", "verified", "candidates"):
        value = entry.get(key)
        if value is not None and int(value) > 0:
            nodes = int(value)
            break
    return TreeRound(accepted=accepted, depth=depth, tree_nodes=nodes)


@dataclass
class TreeAcceptCursor:
    """Consume a request's ``tree_accept`` log one verification round at a time."""

    rounds: Tuple[TreeRound, ...]
    index: int = 0

    @classmethod
    def from_optional(cls, raw: Optional[Sequence[Mapping[str, Any]]]) -> Optional["TreeAcceptCursor"]:
        if not raw:
            return None
        ordered = sorted(enumerate(raw), key=lambda item: (int(item[1].get("round", item[0])), item[0]))
        return cls(rounds=tuple(_round_from_entry(entry) for _, entry in ordered))

    def remaining(self) -> int:
        return max(0, len(self.rounds) - self.index)

    def has_data(self) -> bool:
        return self.remaining() > 0

    def next_round(self) -> TreeRound:
        if not self.has_data():
            raise IndexError("tree_accept replay exhausted")
        rnd = self.rounds[self.index]
        self.index += 1
        return rnd
