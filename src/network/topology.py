"""Generate network-aware latency estimates using NetworkX topologies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None  # type: ignore

from .fat_tree import FatTreeTopology


class NetworkModelError(RuntimeError):
    """Raised when a network topology specification is invalid."""


@dataclass(frozen=True)
class _Device:
    id: str
    role: str


def build_latency_lookup(
    *,
    drafts: Sequence[Mapping[str, object]],
    targets: Sequence[Mapping[str, object]],
    spec: Mapping[str, object],
) -> Dict[Tuple[str, str], float]:
    """Return draft→target latency (ms) using the configured network model.

    Parameters
    ----------
    drafts / targets: iterable of device dictionaries containing at least ``id``
        and ``role``.  For the ``fat_tree`` model each device may optionally
        carry a ``gpu_index`` field (int) that fixes its position in the
        fat-tree; devices without ``gpu_index`` are assigned indices
        sequentially (drafts first, then targets).
    spec: mapping describing the chosen topology. ``type`` selects which
        topology to use. Supported types:
        - ``clos`` / ``two_level_clos``: two-tier leaf/spine fabric.
        - ``complete``: constant-latency fully connected fabric.
        - ``fat_tree``: three-level fat-tree (NVLink intra-node, pod switch,
          root switch).  Optional ``config_path`` key points to a JSON file;
          omit to use the bundled ``fat_tree_config.json``.

    Returns
    -------
    dict[(draft_id, target_id)] -> latency_ms
    """

    model_type = str(spec.get("type", "clos")).lower()
    draft_devices = [_Device(id=str(d["id"]), role="draft") for d in drafts]
    target_devices = [_Device(id=str(t["id"]), role="target") for t in targets]

    if not draft_devices or not target_devices:
        raise NetworkModelError("network_model requires at least one draft and one target")

    if model_type in {"fat_tree", "fattree", "fat-tree"}:
        return _build_fat_tree(draft_devices, target_devices, drafts, targets, spec)

    # All non-fat-tree models require NetworkX.
    if nx is None:
        raise NetworkModelError(
            "networkx is required for network_model but is not installed. "
            "Install the 'networkx' package or disable the network model."
        )

    if model_type in {"clos", "two_level_clos", "leaf_spine"}:
        latency = _build_two_level_clos(draft_devices, target_devices, spec)
    elif model_type in {"complete", "fully_connected", "static"}:
        latency = _build_complete_graph(draft_devices, target_devices, spec)
    else:
        raise NetworkModelError(f"Unsupported network_model.type '{model_type}'")

    return latency


def _build_complete_graph(
    drafts: Sequence[_Device],
    targets: Sequence[_Device],
    spec: Mapping[str, object],
) -> Dict[Tuple[str, str], float]:
    base_latency = float(spec.get("latency_ms", 20.0))
    per_hop = float(spec.get("per_hop_ms", 0.0))

    graph = nx.Graph()
    for device in list(drafts) + list(targets):
        graph.add_node(device.id)

    for i in range(len(graph.nodes)):
        for j in range(i + 1, len(graph.nodes)):
            u = list(graph.nodes)[i]
            v = list(graph.nodes)[j]
            graph.add_edge(u, v, weight=base_latency + per_hop)

    return _pairwise_latencies(graph, drafts, targets)


def _build_two_level_clos(
    drafts: Sequence[_Device],
    targets: Sequence[_Device],
    spec: Mapping[str, object],
) -> Dict[Tuple[str, str], float]:
    hop_latency = float(spec.get("hop_latency_ms", spec.get("leaf_spine_latency_ms", 0.35)))
    edge_latency = float(spec.get("device_edge_latency_ms", spec.get("edge_latency_ms", 0.15)))
    spine_count = max(1, int(spec.get("spine_count", 4)))
    leaf_count = max(1, int(spec.get("leaf_count", max(2, math.ceil((len(drafts) + len(targets)) / 4)))))

    graph = nx.Graph()

    spines = [f"_spine_{i}" for i in range(spine_count)]
    leaves = [f"_leaf_{i}" for i in range(leaf_count)]

    for spine in spines:
        graph.add_node(spine)
    for leaf in leaves:
        graph.add_node(leaf)
        for spine in spines:
            graph.add_edge(leaf, spine, weight=hop_latency)

    # Round-robin placement keeps devices distributed across leaves.
    all_devices = list(targets) + list(drafts)
    assignment: MutableMapping[str, str] = {}
    for idx, device in enumerate(all_devices):
        leaf = leaves[idx % leaf_count]
        assignment[device.id] = leaf
        graph.add_node(device.id)
        graph.add_edge(device.id, leaf, weight=edge_latency)

    return _pairwise_latencies(graph, drafts, targets)


def _build_fat_tree(
    drafts: Sequence[_Device],
    targets: Sequence[_Device],
    draft_dicts: Sequence[Mapping[str, object]],
    target_dicts: Sequence[Mapping[str, object]],
    spec: Mapping[str, object],
) -> Dict[Tuple[str, str], float]:
    """Build draft→target latencies using a fat-tree topology.

    Device-to-GPU index assignment (priority order)
    ------------------------------------------------
    1. ``draft_gpu_indices`` / ``target_gpu_indices`` lists in ``spec``:
       explicit per-position GPU index overrides, useful for scripted
       benchmarks where GPU placement is generated programmatically.
    2. ``gpu_index`` field inside the individual device dict.
    3. Sequential assignment: drafts starting from 0, targets continuing
       from the next available index.
    """
    config_path = spec.get("config_path")
    topo = FatTreeTopology(config=config_path or spec)

    draft_idx_list = list(spec.get("draft_gpu_indices") or [])
    target_idx_list = list(spec.get("target_gpu_indices") or [])

    next_idx = 0
    draft_gpu: Dict[str, int] = {}
    for pos, (device, d) in enumerate(zip(drafts, draft_dicts)):
        if pos < len(draft_idx_list):
            idx = int(draft_idx_list[pos])
        elif d.get("gpu_index") is not None:
            idx = int(d["gpu_index"])
        else:
            idx = next_idx
            next_idx += 1
        draft_gpu[device.id] = idx

    target_gpu: Dict[str, int] = {}
    for pos, (device, t) in enumerate(zip(targets, target_dicts)):
        if pos < len(target_idx_list):
            idx = int(target_idx_list[pos])
        elif t.get("gpu_index") is not None:
            idx = int(t["gpu_index"])
        else:
            idx = next_idx
            next_idx += 1
        target_gpu[device.id] = idx

    latencies: Dict[Tuple[str, str], float] = {}
    for draft in drafts:
        for target in targets:
            gi = draft_gpu[draft.id]
            gj = target_gpu[target.id]
            latencies[(draft.id, target.id)] = topo.latency_ms(gi, gj)
    return latencies


def _pairwise_latencies(
    graph: "nx.Graph",
    drafts: Sequence[_Device],
    targets: Sequence[_Device],
) -> Dict[Tuple[str, str], float]:
    latencies: Dict[Tuple[str, str], float] = {}
    for draft in drafts:
        if draft.id not in graph:
            raise NetworkModelError(f"Draft '{draft.id}' missing from topology graph")
        for target in targets:
            if target.id not in graph:
                raise NetworkModelError(f"Target '{target.id}' missing from topology graph")
            try:
                dist = nx.shortest_path_length(graph, draft.id, target.id, weight="weight")
            except nx.NetworkXNoPath as exc:
                raise NetworkModelError(
                    f"No path between draft '{draft.id}' and target '{target.id}' in network_model"
                ) from exc
            latencies[(draft.id, target.id)] = float(dist)
    return latencies
