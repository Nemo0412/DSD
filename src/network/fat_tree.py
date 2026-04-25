"""Fat-tree network topology with GPU-index-based latency computation.

Hierarchy
---------
Root switch
  └─ Pod switch  (pods_per_root pods)
       └─ Node  (nodes_per_pod nodes per pod)
            └─ GPU  (gpus_per_node GPUs per node, connected via NVLink)

GPU global index (0-based, row-major)
--------------------------------------
  gpu_index = pod_id  * nodes_per_pod * gpus_per_node
            + node_id_in_pod          * gpus_per_node
            + gpu_id_in_node

One-way latency rules
---------------------
  Same node      →  nvlink_ns
  Same pod       →  2 × node_to_pod_ns   (node → pod switch → node)
  Different pod  →  2 × node_to_pod_ns + 2 × pod_to_root_ns
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "fat_tree_config.json"


class FatTreeTopology:
    """Fat-tree topology that maps GPU indices to network latencies.

    Parameters
    ----------
    config:
        One of:
        - ``None``  – load from the default ``fat_tree_config.json`` next to this file.
        - ``str`` / ``Path`` – path to a JSON config file.
        - ``dict`` – inline config dict (same schema as the JSON file).
    """

    def __init__(self, config: Union[Dict, str, Path, None] = None) -> None:
        if config is None:
            cfg = _load_json(_DEFAULT_CONFIG_PATH)
        elif isinstance(config, (str, Path)):
            cfg = _load_json(Path(config))
        else:
            cfg = dict(config)

        self.gpus_per_node: int = int(cfg.get("gpus_per_node", 4))
        self.nodes_per_pod: int = int(cfg.get("nodes_per_pod", 4))
        self.pods_per_root: int = int(cfg.get("pods_per_root", 4))

        lat = cfg.get("latency", {})
        self.nvlink_ns: float = float(lat.get("nvlink_ns", 60.0))
        self.node_to_pod_ns: float = float(lat.get("node_to_pod_ns", 150.0))
        self.pod_to_root_ns: float = float(lat.get("pod_to_root_ns", 250.0))

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def gpus_per_pod(self) -> int:
        return self.nodes_per_pod * self.gpus_per_node

    @property
    def total_gpus(self) -> int:
        return self.pods_per_root * self.nodes_per_pod * self.gpus_per_node

    # ------------------------------------------------------------------
    # Index decomposition
    # ------------------------------------------------------------------

    def gpu_location(self, gpu_index: int) -> Tuple[int, int, int]:
        """Return ``(pod_id, node_id, local_gpu_id)`` for a global GPU index.

        ``node_id`` is the *global* node index across all pods.
        """
        pod_id = gpu_index // self.gpus_per_pod
        node_in_pod = (gpu_index % self.gpus_per_pod) // self.gpus_per_node
        node_id = pod_id * self.nodes_per_pod + node_in_pod
        local_gpu_id = gpu_index % self.gpus_per_node
        return pod_id, node_id, local_gpu_id

    # ------------------------------------------------------------------
    # Latency queries
    # ------------------------------------------------------------------

    def latency_ns(self, gpu_i: int, gpu_j: int) -> float:
        """One-way latency in **nanoseconds** between two GPUs by global index."""
        if gpu_i == gpu_j:
            return 0.0
        pod_i, node_i, _ = self.gpu_location(gpu_i)
        pod_j, node_j, _ = self.gpu_location(gpu_j)
        if node_i == node_j:
            # NVLink: direct GPU-to-GPU within same physical node.
            return self.nvlink_ns
        if pod_i == pod_j:
            # Both nodes attach to the same pod switch.
            return 2.0 * self.node_to_pod_ns
        # Cross-pod: traverse pod switch → root switch → pod switch.
        return 2.0 * self.node_to_pod_ns + 2.0 * self.pod_to_root_ns

    def latency_ms(self, gpu_i: int, gpu_j: int) -> float:
        """One-way latency in **milliseconds** between two GPUs by global index."""
        return self.latency_ns(gpu_i, gpu_j) / 1_000_000.0

    # ------------------------------------------------------------------
    # Lookup table helpers
    # ------------------------------------------------------------------

    def build_pairwise_ms(
        self,
        source_indices: Sequence[int],
        dest_indices: Sequence[int],
    ) -> Dict[Tuple[int, int], float]:
        """Return a ``{(src_gpu, dst_gpu): latency_ms}`` dict for all pairs."""
        return {
            (i, j): self.latency_ms(i, j)
            for i in source_indices
            for j in dest_indices
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe(self) -> str:
        lines = [
            "FatTreeTopology",
            f"  pods per root  : {self.pods_per_root}",
            f"  nodes per pod  : {self.nodes_per_pod}",
            f"  GPUs per node  : {self.gpus_per_node}",
            f"  total GPUs     : {self.total_gpus}",
            "",
            "  Link latencies (one-way):",
            f"    NVLink (same node)  : {self.nvlink_ns:.0f} ns",
            f"    node → pod switch   : {self.node_to_pod_ns:.0f} ns",
            f"    pod  → root switch  : {self.pod_to_root_ns:.0f} ns",
            "",
            "  Effective GPU-to-GPU one-way latencies:",
            f"    same node   : {self.nvlink_ns:.0f} ns"
            f"  ({self.nvlink_ns/1e6:.4f} ms)",
            f"    same pod    : {2*self.node_to_pod_ns:.0f} ns"
            f"  ({2*self.node_to_pod_ns/1e6:.4f} ms)",
            f"    cross-pod   : {2*self.node_to_pod_ns + 2*self.pod_to_root_ns:.0f} ns"
            f"  ({(2*self.node_to_pod_ns + 2*self.pod_to_root_ns)/1e6:.4f} ms)",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FatTreeTopology("
            f"pods={self.pods_per_root}, "
            f"nodes_per_pod={self.nodes_per_pod}, "
            f"gpus_per_node={self.gpus_per_node}, "
            f"total={self.total_gpus} GPUs)"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Dict:
    """Load a JSON config file; fall back to embedded defaults if absent."""
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    # Embedded fallback so the module works even without the JSON file.
    return {
        "gpus_per_node": 4,
        "nodes_per_pod": 4,
        "pods_per_root": 4,
        "latency": {"nvlink_ns": 60, "node_to_pod_ns": 150, "pod_to_root_ns": 250},
    }
