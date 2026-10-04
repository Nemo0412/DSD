"""vLLM general plugin: install dsd-real's hooks inside every engine process.

vLLM loads `vllm.general_plugins` entry points in process0, the engine-core
process, and every worker process -- the only place code can run early
enough (and in the right process) for a monkeypatch to land before engine
construction.

`RemoteDraftProposer` itself does NOT need this. K6 (docs/stage0-findings.md)
confirmed the 0.27.1 pin has native `method="custom_class"` support
(vllm/v1/spec_decode/custom_class_proposer.py): vLLM imports and instantiates
the class named in `speculative_config.model` on its own. Monkeypatching
`GPUModelRunner.drafter` here would be actively wrong -- it would overwrite
the drafter unconditionally, regardless of `speculative_config.method`, and
break the `isinstance(self.drafter, NgramProposer)` assert vLLM's own
dispatch relies on for every non-custom_class run.

This module is therefore just the process-loading proof (register_probe,
K5) for now. It becomes the real injection point again once something needs
to run before engine construction in every process -- e.g. the L2 tap on
RejectionSampler.forward for full_window acceptance_seq capture (plan.md
Stage 2+).
"""
from __future__ import annotations

import json
import os


def register() -> None:
    """Entry point target. Must stay cheap and import-safe."""
    pass


def register_probe() -> None:
    """K5: record which processes actually loaded this plugin."""
    marker = os.environ.get("DSD_PLUGIN_MARKER")
    if not marker:
        return
    os.makedirs(os.path.dirname(marker) or ".", exist_ok=True)
    with open(marker, "a") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "ppid": os.getppid()}) + "\n")
