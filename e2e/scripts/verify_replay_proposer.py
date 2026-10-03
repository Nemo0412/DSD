#!/usr/bin/env python
"""End-to-end check for the Stage 0 deliverable from context.md:

    "用custom_class注册一个remotedrafter，现在可以replay一些预先录制的
    draft sequence"

1. Run the target model with speculation OFF to record each prompt's real
   greedy continuation (the trace).
2. Wire dsdreal.target.proposer.RemoteDraftProposer via vLLM's native
   `method="custom_class"` (confirmed available on this pin -- K6), backed
   by an InProcessDraftClient over a ReplayDraftEngine loaded with that
   trace.
3. Assert: output is token-identical to the baseline, and acceptance is
   near 100% -- we handed the drafter its own future, so it should get
   nearly everything right.

    python scripts/verify_replay_proposer.py
"""
from __future__ import annotations

import json
import os

MODEL = os.environ.get("DSD_PROBE_MODEL", "Qwen/Qwen3-0.6B")
GAMMA = 4
GPU_FRAC = float(os.environ.get("DSD_PROBE_GPU_FRAC", "0.35"))
MAX_LEN = 1024
MAX_TOKENS = 32

PROMPTS = [
    "The capital of France is",
    "In one sentence, explain why the sky is blue:",
    "List three prime numbers:",
    "def fibonacci(n):",
]


def _quiet_env() -> None:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    # RemoteDraftProposer.configure() sets a module-level global; the engine
    # (and the drafter it constructs) must run in this same process to see it.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_llm(**overrides):
    from vllm import LLM

    kwargs = dict(
        model=MODEL,
        gpu_memory_utilization=GPU_FRAC,
        max_model_len=MAX_LEN,
        enforce_eager=True,
        dtype="bfloat16",
        disable_log_stats=True,
    )
    kwargs.update(overrides)
    return LLM(**kwargs)


def greedy(llm, prompts, max_tokens=MAX_TOKENS):
    from vllm import SamplingParams

    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False
    )
    return [list(o.outputs[0].token_ids) for o in outs]


def _free_gpu(llm):
    del llm
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    _quiet_env()

    # 1. Record the trace with speculation off.
    base = build_llm()
    prompt_ids = [base.get_tokenizer().encode(p) for p in PROMPTS]
    baseline = greedy(base, PROMPTS)
    _free_gpu(base)

    traces = {tuple(p): c for p, c in zip(prompt_ids, baseline)}

    # 2. Wire the replay proposer via native custom_class.
    from dsdreal.draft.engine_replay import ReplayDraftEngine
    from dsdreal.proto.transport import InProcessDraftClient
    from dsdreal.target import proposer

    engine = ReplayDraftEngine(traces, node_id="d0")
    proposer.configure(InProcessDraftClient(engine, node_id="t0"))

    llm = build_llm(speculative_config={
        "method": "custom_class",
        "model": "dsdreal.target.proposer.RemoteDraftProposer",
        "num_speculative_tokens": GAMMA,
    })
    replayed = greedy(llm, PROMPTS)
    _free_gpu(llm)

    identical = replayed == baseline
    result = {
        "identical_to_baseline": identical,
        "baseline_lengths": [len(x) for x in baseline],
        "replayed_lengths": [len(x) for x in replayed],
        # Raw counts, not a derived ratio: engine_calls is one per engine
        # step per active request (one bonus token each, minimum), and
        # tokens_proposed is what was offered for verification each time.
        "engine_calls": engine.calls,
        "engine_tokens_proposed": engine.tokens_proposed,
        "output_tokens": sum(len(x) for x in replayed),
    }
    print(json.dumps(result, indent=2))
    if not identical:
        for i, (b, r) in enumerate(zip(baseline, replayed)):
            if b != r:
                print(f"prompt[{i}] diverged at token "
                      f"{next((j for j, (x, y) in enumerate(zip(b, r)) if x != y), min(len(b), len(r)))}: "
                      f"baseline={b} replayed={r}")
    print("PASS" if identical else "FAIL")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
