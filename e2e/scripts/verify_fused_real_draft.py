#!/usr/bin/env python
"""Stage 2a gate: real draft model + InProcessDraftClient (fused, no network).

Per context.md: "2a fused：真 draft 模型 + InProcessDraftClient（不碰网络）。
门禁 = 无损性，fp32 下 γ∈{1,2,4,8} 逐 token 相同。"

Two real vLLM engines share one process (VLLM_ENABLE_V1_MULTIPROCESSING=0,
same constraint as the custom_class configure() global): a target model
generating with RemoteDraftProposer wired to a VllmDraftEngine wrapping a
second, smaller model. fp32 + enforce_eager + greedy makes the whole pipeline
deterministic, so the gate is exact token equality against a non-speculative
fp32 baseline -- not a statistical closeness check.

    python scripts/verify_fused_real_draft.py
"""
from __future__ import annotations

import json
import os

TARGET_MODEL = os.environ.get("DSD_TARGET_MODEL", "Qwen/Qwen3-1.7B")
DRAFT_MODEL = os.environ.get("DSD_DRAFT_MODEL", "Qwen/Qwen3-0.6B")
TARGET_GPU_FRAC = float(os.environ.get("DSD_TARGET_GPU_FRAC", "0.45"))
DRAFT_GPU_FRAC = float(os.environ.get("DSD_DRAFT_GPU_FRAC", "0.20"))
MAX_LEN = 1024
MAX_TOKENS = 32
GAMMAS = (1, 2, 4, 8)

PROMPTS = [
    "The capital of France is",
    "In one sentence, explain why the sky is blue:",
    "List three prime numbers:",
    "def fibonacci(n):",
]


def _quiet_env() -> None:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    # RemoteDraftProposer.configure() sets a module-level global; target and
    # draft engines must both run in this process to see it and to share it.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _cleanup_gpu() -> None:
    # LLMEngine.__del__ only tears down its dp_group, never engine_core, so
    # plain `del` + gc leaves the previous engine's CUDA memory (and its NCCL
    # process group) reserved -- fine with one engine alive at a time, but
    # this script keeps a draft engine alive across every target rebuild, so
    # gpu_memory_utilization's headroom math needs it actually released.
    # Caller must `del` its own reference(s) first -- deleting a parameter
    # inside a helper only clears the helper's local binding, not the
    # caller's, and the object never actually gets collected.
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


def greedy(llm, prompts, max_tokens=MAX_TOKENS):
    from vllm import SamplingParams

    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False
    )
    return [list(o.outputs[0].token_ids) for o in outs]


def main() -> int:
    _quiet_env()
    from vllm import LLM

    common = dict(enforce_eager=True, dtype="float32", disable_log_stats=True,
                  max_model_len=MAX_LEN)

    # 1. fp32 baseline, speculation off.
    target = LLM(model=TARGET_MODEL, gpu_memory_utilization=TARGET_GPU_FRAC, **common)
    baseline = greedy(target, PROMPTS)
    del target
    _cleanup_gpu()

    # 2. One draft model, reused across every gamma below -- only the target
    #    needs rebuilding per gamma (num_speculative_tokens is fixed at
    #    construction). Sharing seq_ids never collides across gammas because
    #    RemoteDraftProposer mints a fresh one per (fresh vllm_config
    #    instantiation x slot), so stale VllmDraftEngine.state entries from a
    #    prior gamma are just harmless dead weight for this short-lived run.
    from dsdreal.draft.engine_vllm import VllmDraftEngine
    from dsdreal.proto.transport import InProcessDraftClient
    from dsdreal.target import proposer

    draft_llm = LLM(model=DRAFT_MODEL, gpu_memory_utilization=DRAFT_GPU_FRAC, **common)
    draft_engine = VllmDraftEngine(draft_llm, node_id="d0")
    proposer.configure(InProcessDraftClient(draft_engine, node_id="t0"))

    results = {}
    for gamma in GAMMAS:
        target = LLM(
            model=TARGET_MODEL, gpu_memory_utilization=TARGET_GPU_FRAC,
            speculative_config={
                "method": "custom_class",
                "model": "dsdreal.target.proposer.RemoteDraftProposer",
                "num_speculative_tokens": gamma,
            },
            **common,
        )
        replayed = greedy(target, PROMPTS)
        del target
        _cleanup_gpu()

        identical = replayed == baseline
        results[gamma] = {
            "identical_to_baseline": identical,
            "engine_calls": draft_engine.calls,
        }
        print(json.dumps({"gamma": gamma, **results[gamma]}))
        if not identical:
            for i, (b, r) in enumerate(zip(baseline, replayed)):
                if b != r:
                    div = next((j for j, (x, y) in enumerate(zip(b, r)) if x != y),
                               min(len(b), len(r)))
                    print(f"  prompt[{i}] diverged at token {div}: "
                          f"baseline={b} replayed={r}")

    proposer.configure(None)
    del draft_llm, draft_engine
    _cleanup_gpu()

    all_ok = all(r["identical_to_baseline"] for r in results.values())
    print("=" * 60)
    for gamma, r in results.items():
        print(f"  gamma={gamma}  {'PASS' if r['identical_to_baseline'] else 'FAIL'}")
    print("=" * 60)
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
