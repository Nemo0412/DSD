#!/usr/bin/env python
"""K8: can a custom_class proposer get the target's EAGLE3 aux hidden states,
aligned to the right positions?

Runs the target with dsdreal.target.hidden_states installed and a capturing
in-process drafter, under a small max_num_batched_tokens so prompts are
chunk-prefilled (the buffered path). Two checks:

1. alignment bookkeeping: RemoteDraftProposer's live check (shipped positions
   == num_confirmed - 1 at every round) never fires;
2. the values are the right positions: per sequence, concatenate everything
   shipped and compare position by position with a HF reference forward
   over the final sequence (`output_hidden_states`, layers
   get_eagle3_default_aux_hidden_state_layers = (2, L//2, L-3); vLLM's aux
   for layer k is layer k's input = HF hidden_states[k]). Mean cosine must be
   high, and clearly higher than the same comparison shifted by one
   position -- a one-off misalignment is exactly what this has to catch.

    python scripts/verify_aux_hidden_states.py --target-model Qwen/Qwen3-1.7B
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", default=os.environ.get("DSD_TARGET_MODEL", "Qwen/Qwen3-1.7B"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-frac", type=float, default=0.5)
    ap.add_argument("--n-requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--batched-tokens", type=int, default=96,
                    help="small, to force chunked prefill")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--min-cos", type=float, default=0.98)
    args = ap.parse_args()

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import EngineArgs, LLMEngine, SamplingParams

    from dsdreal.draft.engine_echo import EchoDraftEngine
    from dsdreal.experiments.driver import DATA_DIR, run_open_loop
    from dsdreal.proto.transport import InProcessDraftClient
    from dsdreal.target import hidden_states, proposer
    from dsdreal.workload.trace import TraceEntry, build_requests, load_prompt_texts

    captured: dict[str, list] = {}

    class Capture(EchoDraftEngine):
        def speculate(self, request):
            captured.setdefault(request.seq_id, []).append(
                (request.num_confirmed, request.prompt_token_ids, request.hidden_states,
                 request.hidden_dim, request.hidden_dtype))
            return super().speculate(request)

    hidden_states.install()
    proposer.configure(InProcessDraftClient(Capture("d0")), hidden_states=True)

    tok = AutoTokenizer.from_pretrained(args.target_model)
    lens = [150 + 23 * i for i in range(args.n_requests)]  # > batched-tokens: chunked
    entries = [TraceEntry(f"r{i}", 20.0 * i, n, args.max_tokens) for i, n in enumerate(lens)]
    items = build_requests(entries, load_prompt_texts(DATA_DIR / "gsm8k_trace_source.jsonl"),
                           lambda s: tok.encode(s, add_special_tokens=False),
                           bos_token_id=tok.bos_token_id)

    engine = LLMEngine.from_engine_args(EngineArgs(
        model=args.target_model, dtype=args.dtype, enforce_eager=True,
        gpu_memory_utilization=args.gpu_frac, max_model_len=1024,
        max_num_batched_tokens=args.batched_tokens, max_num_seqs=8,
        enable_chunked_prefill=True, enable_prefix_caching=False, disable_log_stats=True,
        speculative_config={"method": "custom_class",
                            "model": "dsdreal.target.proposer.RemoteDraftProposer",
                            "num_speculative_tokens": args.gamma}))
    runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    m = runner.get_model()
    layers = tuple(getattr(m, "aux_hidden_state_layers", None)
                   or getattr(m.model, "aux_hidden_state_layers", ()))
    assert layers, "aux hidden state layers were never set up on the target"
    _, reqs = run_open_loop(engine, items, lambda w: SamplingParams(
        temperature=0.0, max_tokens=w.max_tokens, ignore_eos=True))
    del engine, runner, m
    proposer.configure(None)
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    cleanup_dist_env_and_memory()

    # seq -> full sequence + shipped features
    by_prompt = {tuple(r.prompt_token_ids): r for r in reqs}
    hf = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=getattr(torch, args.dtype)).cuda().eval()
    results = []
    for seq_id, rounds in captured.items():
        first = rounds[0][1]
        req = next((r for p, r in by_prompt.items() if tuple(first[:len(p)]) == p), None)
        if req is None:
            continue
        parts = []
        for _, _, raw, dim, dt in rounds:
            if raw:
                t = torch.frombuffer(bytearray(raw), dtype=torch.uint8 if dt == "bfloat16" else getattr(torch, dt))
                parts.append((t.view(torch.bfloat16) if dt == "bfloat16" else t).view(-1, dim).float())
        shipped = torch.cat(parts)
        n = rounds[-1][0] - 1
        assert shipped.shape[0] == n, (seq_id, shipped.shape, n)
        full = req.prompt_token_ids + req.output_token_ids
        with torch.no_grad():
            out = hf(torch.tensor([full[:n]], device="cuda"), output_hidden_states=True)
        ref = torch.cat([out.hidden_states[k][0] for k in layers], dim=-1).float().cpu()
        cos = torch.nn.functional.cosine_similarity(shipped, ref, dim=-1)
        shifted = torch.nn.functional.cosine_similarity(shipped[1:], ref[:-1], dim=-1)
        results.append({"seq": seq_id, "positions": n, "chunks": len(parts),
                        "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
                        "cos_shifted_mean": float(shifted.mean())})

    ok = bool(results) and all(r["cos_mean"] >= args.min_cos and
                               r["cos_mean"] - r["cos_shifted_mean"] > 0.02 for r in results)
    print(json.dumps({"aux_layers": layers, "sequences": len(results),
                      "cos_mean_min": min((r["cos_mean"] for r in results), default=None),
                      "cos_shifted_mean_max": max((r["cos_shifted_mean"] for r in results), default=None),
                      "per_seq": results[:4]}, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
