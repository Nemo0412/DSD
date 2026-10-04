#!/usr/bin/env python
"""Stage 2b gate: distributed (real ZMQ) -- TPOT must rise linearly with
injected RTT.

Per context.md: "2b 分布式：同样的模型 + ZMQ。接受率此时已知正确，所以这步只
测传输。门禁 = TPOT 随 RTT 线性上升，斜率 ≈ 1/L̄——不线性就说明 propose() 掉出
了关键路径（K2 的运行时复查）。"

2a already proved the pipeline is lossless; this script never checks output
correctness again, only timing. The target is built ONCE (num_speculative_
tokens is fixed at construction) and reused across every RTT point --
`ZmqDraftClient.emulator` is a plain mutable attribute, so the sweep is just
swapping it between generate() calls, no engine rebuild needed.

Run against a draft server already listening (see run_draft_server.py):

    python scripts/verify_distributed_rtt_sweep.py --draft-addr tcp://127.0.0.1:5560
"""
from __future__ import annotations

import argparse
import json
import os
import time

GAMMA = 4
MAX_LEN = 1024
MAX_TOKENS = 96
REPEATS = 3
RTTS_MS = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)

PROMPTS = [
    "The capital of France is",
    "In one sentence, explain why the sky is blue:",
    "List three prime numbers:",
    "def fibonacci(n):",
]


def _quiet_env() -> None:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    # RemoteDraftProposer.configure() sets a module-level global; the
    # target's OWN engine-core must run in this process to see it, whether
    # or not the drafter is a separate process (it is, here).
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def greedy(llm, prompts, max_tokens):
    from vllm import SamplingParams

    return llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False
    )


def one_tpot_sample_ms(llm, prompts, max_tokens):
    t0 = time.perf_counter()
    outs = greedy(llm, prompts, max_tokens=max_tokens)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    total_new_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    # Each request's first token comes from the prefill step, not a decode
    # step -- exclude one per request so this is decode-only time per token.
    decode_tokens = total_new_tokens - len(prompts)
    return wall_ms / max(decode_tokens, 1)


def median_tpot_ms(llm, prompts, max_tokens, repeats):
    # One untimed warmup call so lazy setup (first-call overhead, allocator
    # warmup) doesn't pollute the timed measurements, matching K2's method.
    greedy(llm, prompts, max_tokens=8)
    samples = [one_tpot_sample_ms(llm, prompts, max_tokens) for _ in range(repeats)]
    samples.sort()
    return samples[len(samples) // 2], samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-addr", required=True)
    ap.add_argument("--target-model", default=os.environ.get("DSD_TARGET_MODEL", "Qwen/Qwen3-1.7B"))
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()

    _quiet_env()
    from vllm import LLM

    from dsdreal.netem.sleep import SleepEmulator
    from dsdreal.proto.transport import ZmqDraftClient
    from dsdreal.target import proposer

    client = ZmqDraftClient({"d0": args.draft_addr})
    proposer.configure(client)

    target = LLM(
        model=args.target_model, gpu_memory_utilization=args.gpu_frac,
        enforce_eager=True, dtype="float32", disable_log_stats=True,
        max_model_len=MAX_LEN,
        speculative_config={
            "method": "custom_class",
            "model": "dsdreal.target.proposer.RemoteDraftProposer",
            "num_speculative_tokens": GAMMA,
        },
    )

    points = []
    for rtt in RTTS_MS:
        client.emulator = SleepEmulator(rtt_ms=rtt)
        tpot, samples = median_tpot_ms(target, PROMPTS, MAX_TOKENS, REPEATS)
        points.append((rtt, tpot))
        print(json.dumps({"rtt_ms": rtt, "tpot_ms_median": round(tpot, 4),
                           "samples_ms": [round(s, 4) for s in samples]}), flush=True)

    proposer.configure(None)
    client.close()

    import numpy as np

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Each engine round costs one round trip (rtt_ms) and yields ~L̄ tokens
    # PER ACTIVE REQUEST -- but all len(PROMPTS) requests are batched into
    # that same round trip (the fan-out client sends everything before
    # receiving anything), so one rtt_ms buys len(PROMPTS) * L̄ tokens of
    # output, not just L̄. dTPOT/d(rtt_ms) = 1/(len(PROMPTS) * L̄), which is
    # exactly the "max not sum" fan-out property this stage exists to prove
    # showing up in real timing data. L̄_eff has to land in [1, GAMMA+1]
    # (mean accepted length can't go below "1 bonus token, everything
    # rejected" or above "everything accepted").
    l_bar_eff = 1.0 / (slope * len(PROMPTS)) if slope > 0 else float("inf")
    result = {"points": points, "slope_ms_per_ms": round(float(slope), 4),
              "intercept_ms": round(float(intercept), 4), "r2": round(r2, 5),
              "l_bar_effective": round(l_bar_eff, 3)}
    print(json.dumps(result, indent=2))

    linear = r2 > 0.9 and slope > 0
    plausible_l_bar = 1.0 <= l_bar_eff <= GAMMA + 1
    ok = linear and plausible_l_bar
    if not ok:
        print(f"linear={linear} (r2={r2:.3f}) plausible_l_bar={plausible_l_bar} "
              f"(L̄_eff={l_bar_eff:.2f}, expected in [1, {GAMMA + 1}])")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
