#!/usr/bin/env python
"""Is a lossless-gate mismatch a bug or a numerical near-tie?

Rebuilds the request's prompt (same workload args as the runs), takes the
common prefix up to the first divergent token, runs one fp32 HF forward, and
prints the target's top-5 next-token logits and the gap between the two
tokens the runs chose. A gap at the level of fp32 reduction-order noise
(~1e-3 relative or less) means batch-dependent numerics flipped an argmax,
not that speculation changed the output distribution.

    python scripts/inspect_divergence.py runs/stepA/fp32_base runs/stepA/fp32_spec gsm8k_00016 \\
        --device cpu -- --n-requests 40 --max-output-tokens 128 --per-draft-rps 2 --chat-template
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    wl = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("request_id")
    ap.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM

    from dsdreal.experiments.driver import add_workload_args, build_workload

    wp = argparse.ArgumentParser()
    add_workload_args(wp)
    wp.add_argument("--target-model", default=args.target_model)
    wp.add_argument("--seed", type=int, default=0)
    wargs, _ = wp.parse_known_args(wl + ["--target-model", args.target_model])
    items, _ = build_workload(wargs, 1)
    prompt = next(w.prompt_token_ids for w in items if w.request_id == args.request_id)

    def out(run):
        for line in (Path(run) / "outputs.jsonl").read_text().split("\n"):
            if line.strip():
                d = json.loads(line)
                if d["request_id"] == args.request_id:
                    return d["token_ids"]

    a, b = out(args.run_a), out(args.run_b)
    k = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float32).to(args.device).eval()
    with torch.no_grad():
        logits = model(torch.tensor([prompt + a[:k]], device=args.device)).logits[0, -1].float().cpu()
    top = torch.topk(logits, 5)
    gap = float(logits[a[k]] - logits[b[k]])
    print(json.dumps({
        "request_id": args.request_id, "divergence_index": k,
        "run_a_token": a[k], "run_b_token": b[k],
        "logit_a": float(logits[a[k]]), "logit_b": float(logits[b[k]]),
        "gap_a_minus_b": gap, "relative_gap": abs(gap) / float(logits[a[k]].abs()),
        "top5": [[int(i), float(v)] for v, i in zip(top.values, top.indices)],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
