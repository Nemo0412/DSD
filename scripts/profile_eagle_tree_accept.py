#!/usr/bin/env python3
"""Profile EAGLE tree speculative decoding and emit tree_accept JSONL logs.

Uses SafeAILab/EAGLE EaModel. Each verification round records:
  round, depth, candidates, verified, accepted_path_len, accepted_branch,
  draft_generation_ms, target_verification_ms
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

EAGLE_ROOT = Path(os.environ.get("EAGLE_ROOT", "/scratch/ll5914/DSD-SIM/eagle/EAGLE"))
sys.path.insert(0, str(EAGLE_ROOT))

from eagle.model.ea_model import EaModel  # noqa: E402
from eagle.model.kv_cache import initialize_past_key_values  # noqa: E402
from eagle.model.utils import (  # noqa: E402
    evaluate_posterior,
    initialize_tree,
    prepare_logits_processor,
    reset_tree_mode,
    tree_decoding,
    update_inference_inputs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen2-7B-Instruct")
    p.add_argument("--ea-model", default="yuhuili/EAGLE-Qwen2-7B-Instruct")
    p.add_argument("--prompts-file", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=float, default=0.0)
    p.add_argument("--total-token", type=int, default=60)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--tree-top-k", type=int, default=10)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=121)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--use-eagle3", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_prompts(path: Path, limit: Optional[int]) -> List[str]:
    prompts: List[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("prompt") or obj.get("question") or obj.get("text")
            if not text:
                raise ValueError(f"missing prompt field in {path}")
            prompts.append(str(text))
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


@torch.no_grad()
def profile_one(
    model: EaModel,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    top_k: float,
    max_new_tokens: int,
    max_length: int,
    max_prompt_tokens: int,
) -> Dict[str, Any]:
    tokenizer = model.get_tokenizer()
    # Chat-style wrap for Instruct models; fall back to raw prompt.
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    input_ids = encoded.input_ids.to(model.base_model.device)
    prompt_len = int(input_ids.shape[1])

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(
            temperature=temperature, top_p=top_p, top_k=top_k
        )
    else:
        logits_processor = None

    padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
    input_ids = input_ids.clone()
    model.ea_layer.reset_kv()

    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model, max_length=max_length)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    reset_tree_mode(model)
    tree_accept: List[Dict[str, Any]] = []
    new_token = 0
    hard_max = max_length - model.ea_layer.total_tokens - 10

    t0 = time.perf_counter()
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
        input_ids, model, past_key_values, logits_processor
    )
    draft_ms = (time.perf_counter() - t0) * 1000.0

    for round_idx in range(hard_max):
        model.base_model.model.tree_mask = tree_mask
        draft_tokens = draft_tokens.to(input_ids.device)

        torch.cuda.synchronize()
        tv0 = time.perf_counter()
        logits, hidden_state_new, outputs = tree_decoding(
            model,
            draft_tokens,
            past_key_values,
            tree_position_ids,
            input_ids,
            retrieve_indices,
        )
        torch.cuda.synchronize()
        verify_ms = (time.perf_counter() - tv0) * 1000.0

        draft_tokens = torch.cat((draft_tokens, padding), dim=1)
        candidates = draft_tokens[0, retrieve_indices]
        n_candidates = int(candidates.shape[0])
        depth = int(candidates.shape[1]) if candidates.ndim == 2 else 0
        verified = int(n_candidates * max(depth, 1))

        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits, candidates, logits_processor
        )
        accept_len = int(accept_length.item() if hasattr(accept_length, "item") else accept_length)
        best_idx = int(best_candidate.item() if hasattr(best_candidate, "item") else best_candidate)

        # Path indices along the accepted branch (0..accept_len-1 positions).
        # EAGLE stores cartesian candidates; branch id is best candidate index.
        accepted_branch = [best_idx] + [0] * max(accept_len - 1, 0)

        tree_accept.append(
            {
                "round": round_idx,
                "depth": depth,
                "candidates": n_candidates,
                "verified": verified,
                "accepted_path_len": accept_len,
                "accepted_branch": accepted_branch[: max(accept_len, 1)],
                "draft_generation_ms": round(draft_ms, 3),
                "target_verification_ms": round(verify_ms, 3),
            }
        )

        torch.cuda.synchronize()
        td0 = time.perf_counter()
        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            retrieve_indices,
            logits_processor,
            new_token,
            past_key_values_data,
            current_length_data,
            model,
            hidden_state_new,
            sample_p,
        )
        torch.cuda.synchronize()
        draft_ms = (time.perf_counter() - td0) * 1000.0

        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in input_ids[0, prompt_len:].tolist():
            break
        if new_token > max_new_tokens:
            break
        if input_ids.shape[1] > hard_max:
            break

    generated = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
    return {
        "prompt_tokens": prompt_len,
        "target_tokens": int(new_token),
        "generated_text": generated,
        "tree_accept": tree_accept,
        "num_rounds": len(tree_accept),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompts = load_prompts(args.prompts_file, args.limit)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"Loading base={args.base_model} ea={args.ea_model}", flush=True)
    # EAGLE-Qwen2 / Vicuna heads are EAGLE-1/2 (use_eagle3=False). EAGLE3-* weights need --use-eagle3.
    model = EaModel.from_pretrained(
        use_eagle3=bool(args.use_eagle3),
        base_model_path=args.base_model,
        ea_model_path=args.ea_model,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.tree_top_k,
        threshold=args.threshold,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    print("Model ready", flush=True)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as fout:
        for i, prompt in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] profiling...", flush=True)
            result = profile_one(
                model,
                prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
                max_prompt_tokens=args.max_prompt_tokens,
            )
            row = {
                "request_id": f"eagle_gsm8k_{i:04d}",
                "arrival_ms": float(i * 25.0),
                "prompt_tokens": result["prompt_tokens"],
                "target_tokens": max(1, result["target_tokens"]),
                "device_tier": "default",
                "tree_accept": result["tree_accept"],
                "metadata": {
                    "algorithm": "EAGLE",
                    "dataset": "GSM8K",
                    "dataset_split": "test_smoke_32",
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "base_model": args.base_model,
                    "ea_model": args.ea_model,
                    "total_token": args.total_token,
                    "depth": args.depth,
                    "tree_top_k": args.tree_top_k,
                    "max_new_tokens": args.max_new_tokens,
                    "use_eagle3": bool(args.use_eagle3),
                    "acceptance_source": "hardware_profile",
                    "num_rounds": result["num_rounds"],
                    "prompt_preview": prompt[:120],
                    "generated_preview": result["generated_text"][:120],
                    "paper_model_pair": "Vicuna/LLaMA2-Chat + EAGLE head",
                    "smoke_model_pair": f"{args.base_model} + {args.ea_model}",
                    "note": "Llama gated; Qwen2+EAGLE-Qwen2 smoke substitute",
                },
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            print(
                f"  tokens={row['target_tokens']} rounds={result['num_rounds']}",
                flush=True,
            )

    print(f"Wrote {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
