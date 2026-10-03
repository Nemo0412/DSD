"""Workload: dsdSim traces in, token-exact requests out.

dsdSim's traces (traces/gsm8k_trace_*.jsonl) carry only lengths --
`prompt_tokens`, `target_tokens` -- plus `arrival_ms`. To replay one on a
real engine each request needs real prompt tokens of exactly that length and
an output of exactly `target_tokens` (run with ignore_eos, since the
simulator's conversation length is `target_tokens` regardless of content).

Prompts are cut from the simulator's own GSM8K prompt source
(aws-trn/prompts/gsm8k_trace_source.jsonl, vendored under data/dsdsim/):
token streams of consecutive questions, starting at a different question per
request so identical-length requests still get distinct prompts.
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TraceEntry:
    request_id: str
    arrival_ms: float
    prompt_tokens: int
    target_tokens: int


@dataclass(frozen=True, slots=True)
class WorkItem:
    request_id: str
    arrival_ms: float          # relative to run start
    prompt_token_ids: list[int]
    max_tokens: int


def load_sim_trace(path: str | Path) -> list[TraceEntry]:
    out = []
    for line in Path(path).read_text().split("\n"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(TraceEntry(str(d["request_id"]), float(d["arrival_ms"]),
                              int(d["prompt_tokens"]), int(d["target_tokens"])))
    out.sort(key=lambda e: e.arrival_ms)
    return out


def load_prompt_texts(path: str | Path) -> list[str]:
    return [json.loads(line)["prompt"] for line in Path(path).read_text().split("\n")
            if line.strip()]


def poisson_arrivals(n: int, rate_rps: float, seed: int = 0) -> list[float]:
    """Open-loop arrival times (ms), first at 0."""
    if rate_rps <= 0:
        raise ValueError("rate_rps must be positive")
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(rate_rps) * 1000.0
    return out


def retime(entries: list[TraceEntry], rate_rps: float, seed: int = 0) -> list[TraceEntry]:
    """Keep the trace's lengths, replace its arrivals with Poisson at `rate_rps`.

    dsdSim sets load as `per_draft_rps`, so callers pass per_draft_rps * N.
    """
    times = poisson_arrivals(len(entries), rate_rps, seed)
    return [replace(e, arrival_ms=t) for e, t in zip(entries, times)]


def build_requests(entries: list[TraceEntry], texts: list[str],
                   tokenize: Callable[[str], list[int]],
                   bos_token_id: int | None = None,
                   max_prompt_tokens: int | None = None,
                   max_output_tokens: int | None = None,
                   wrap: tuple[list[int], list[int]] | None = None,
                   min_body: int = 16) -> list[WorkItem]:
    """Token-exact prompts: `prompt_tokens` long including BOS if given.

    `wrap=(prefix, suffix)` puts the body inside fixed token spans -- a chat
    template's user-turn header and assistant-turn opener (see
    `chat_wrap`) -- still exactly `prompt_tokens` long overall. Instruct
    models, and EAGLE heads trained on chat data above all, draft far worse
    on bare text.

    Prompts are always pairwise distinct. Identical prompts give identical
    greedy histories, which RemoteDraftProposer (it recognises a request by
    its token history) cannot tell apart -- rounds of one get booked to the
    other. With `wrap`, the body keeps at least `min_body` tokens even if the
    trace asked for less (a 35-token chat header would otherwise leave 1).
    """
    if not texts:
        raise ValueError("no prompt texts")
    pool = [tokenize(t) for t in texts]
    items = []
    seen: set[tuple[int, ...]] = set()
    for i, e in enumerate(entries):
        want = e.prompt_tokens if max_prompt_tokens is None else min(e.prompt_tokens, max_prompt_tokens)
        prefix = [bos_token_id] if bos_token_id is not None else []
        suffix: list[int] = []
        if wrap is not None:
            prefix, suffix = list(wrap[0]), list(wrap[1])
        body_len = max(want - len(prefix) - len(suffix), min_body if wrap is not None else 1)
        for start in range(i, i + len(pool)):
            body: list[int] = []
            j = start
            while len(body) < body_len:
                body.extend(pool[j % len(pool)])
                j += 1
                if j - start > len(pool) and not body:
                    raise ValueError("prompt texts tokenize to nothing")
            prompt = prefix + body[:body_len] + suffix
            if tuple(prompt) not in seen:
                break
        else:
            raise ValueError(f"cannot make a distinct {len(prompt)}-token prompt for {e.request_id}")
        seen.add(tuple(prompt))
        max_tokens = e.target_tokens if max_output_tokens is None else min(e.target_tokens, max_output_tokens)
        items.append(WorkItem(e.request_id, e.arrival_ms, prompt, max_tokens))
    return items


def chat_wrap(tokenizer) -> tuple[list[int], list[int]]:
    """(prefix, suffix) token spans of the tokenizer's chat template around a
    single user message, generation prompt included. Found by rendering a
    sentinel message and splitting on it.

    A trace prompt shorter than prefix + suffix + 1 cannot stay exact and
    comes out longer; the exported prompt_len is always the real one."""
    sentinel = "\u2063SENTINEL\u2063"
    # Llama 3.x templates stamp "Today Date"; pin it so prompts do not change
    # from one day's run to the next. Templates without the variable ignore it.
    text = tokenizer.apply_chat_template([{"role": "user", "content": sentinel}],
                                         tokenize=False, add_generation_prompt=True,
                                         date_string="26 Jul 2024")
    pre, post = text.split(sentinel)
    enc = lambda t: tokenizer.encode(t, add_special_tokens=False)  # noqa: E731
    return enc(pre), enc(post)
