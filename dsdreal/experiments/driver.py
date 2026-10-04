"""Online driver: submit requests at their arrival times, stamp every token.

In-process `LLMEngine` (VLLM_ENABLE_V1_MULTIPROCESSING=0) with our own
add_request/step() loop, not AsyncLLM: AsyncLLM always puts the engine core
in another process, where the module globals RemoteDraftProposer reads
(`proposer.configure(...)`: the client, the round sink) do not exist and the
clock is not ours. In-process, proposer and driver share one perf_counter.

Arrival is the *scheduled* time, not when the loop got around to admitting
it: a request that arrives while step() is running has been waiting since
its arrival, exactly as it would behind a real frontend.

    python -m dsdreal.experiments.driver --draft-addr d0=tcp://127.0.0.1:5560 \\
        --target-model meta-llama/Llama-3.1-8B-Instruct --gamma 4 --rtt-ms 10 \\
        --n-requests 100 --per-draft-rps 2 --out runs/vanilla_N1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dsdreal.metrics.records import RequestRecord
from dsdreal.workload.trace import WorkItem

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "dsdsim"


class Engine(Protocol):
    def add_request(self, request_id: str, prompt, params) -> None: ...
    def step(self) -> list: ...
    def has_unfinished_requests(self) -> bool: ...


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def run_open_loop(engine: Engine, items: list[WorkItem],
                  make_params: Callable[[WorkItem], object],
                  clock: Callable[[], float] = now_ms,
                  sleep: Callable[[float], None] = time.sleep,
                  uplink_ms: float = 0.0, downlink_ms: float = 0.0,
                  ) -> tuple[float, list[RequestRecord]]:
    """Returns (t0_ms, records). Outputs are read as CUMULATIVE token lists.

    uplink_ms / downlink_ms model dsdSim's scenario, where a request is born
    at its drafter's client: it is admitted uplink_ms after its (client-side)
    arrival, and its tokens are stamped downlink_ms after the target emitted
    them. Both are pure delays, as dsdSim's links are."""
    items = sorted(items, key=lambda w: w.arrival_ms)
    records: dict[str, RequestRecord] = {}
    t0 = clock()
    i = 0
    while i < len(items) or engine.has_unfinished_requests():
        now = clock()
        while i < len(items) and t0 + items[i].arrival_ms + uplink_ms <= now:
            w = items[i]
            records[w.request_id] = RequestRecord(
                request_id=w.request_id, arrival_ms=t0 + w.arrival_ms,
                prompt_token_ids=list(w.prompt_token_ids), max_tokens=w.max_tokens,
                uplink_ms=uplink_ms, downlink_ms=downlink_ms)
            engine.add_request(w.request_id, {"prompt_token_ids": list(w.prompt_token_ids)},
                               make_params(w))
            i += 1
        if engine.has_unfinished_requests():
            outs = engine.step()
            ts = clock() + downlink_ms
            for o in outs:
                rec = records[o.request_id]
                toks = o.outputs[0].token_ids
                n = len(toks)
                if n > rec.output_len:
                    rec.token_ts_ms.extend([ts] * (n - rec.output_len))
                    rec.output_token_ids = list(toks)
                if o.finished and rec.finish_ms is None:
                    rec.finish_ms = ts
        elif i < len(items):
            sleep(max(0.0, (t0 + items[i].arrival_ms + uplink_ms - clock()) / 1000.0))
    return t0, list(records.values())


# --------------------------------------------------------------------------- GPU entry


def _parse_addrs(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        drafter_id, addr = pair.split("=", 1)
        out[drafter_id] = addr
    return out


def add_target_args(ap: argparse.ArgumentParser) -> None:
    """Everything that defines one target engine and how it speculates."""
    ap.add_argument("--target-model", default=os.environ.get("DSD_TARGET_MODEL",
                                                             "meta-llama/Llama-3.1-8B-Instruct"))
    ap.add_argument("--draft-addr", action="append", default=[],
                    help="drafter_id=addr, repeat per drafter; none = no speculation (baseline)")
    ap.add_argument("--transport", choices=("zmq", "http"), default="zmq")
    ap.add_argument("--fused-draft-model", default=None,
                    help="run one drafter in-process (InProcessDraftClient) instead of --draft-addr")
    ap.add_argument("--fused-draft-gpu-frac", type=float, default=0.2)
    ap.add_argument("--draft-kind", choices=("lm", "eagle3"), default="lm",
                    help="eagle3: ship the target's aux hidden states to a remote EAGLE3 head")
    ap.add_argument("--native-eagle3", default=None, metavar="HEAD",
                    help="reference only: vLLM's own fused method=eagle3 with this head")
    ap.add_argument("--gamma", type=int, default=4,
                    help="static gamma; with --gamma-policy specpp, the upper bound")
    ap.add_argument("--gamma-policy", choices=("static", "specpp"), default="static")
    ap.add_argument("--acceptance-profile", default=None,
                    help="AcceptanceProfile JSON for specpp (scripts/build_acceptance_profile.py)")
    ap.add_argument("--specpp-min", type=int, default=2)
    ap.add_argument("--specpp-threshold", type=float, default=0.7)
    ap.add_argument("--rtt-ms", type=float, default=0.0)
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2,
                    help="requests run before t0 to absorb first-step JIT; not measured")
    ap.add_argument("--draft-timeout-s", type=float, default=30.0)
    ap.add_argument("--no-prefix-caching", action="store_true",
                    help="disable the target's prefix cache (always off with --draft-kind eagle3)")


def add_workload_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--trace", default=str(DATA_DIR / "gsm8k_trace_master.jsonl"))
    ap.add_argument("--prompts", default=str(DATA_DIR / "gsm8k_trace_source.jsonl"))
    ap.add_argument("--n-requests", type=int, default=100)
    ap.add_argument("--per-draft-rps", type=float, default=None,
                    help="Poisson load = per_draft_rps * N drafters; default: trace arrivals")
    ap.add_argument("--rate-rps", type=float, default=None, help="total Poisson load (overrides)")
    ap.add_argument("--max-output-tokens", type=int, default=None)
    ap.add_argument("--client-side", action="store_true",
                    help="dsdSim's scenario: requests are born at the drafters' clients, so the "
                         "prompt pays an uplink and every token a downlink of rtt/2 each")
    ap.add_argument("--chat-template", action="store_true",
                    help="wrap prompts in the target tokenizer's chat template (same length)")


def setup_env() -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # custom_class speculation runs on the V1 GPUModelRunner; keep the
    # no-spec baseline and the drafter on the same runner so comparisons are
    # like for like (V2 also needs UVA, which WSL lacks).
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class Target:
    engine: object
    client: object | None
    rounds: list
    make_params: Callable
    addresses: dict[str, str]
    n_drafters: int

    def close(self) -> None:
        from dsdreal.target import proposer
        proposer.configure(None)
        if self.client is not None:
            self.client.close()


def build_target(args, error: Callable[[str], None]) -> Target:
    """The target engine plus its proposer wiring. Must run in the process
    that will step the engine (module-global proposer config)."""
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    from dsdreal.netem.sleep import SleepEmulator
    from dsdreal.policy.routing import RoundRobin
    from dsdreal.proto.transport import make_client
    from dsdreal.target import proposer

    addresses = _parse_addrs(args.draft_addr)
    if args.fused_draft_model and addresses:
        error("--fused-draft-model and --draft-addr are exclusive")
    n_drafters = len(addresses) or (1 if args.fused_draft_model else 0)

    rounds: list = []
    client = None
    spec_cfg = None
    if args.draft_kind == "eagle3":
        from dsdreal.target import hidden_states
        hidden_states.install()
    emulator = SleepEmulator(rtt_ms=args.rtt_ms, jitter_ms=args.jitter_ms, seed=args.seed)
    if args.fused_draft_model:
        if args.draft_kind == "eagle3":
            import torch

            from dsdreal.draft.eagle3_model import load_eagle3_head
            from dsdreal.draft.engine_eagle3 import Eagle3DraftEngine
            draft_engine = Eagle3DraftEngine(load_eagle3_head(
                args.fused_draft_model, args.target_model, dtype=getattr(torch, args.dtype)))
        else:
            from vllm import LLM

            from dsdreal.draft.engine_vllm import VllmDraftEngine
            draft_llm = LLM(model=args.fused_draft_model, dtype=args.dtype, enforce_eager=True,
                            gpu_memory_utilization=args.fused_draft_gpu_frac,
                            max_model_len=args.max_model_len, disable_log_stats=True)
            draft_engine = VllmDraftEngine(draft_llm, node_id="d0")
        client = make_client("inprocess", engines={"d0": draft_engine}, emulator=emulator)
    elif addresses:
        client = make_client(args.transport, addresses=addresses,
                             timeout_s=args.draft_timeout_s, emulator=emulator)
    window = None
    if args.gamma_policy == "specpp":
        from dsdreal.policy.window import AcceptanceProfile, SpecPPGamma
        if not args.acceptance_profile:
            error("--gamma-policy specpp needs --acceptance-profile")
        window = SpecPPGamma(AcceptanceProfile.from_json(args.acceptance_profile),
                             min_gamma=args.specpp_min, max_gamma=args.gamma,
                             stop_threshold=args.specpp_threshold)
    if client is not None:
        proposer.configure(client, routing=RoundRobin(), rounds=rounds, window=window,
                           hidden_states=args.draft_kind == "eagle3")
        spec_cfg = {"method": "custom_class",
                    "model": "dsdreal.target.proposer.RemoteDraftProposer",
                    "num_speculative_tokens": args.gamma}
    if args.native_eagle3:
        if client is not None:
            error("--native-eagle3 runs vLLM's own drafter; drop --draft-addr/--fused-draft-model")
        spec_cfg = {"method": "eagle3", "model": args.native_eagle3,
                    "num_speculative_tokens": args.gamma}
    # A prefix-cache hit skips the forward over the cached positions, so their
    # aux hidden states are never produced and a remote EAGLE3 head would be
    # missing features for them (hidden_states.py refuses that combination).
    prefix_caching = not (args.no_prefix_caching or args.draft_kind == "eagle3")
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=args.target_model, dtype=args.dtype, enforce_eager=True,
        enable_prefix_caching=prefix_caching,
        gpu_memory_utilization=args.gpu_frac, max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs, disable_log_stats=not args.native_eagle3, seed=args.seed,
        speculative_config=spec_cfg,
    ))

    def make_params(w):
        return SamplingParams(temperature=0.0, max_tokens=w.max_tokens, ignore_eos=True,
                              output_kind=RequestOutputKind.CUMULATIVE)

    return Target(engine, client, rounds, make_params, addresses, n_drafters)


def warmup(target: Target, n: int) -> None:
    """Synthetic prompts: reusing real ones would pre-fill the prefix cache
    and flatter the first measured requests' TTFT."""
    if n:
        warm = [WorkItem(f"warmup-{i}", 0.0, [1000 + 7 * i + j for j in range(64)], 16)
                for i in range(n)]
        run_open_loop(target.engine, warm, target.make_params)
        target.rounds.clear()


def build_workload(args, n_drafters: int) -> tuple[list[WorkItem], float | None]:
    from transformers import AutoTokenizer

    from dsdreal.workload.trace import (build_requests, chat_wrap, load_prompt_texts,
                                        load_sim_trace, retime)

    entries = load_sim_trace(args.trace)[:args.n_requests]
    rate = args.rate_rps or (args.per_draft_rps * max(n_drafters, 1) if args.per_draft_rps else None)
    if rate:
        entries = retime(entries, rate, seed=args.seed)
    tok = AutoTokenizer.from_pretrained(args.target_model)
    items = build_requests(entries, load_prompt_texts(args.prompts),
                           lambda s: tok.encode(s, add_special_tokens=False),
                           bos_token_id=tok.bos_token_id,
                           max_output_tokens=args.max_output_tokens,
                           wrap=chat_wrap(tok) if args.chat_template else None)
    return items, rate


def write_outputs(out: str | Path, reqs) -> None:
    """Output tokens, for the lossless gate: same trace, spec vs no-draft
    baseline, fp32 -> must match token for token (scripts/compare_outputs.py)."""
    with open(Path(out) / "outputs.jsonl", "w") as f:
        for r in sorted(reqs, key=lambda r: r.request_id):
            f.write(json.dumps({"request_id": r.request_id,
                                "token_ids": r.output_token_ids}) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    add_target_args(ap)
    add_workload_args(ap)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    setup_env()

    from dsdreal.metrics.collector import collect
    from dsdreal.metrics.export import write_run

    target = build_target(args, ap.error)
    items, rate = build_workload(args, target.n_drafters)
    warmup(target, args.warmup)
    link = args.rtt_ms / 2.0 if args.client_side else 0.0
    t0, reqs = run_open_loop(target.engine, items, target.make_params,
                             uplink_ms=link, downlink_ms=link)
    per_request = collect(reqs, target.rounds)
    config = {k: v for k, v in vars(args).items()}
    config.update(n_drafters=target.n_drafters, rate_rps=rate, drafters=target.addresses)
    system = write_run(args.out, per_request, t0, config)
    write_outputs(args.out, reqs)
    if args.native_eagle3:
        # No rounds are visible to us on vLLM's own path; take its counters.
        native = {m.name: getattr(m, "value", None) for m in target.engine.get_metrics()
                  if "spec_decode" in m.name and hasattr(m, "value")}
        (Path(args.out) / "native_spec_metrics.json").write_text(json.dumps(native, indent=2))
        print(json.dumps(native, indent=2))
    print(json.dumps(system, indent=2, default=str))
    target.close()
    return 0 if system.get("accounting_ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
