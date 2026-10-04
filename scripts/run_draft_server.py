#!/usr/bin/env python
"""Stage 2b: draft-side process. Runs a real vLLM model behind a socket, so
the target reaches it over the network instead of an in-process call -- the
whole point of 2b versus 2a.

Two wires, same engine and same observable behaviour (--transport):
  zmq   DEALER/ROUTER + msgpack  (default; what the Stage 2b GPU runs used)
  http  HTTP/JSON                (inspectable with curl, easy to sniff)

For N-drafter fan-out, launch one of these per drafter with a distinct
--drafter-id and --bind, then point the matching client at all of them keyed
by that same id: {"d0": addr0, "d1": addr1, ...}.

    python scripts/run_draft_server.py --bind tcp://0.0.0.0:5560 \\
        --drafter-id d0 --ready-file runs/draft_ready
    python scripts/run_draft_server.py --transport http \\
        --bind http://0.0.0.0:5560 --drafter-id d0

--ready-file receives the bound address in whichever form that transport's
client expects, so a launcher script does not need to know which is in use.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys


def _vllm_engine(args):
    print("draft server: constructing LLM...", flush=True)
    from vllm import LLM

    from dsdreal.draft.engine_vllm import VllmDraftEngine

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              enforce_eager=True, dtype=args.dtype, disable_log_stats=True,
              max_model_len=args.max_model_len)
    # First use of a shape/path JIT-compiles kernels (seconds, occasionally
    # far more): do it before reporting ready, so the target's first rounds
    # do not time out on it. Several batch sizes, then the same prompts again
    # extended by a few tokens -- the prefix-cache-hit path every later
    # round takes.
    from vllm import SamplingParams
    warm = SamplingParams(temperature=0.0, max_tokens=8)
    for bs in (1, 8, 32):
        prompts = [[1000 + 13 * i + j for j in range(96 + i)] for i in range(bs)]
        llm.generate([{"prompt_token_ids": p} for p in prompts], warm, use_tqdm=False)
        llm.generate([{"prompt_token_ids": p + [7, 8, 9]} for p in prompts], warm, use_tqdm=False)
    print("draft server: LLM constructed, wiring VllmDraftEngine...", flush=True)
    engine = VllmDraftEngine(llm, node_id=args.drafter_id)

    return engine


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DSD_DRAFT_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--transport", choices=("zmq", "http"), default="zmq")
    ap.add_argument("--bind", default=None,
                     help="zmq: tcp://host:port (0 = ephemeral). "
                          "http: http://host:port. "
                          "Defaults to <transport>://0.0.0.0:5560")
    ap.add_argument("--drafter-id", default="d0",
                     help="must match the key the client's address dict "
                          "uses for this process")
    ap.add_argument("--ready-file", default=None,
                     help="written with the bound address once serving starts")
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--dtype", default="float32",
                     help="float32 for the lossless gates; bfloat16 for timing runs")
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--kind", choices=("vllm", "eagle3"), default="vllm",
                     help="vllm: an independent draft LM (--model); eagle3: a remote EAGLE3 "
                          "head (--model = head checkpoint) fed the target's hidden states")
    ap.add_argument("--eagle-target-model", default=os.environ.get("DSD_TARGET_MODEL"),
                     help="eagle3: target checkpoint whose embed_tokens the head shares")
    args = ap.parse_args()
    if args.bind is None:
        args.bind = ("tcp://0.0.0.0:5560" if args.transport == "zmq"
                     else "http://0.0.0.0:5560")

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # This process has its own dedicated GPU and no module-global config to
    # share with a parent -- unlike the target side, multiprocessing here
    # would be fine, but there is nothing to gain from it for one small model.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # custom_class speculation runs on the V1 GPUModelRunner; keep the
    # no-spec baseline and the drafter on the same runner so comparisons are
    # like for like (V2 also needs UVA, which WSL lacks).
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    # `kill -USR1 <pid>` dumps every thread's stack: the way to see where a
    # drafter is stuck when the target only reports "no reply".
    import faulthandler
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    if args.kind == "eagle3":
        import torch

        from dsdreal.draft.eagle3_model import load_eagle3_head
        from dsdreal.draft.engine_eagle3 import Eagle3DraftEngine
        if not args.eagle_target_model:
            ap.error("--kind eagle3 needs --eagle-target-model (or DSD_TARGET_MODEL)")
        print("draft server: loading EAGLE3 head...", flush=True)
        head = load_eagle3_head(args.model, args.eagle_target_model,
                                dtype=getattr(torch, args.dtype))
        engine = Eagle3DraftEngine(head, node_id=args.drafter_id)
    else:
        engine = _vllm_engine(args)
    print(f"draft server: binding {args.transport} to {args.bind}...", flush=True)
    if args.transport == "zmq":
        from dsdreal.draft.server import DraftServer

        server = DraftServer(engine, args.bind)
        address = server.bound_addr
    else:
        from urllib.parse import urlparse

        from dsdreal.draft.http_server import HttpDraftServer

        parsed = urlparse(args.bind)
        server = HttpDraftServer(engine, host=parsed.hostname or "0.0.0.0",
                                 port=parsed.port or 0)
        address = server.base_url
    print(f"draft server: bound to {address}, starting serve thread...", flush=True)
    server.start()
    print(f"draft server serving on {address}", flush=True)
    if args.ready_file:
        with open(args.ready_file, "w") as fh:
            fh.write(address)

    stop = {"flag": False}

    def _handle(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    while not stop["flag"]:
        signal.pause()

    print(f"draft server stopping; served {engine.calls} requests", flush=True)
    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
