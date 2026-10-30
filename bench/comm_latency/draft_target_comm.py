#!/usr/bin/env python
"""Draft/target communication overhead, same definition as the A6000 runs.

One process is the draft server, one is the target. They speak the real ZMQ
speculate RPC (token ids, gamma=4). EchoDraftEngine keeps draft compute at
~0 so net_excess is the communication cost:

    net_excess_s = (t_recv - t_send - draft_compute) / 1000

Every round is written. Times are seconds.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GAMMA = 4
PROMPT_LEN = 128
# A typical accepted step is a few confirmed tokens, not a 1MB buffer.
ACCEPT_PER_STEP = 3


def serve(bind_host: str, advertise_host: str, ready_file: str, done_file: str) -> None:
    from dspec_real.draft.engine_echo import EchoDraftEngine
    from dspec_real.draft.server import DraftServer

    server = DraftServer(EchoDraftEngine("d0"), f"tcp://{bind_host}:0").start()
    endpoint = server.bound_addr  # tcp://host:port
    port = endpoint.rsplit(":", 1)[-1]
    advertised = f"tcp://{advertise_host}:{port}"
    with open(ready_file, "w") as f:
        f.write(advertised)
    print(f"draft listening {advertised}", flush=True)
    try:
        while not os.path.exists(done_file):
            time.sleep(0.2)
    finally:
        server.stop()
        print("draft stopped", flush=True)


def measure(addr: str, out_path: str, runs: int, steps: int) -> None:
    from dspec_real.proto.messages import SpeculateRequest
    from dspec_real.proto.transport import ZmqDraftClient

    client = ZmqDraftClient({"d0": addr}, timeout_s=30.0)
    new = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    out = open(out_path, "a", newline="")
    writer = csv.writer(out)
    if new:
        writer.writerow([
            "run", "step", "phase", "rtt_s", "net_excess_s", "draft_s",
            "gamma", "prompt_tokens", "accepted_tokens",
        ])
        out.flush()

    def one_round(run: int, step: int, phase: str, seq_id: str, confirmed: int,
                  prompt: list[int] | None, accepted: list[int]) -> int:
        req = SpeculateRequest(
            seq_id=seq_id,
            drafter_id="d0",
            gamma=GAMMA,
            num_confirmed=confirmed,
            accepted_tokens=accepted,
            prompt_token_ids=prompt,
        )
        reply = client.speculate([req])[0]
        rtt_s = reply.rtt_observed_ms / 1000.0
        net_s = reply.net_excess_ms / 1000.0
        draft_s = reply.draft_ms / 1000.0
        writer.writerow([
            run, step, phase, f"{rtt_s:.9f}", f"{net_s:.9f}", f"{draft_s:.9f}",
            GAMMA, 0 if prompt is None else len(prompt), len(accepted),
        ])
        out.flush()
        print(
            f"run={run} step={step} phase={phase} "
            f"rtt_s={rtt_s:.9f} net_excess_s={net_s:.9f}",
            flush=True,
        )
        return confirmed

    # Connection warmup. Still recorded, marked phase=warmup.
    confirmed = PROMPT_LEN
    one_round(0, 0, "warmup", "warmup", confirmed, list(range(PROMPT_LEN)), [])
    for step in range(1, 6):
        accepted = [1000 + step * 10 + i for i in range(ACCEPT_PER_STEP)]
        confirmed += len(accepted)
        one_round(0, step, "warmup", "warmup", confirmed, None, accepted)
    client.free(["warmup"])

    for run in range(1, runs + 1):
        seq = f"run{run}"
        confirmed = PROMPT_LEN
        one_round(run, 0, "prompt", seq, confirmed, list(range(PROMPT_LEN)), [])
        for step in range(1, steps):
            accepted = [run * 1000 + step * 10 + i for i in range(ACCEPT_PER_STEP)]
            confirmed += len(accepted)
            one_round(run, step, "decode", seq, confirmed, None, accepted)
        client.free([seq])

    client.close()
    out.close()
    print(f"wrote {out_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("server", "client"), required=True)
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--advertise-host", default="127.0.0.1")
    ap.add_argument("--ready-file", default="")
    ap.add_argument("--done-file", default="")
    ap.add_argument("--addr", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    if args.role == "server":
        serve(args.bind_host, args.advertise_host, args.ready_file, args.done_file)
    else:
        measure(args.addr, args.out, args.runs, args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
