"""One EAGLE-tree target. Same router protocol as target_server.

The base model is local. Each request is pinned to one remote head, which
runs topK_genrate. Requests on this GPU run one at a time: the EAGLE tree
kernel is batch-1.
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import torch
import zmq

from dsdreal.metrics.records import RequestRecord, RoundRecord
from dsdreal.netem.null import NullEmulator
from dsdreal.proto.transport import make_client
from dsdreal.router.target_server import ADD, DEPTH, DONE, STATS, now_ms
from dsdreal.target.eagle_tree_remote import RemoteTopK, generate, load_eagle_base


class EagleTreeServer:
    def __init__(self, model, client, drafter_ids: list[str], bind: str) -> None:
        self.model = model
        self.client = client
        self.drafter_ids = drafter_ids
        self.records: dict[str, RequestRecord] = {}
        self.rounds: list[RoundRecord] = []
        self._rr = 0
        self._ctx = zmq.Context.instance()
        self.sock = self._ctx.socket(zmq.ROUTER)
        self.sock.bind(bind)
        self.bound_addr = self.sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        import msgspec
        self._enc = msgspec.msgpack.Encoder()
        self._dec = msgspec.msgpack.Decoder()
        self._peers: set[bytes] = set()
        self._adds = 0
        self._queue: list[str] = []

    def queue_depth(self) -> int:
        return sum(1 for r in self.records.values() if r.finish_ms is None)

    def _publish_depth(self) -> None:
        payload = self._enc.encode({"queue_depth": self.queue_depth(), "adds": self._adds})
        for ident in self._peers:
            self.sock.send_multipart([ident, DEPTH, payload])

    def _handle(self, block_ms: int) -> bool:
        done = False
        if not self.sock.poll(block_ms):
            return False
        while True:
            try:
                ident, kind, payload = self.sock.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return done
            msg = self._dec.decode(payload)
            self._peers.add(ident)
            if kind == ADD:
                self._adds += 1
                rec = RequestRecord(request_id=msg["request_id"], arrival_ms=msg["arrival_ms"],
                                    prompt_token_ids=list(msg["prompt_token_ids"]),
                                    max_tokens=msg["max_tokens"])
                self.records[rec.request_id] = rec
                self._queue.append(rec.request_id)
                self._publish_depth()
            elif kind == STATS:
                self.sock.send_multipart([ident, STATS, self._enc.encode(
                    {"queue_depth": self.queue_depth()})])
            elif kind == DONE:
                self._done_ident = ident
                done = True
        return done

    def _run_one(self, rec: RequestRecord) -> None:
        drafter = self.drafter_ids[self._rr % len(self.drafter_ids)]
        self._rr += 1
        remote = RemoteTopK(self.client, drafter, f"{rec.request_id}")
        ids = torch.tensor(rec.prompt_token_ids, dtype=torch.long)
        t_start = now_ms()
        produced, rounds = generate(self.model, remote, ids, rec.max_tokens)
        t_end = now_ms()
        cursor = 0
        for i, rd in enumerate(rounds):
            reply = rd["reply"]
            n_new = rd["accept_len"] + 1
            stamp = rd["verified_ms"]
            for _ in range(n_new):
                if cursor >= len(produced):
                    break
                rec.token_ts_ms.append(stamp)
                cursor += 1
            self.rounds.append(RoundRecord(
                seq_id=remote.seq_id,
                drafter_id=drafter,
                gamma=rd["n_draft"],
                n_draft=rd["n_draft"],
                num_confirmed=rd["num_confirmed"] if i else len(rec.prompt_token_ids) + 1,
                t_send_ms=reply.sent_ms if reply is not None else t_start,
                t_recv_ms=reply.recv_ms if reply is not None else stamp,
                draft_ms=reply.draft_ms if reply is not None else 0.0,
                injected_ms=0.0,
                prompt_token_ids=list(rec.prompt_token_ids) if i == 0 else None,
            ))
        while cursor < len(produced):
            rec.token_ts_ms.append(t_end)
            cursor += 1
        rec.output_token_ids = produced
        rec.finish_ms = t_end
        self.client.free([remote.seq_id])

    def serve(self, reuse: bool = False) -> None:
        import time
        while True:
            done = False
            while not done or self._queue:
                done = self._handle(0 if self._queue else 50) or done
                if self._queue:
                    rid = self._queue.pop(0)
                    self._run_one(self.records[rid])
                    self._publish_depth()
            self.sock.send_multipart([self._done_ident, DONE, self._enc.encode({
                "requests": [dataclasses.asdict(r) for r in self.records.values()],
                "rounds": [dataclasses.asdict(r) for r in self.rounds],
            })])
            if not reuse:
                break
            self.records.clear()
            self.rounds.clear()
            self._queue.clear()
            self._adds = 0
            self._peers.clear()
        time.sleep(0.2)
        self.sock.close(linger=1000)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--ea-model", required=True)
    ap.add_argument("--draft-addr", action="append", required=True)
    ap.add_argument("--bind", required=True)
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--total-token", type=int, default=60)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--reuse", action="store_true",
                    help="stay up after DONE so another dataset can use this engine")
    args = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from dsdreal.experiments.driver import _parse_addrs
    addresses = _parse_addrs(args.draft_addr)
    print(f"loading EAGLE base {args.base_model}", flush=True)
    model = load_eagle_base(args.base_model, args.ea_model, dtype=args.dtype,
                            total_tokens=args.total_token, depth=args.depth,
                            top_k=args.top_k, device=args.device)
    client = make_client("zmq", addresses=addresses, timeout_s=180.0, emulator=NullEmulator())
    server = EagleTreeServer(model, client, list(addresses), args.bind)
    if args.ready_file:
        with open(args.ready_file, "w") as fh:
            fh.write(server.bound_addr)
    print(f"eagle tree target on {server.bound_addr}", flush=True)
    server.serve(reuse=args.reuse)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
