"""`custom_class` speculative proposer: turns vLLM's propose() calls into
draft-client requests instead of running a draft model.

Registered via vLLM's native `method="custom_class"` plug point
(vllm/v1/spec_decode/custom_class_proposer.py, confirmed present in the
0.27.1 pin -- see docs/stage0-findings.md K6). vLLM imports the class named
in `speculative_config.model`, instantiates it with only `vllm_config`, and
calls `propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu,
slot_mappings=None)` once per engine step (gpu_model_runner.py, the
`method == "custom_class"` branch). No NgramProposer subclassing, no
isinstance dispatch assert, and no constructor warm-up call to defend
against -- those are quirks of the ngram dispatch path this one doesn't
share.

`num_tokens_no_spec[i]` is slot i's confirmed length: prompt plus every
accepted/bonus token so far, already including whatever was just sampled
this step (NgramProposer itself relies on this to search its own history for
matches). `token_ids_cpu[i, :num_tokens_no_spec[i]]` is that history.

`propose()` gets no request ids, and a row index is not an identity: the
input batch reuses rows as requests finish and moves requests between rows
(InputBatch.condense / swap_states). So identity is recovered from content:
each known sequence remembers its confirmed history, and a row belongs to the
sequence whose history is a prefix of the row's. The row's previous occupant
is tried first (the common case, one O(len) compare); only a miss scans the
other sequences; no match at all means a new request.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from dsdreal.metrics.records import RoundRecord
from dsdreal.policy.routing import RoutingPolicy
from dsdreal.policy.window import WindowPolicy
from dsdreal.proto.transport import DraftClient
from dsdreal.proto.messages import SpeculateRequest

_default_client: DraftClient | None = None
_stats_sink: list[int] | None = None
_default_routing: RoutingPolicy | None = None
_assignment_sink: list[str] | None = None
_round_sink: list[RoundRecord] | None = None
_default_window: WindowPolicy | None = None
_want_hidden = False


def configure(client: DraftClient | None, stats: list[int] | None = None,
              routing: RoutingPolicy | None = None,
              assignments: list[str] | None = None,
              rounds: list[RoundRecord] | None = None,
              window: WindowPolicy | None = None,
              hidden_states: bool = False) -> None:
    """Call before constructing the LLM.

    `custom_class` instantiates the proposer with only `vllm_config`
    (create_custom_proposer in vLLM), so a module-level default is the only
    way to hand it a transport. Pass None to drop the reference once done
    (e.g. before freeing the client's underlying engine/model) -- this
    module holding it alive would otherwise outlive `del` calls elsewhere.

    `stats`, if given, is a list this module appends to: one int per
    non-fresh `propose()` round, the number of tokens confirmed since the
    prior round for that slot (accepted draft tokens + the bonus token).
    That is the ground-truth per-round acceptance length -- for measuring it
    directly instead of inferring it from a timing slope (see
    scripts/verify_acceptance_length_parity.py). Pass None (the default) to
    disable collection.

    `routing`, if given, picks which of `client.drafter_ids` a *fresh*
    request is assigned to (see dsdreal/policy/routing.py). Pass None (the
    default) to keep the single-drafter behavior: every request goes to
    `drafter_id` as passed to RemoteDraftProposer's constructor (or its
    default, "d0").

    `assignments`, if given, is a list this module appends to: the chosen
    drafter_id every time `routing` assigns one to a fresh request. Only
    populated when `routing` is set -- for measuring the actual fan-out
    distribution (see scripts/verify_ndrafter_fanout.py) without having to
    scrape per-drafter server logs.

    `rounds`, if given, is a list this module appends one RoundRecord to per
    sequence per `speculate()` call -- the raw material for
    dsdreal/metrics/collector.py (per-round timing, gamma, and, via the next
    round's num_confirmed, how many were accepted).

    `window`, if given, picks each request's gamma once, on its fresh round
    (dsdreal/policy/window.py -- the dsdSim GammaPolicy contract), capped at
    `num_speculative_tokens`. None keeps a fixed gamma = num_speculative_tokens.

    `hidden_states=True` makes the proposer ask the target for its EAGLE3 aux
    hidden states and ship them with every request (remote EAGLE3 head).
    Needs dsdreal.target.hidden_states.install() before engine construction.
    """
    global _default_client, _stats_sink, _default_routing, _assignment_sink
    global _round_sink, _default_window, _want_hidden
    _default_client = client
    _stats_sink = stats
    _default_routing = routing
    _assignment_sink = assignments
    _round_sink = rounds
    _default_window = window
    _want_hidden = hidden_states


@dataclass
class _Seq:
    seq_id: str
    drafter_id: str
    hist: np.ndarray        # confirmed history as of the last round
    gamma: int
    last_seen: int          # propose() call count
    n_hidden: int = 0       # positions whose hidden states were shipped


class RemoteDraftProposer:
    # A sequence unseen for this many propose() calls is taken as finished and
    # its drafter-side state freed (vLLM does not tell the proposer).
    STALE_AFTER_CALLS = 2000

    def __init__(self, vllm_config, client: DraftClient | None = None,
                 drafter_id: str = "d0",
                 routing: RoutingPolicy | None = None) -> None:
        spec_config = vllm_config.speculative_config
        self.k = spec_config.num_speculative_tokens
        self.drafter_id = drafter_id
        self.routing = routing if routing is not None else _default_routing
        self.window = _default_window
        # Read by dsdreal/target/hidden_states.py's runner patches.
        self.wants_aux_hidden_states = _want_hidden
        self._pending_hidden: dict[str, list] = {}   # req_id -> [tensor, ...]
        self._row_req: dict[int, str] = {}
        self.client = client if client is not None else _default_client
        if self.client is None:
            raise RuntimeError(
                "RemoteDraftProposer needs a DraftClient; call "
                "dsdreal.target.proposer.configure(client) before "
                "constructing the LLM."
            )
        # A drafter (VllmDraftEngine, a real server) can outlive any single
        # RemoteDraftProposer -- reused across engine restarts, or serving
        # more than one target. Instance-unique seq_ids keep a stale entry
        # left in the drafter's state by an earlier instance from colliding.
        self._instance_id = uuid.uuid4().hex[:8]
        self._seqs: dict[str, _Seq] = {}
        self._row_seq: dict[int, str] = {}
        self._next_seq = 0
        self._calls = 0

    def _owns(self, seq: _Seq, row: np.ndarray, n_hist: int) -> bool:
        n = len(seq.hist)
        return n <= n_hist and np.array_equal(row[:n], seq.hist)

    def _match(self, i: int, row: np.ndarray, n_hist: int, claimed: set[str]) -> _Seq | None:
        prev = self._seqs.get(self._row_seq.get(i, ""))
        if prev is not None and prev.seq_id not in claimed and self._owns(prev, row, n_hist):
            return prev
        best = None
        for seq in self._seqs.values():
            if seq.seq_id in claimed or seq is prev:
                continue
            if self._owns(seq, row, n_hist) and (best is None or len(seq.hist) > len(best.hist)):
                best = seq
        return best

    def receive_hidden_states(self, rows: dict) -> None:
        """Called by the runner patch right before propose(), every step:
        {row: (req_id, tensor [n, 3 * hidden])} for rows with new positions.
        Buffered per req_id, because a chunked prefill delivers positions on
        steps where propose() has nothing to send for that row yet."""
        self._row_req = {}
        for row, (rid, t) in rows.items():
            self._row_req[row] = rid
            self._pending_hidden.setdefault(rid, []).append(t)

    def _take_hidden(self, row: int, seq: "_Seq", n_hist: int) -> tuple[bytes, int, str]:
        import torch

        rid = self._row_req.get(row)
        parts = self._pending_hidden.pop(rid, []) if rid is not None else []
        feats = torch.cat(parts, dim=0) if parts else None
        n_new = 0 if feats is None else feats.shape[0]
        seq.n_hidden += n_new
        if seq.n_hidden != n_hist - 1:  # K8 live check
            raise RuntimeError(
                f"hidden-state misalignment on {seq.seq_id} (req {rid}): shipped "
                f"{seq.n_hidden} positions, confirmed length {n_hist} needs {n_hist - 1}")
        if feats is None:
            return b"", 0, ""
        dtype = str(feats.dtype).removeprefix("torch.")
        raw = feats.contiguous().view(torch.uint8) if feats.dtype == torch.bfloat16 else feats.contiguous()
        return raw.numpy().tobytes(), feats.shape[1], dtype

    def _gc(self) -> None:
        stale = [sid for sid, s in self._seqs.items()
                 if self._calls - s.last_seen > self.STALE_AFTER_CALLS]
        if stale:
            for sid in stale:
                del self._seqs[sid]
            self.client.free(stale)

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu,
                slot_mappings=None) -> list[list[int]]:
        self._calls += 1
        requests: list[SpeculateRequest] = []
        active_rows: list[int] = []
        claimed: set[str] = set()
        for i, sampled in enumerate(sampled_token_ids):
            if not sampled:
                continue
            n_hist = int(num_tokens_no_spec[i])
            row = np.asarray(token_ids_cpu[i])
            seq = self._match(i, row, n_hist, claimed)
            if seq is None:
                seq_id = f"{self._instance_id}-seq{self._next_seq}"
                self._next_seq += 1
                if self.routing is not None:
                    drafter_id = self.routing.select_drafter(self.client.drafter_ids)
                    if _assignment_sink is not None:
                        _assignment_sink.append(drafter_id)
                else:
                    drafter_id = self.drafter_id
                # Context = prompt: the fresh round's history is the prompt
                # plus the one token sampled by prefill.
                gamma = (min(self.k, max(1, self.window.select_gamma(n_hist - 1)))
                         if self.window is not None else self.k)
                seq = _Seq(seq_id, drafter_id, row[:0].copy(), gamma, self._calls)
                self._seqs[seq_id] = seq
                prompt_token_ids = [int(t) for t in row[:n_hist]]
                accepted_tokens: list[int] = []
            else:
                prompt_token_ids = None
                # Delta since our last call -- a stateful drafter (anything
                # that isn't purely positional like ReplayDraftEngine) needs
                # this to keep its own confirmed length in lockstep with
                # num_confirmed. Omitting it is exactly the bug that let
                # VllmDraftEngine/EchoDraftEngine desync while
                # ReplayDraftEngine (num_confirmed-only) stayed silent.
                accepted_tokens = [int(t) for t in row[len(seq.hist):n_hist]]
                if _stats_sink is not None:
                    _stats_sink.append(len(accepted_tokens))
            claimed.add(seq.seq_id)
            seq.hist = row[:n_hist].copy()
            seq.last_seen = self._calls
            self._row_seq[i] = seq.seq_id

            hidden, hidden_dim, hidden_dtype = (
                self._take_hidden(i, seq, n_hist) if self.wants_aux_hidden_states
                else (None, 0, ""))
            requests.append(SpeculateRequest(
                seq_id=seq.seq_id,
                drafter_id=seq.drafter_id,
                gamma=seq.gamma,
                num_confirmed=n_hist,
                accepted_tokens=accepted_tokens,
                prompt_token_ids=prompt_token_ids,
                hidden_states=hidden,
                hidden_dim=hidden_dim,
                hidden_dtype=hidden_dtype,
            ))
            active_rows.append(i)

        if self._calls % 256 == 0:
            self._gc()

        out: list[list[int]] = [[] for _ in sampled_token_ids]
        if not requests:
            return out

        replies = self.client.speculate(requests)
        for row, reply in zip(active_rows, replies):
            out[row] = list(reply.draft_tokens)
        if _round_sink is not None:
            for req, reply in zip(requests, replies):
                _round_sink.append(RoundRecord(
                    seq_id=req.seq_id,
                    drafter_id=req.drafter_id,
                    gamma=req.gamma,
                    n_draft=len(reply.draft_tokens),
                    num_confirmed=req.num_confirmed,
                    t_send_ms=reply.sent_ms,
                    t_recv_ms=reply.recv_ms,
                    draft_ms=reply.draft_ms,
                    injected_ms=reply.injected_delay_ms,
                    prompt_token_ids=req.prompt_token_ids,
                ))
        return out
