"""Remote EAGLE3 drafter: the chain-drafting loop of vLLM's EagleProposer,
run against target hidden states that arrive over the wire.

Per sequence the drafter keeps:
- `tokens`: the confirmed token history (length n = num_confirmed);
- `feats`:  fc'd target aux hidden states for positions 0 .. n-2 (the newest
  token's hidden state only exists after the target's next step);
- a KV cache whose first `m` entries are "confirmed": position t's entry was
  computed from the pair (token t+1, target feature t), both confirmed.
  Entries past `m` came from last round's chained drafting (draft tokens,
  the head's own hidden states).

Each round (context.md's "kv 回退 + reprefill"):
1. adopt the target's accepted tokens and the new positions' hidden states;
2. roll back: truncate the KV to `m` -- the drafted tail is stale whether or
   not the target accepted it, because it was computed from the head's own
   hidden states, not the target's;
3. re-prefill positions m .. n-2 from (token t+1, feature t); now m = n-1;
   the last position's logits give draft 1;
4. chain: draft k+1 from (draft k, previous step's prenorm hidden) at the
   next position, gamma - 1 times.

All sequences in a request batch go through each of 3 and 4 as one padded
forward. Greedy (argmax over the draft vocab, mapped to target ids via d2t).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from dsdreal.draft.eagle3_model import Eagle3Head, StepInput
from dsdreal.proto.messages import SpeculateReply, SpeculateRequest

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def decode_hidden(raw: bytes, dim: int, dtype: str) -> torch.Tensor:
    if not raw:
        return torch.empty(0, dim)
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return t.view(_DTYPES[dtype]).view(-1, dim)


@dataclass
class _Seq:
    tokens: list[int] = field(default_factory=list)
    feats: torch.Tensor | None = None          # [n-1, H] on device
    k: torch.Tensor | None = None              # [n_kv, len, d]
    v: torch.Tensor | None = None
    m: int = 0                                 # confirmed KV length


class Eagle3DraftEngine:
    def __init__(self, head: Eagle3Head, node_id: str = "d0") -> None:
        self.node_id = node_id
        self.head = head
        self.state: dict[str, _Seq] = {}
        self.calls = 0
        self.dev = head.fc.weight.device
        self.dt = head.fc.weight.dtype

    def _empty_kv(self):
        return torch.empty(self.head.nkv, 0, self.head.hd, dtype=self.dt, device=self.dev)

    def _advance(self, r: SpeculateRequest) -> _Seq:
        seq = self.state.get(r.seq_id)
        if seq is None:
            seq = _Seq(tokens=list(r.prompt_token_ids or []), k=self._empty_kv(), v=self._empty_kv(),
                       feats=torch.empty(0, self.head.config.hidden_size, dtype=self.dt, device=self.dev))
            self.state[r.seq_id] = seq
        seq.tokens.extend(r.accepted_tokens)
        if r.num_confirmed and len(seq.tokens) != r.num_confirmed:
            raise ValueError(f"drafter desync on {r.seq_id}: local {len(seq.tokens)} "
                             f"!= target {r.num_confirmed}")
        raw = decode_hidden(r.hidden_states or b"", r.hidden_dim, r.hidden_dtype or "bfloat16")
        if raw.shape[0]:
            seq.feats = torch.cat([seq.feats, self.head.combine(raw.to(self.dev))])
        n = len(seq.tokens)
        if seq.feats.shape[0] != n - 1:
            raise ValueError(f"hidden-state count on {r.seq_id}: have {seq.feats.shape[0]} "
                             f"features, need {n - 1} for {n} confirmed tokens")
        return seq

    def speculate(self, request: SpeculateRequest) -> SpeculateReply:
        (res,) = self.speculate_batch([request])
        if isinstance(res, Exception):
            raise res
        return res

    @torch.no_grad()
    def speculate_batch(self, requests: list[SpeculateRequest]) -> list[SpeculateReply | Exception]:
        t0 = time.perf_counter()
        self.calls += len(requests)
        results: list[SpeculateReply | Exception | None] = [None] * len(requests)
        live: list[tuple[int, _Seq, int]] = []
        for i, r in enumerate(requests):
            try:
                seq = self._advance(r)
            except Exception as exc:  # noqa: BLE001 - this sequence's error only
                results[i] = exc
                continue
            if r.gamma <= 0:
                results[i] = SpeculateReply(seq_id=r.seq_id, draft_tokens=[])
                continue
            live.append((i, seq, r.gamma))

        if live:
            drafts = self._draft([(s, g) for _, s, g in live])
            ms = (time.perf_counter() - t0) * 1000.0
            for (i, _, _), toks in zip(live, drafts):
                results[i] = SpeculateReply(seq_id=requests[i].seq_id, draft_tokens=toks, draft_ms=ms)
        return results  # type: ignore[return-value]

    def _draft(self, batch: list[tuple[_Seq, int]]) -> list[list[int]]:
        head, dev = self.head, self.dev
        # 2 + 3: roll back to the confirmed prefix, re-prefill new positions.
        items = []
        for seq, _ in batch:
            seq.k, seq.v = seq.k[:, :seq.m], seq.v[:, :seq.m]
            n = len(seq.tokens)
            t = torch.arange(seq.m, n - 1, device=dev)
            items.append(StepInput(
                token_ids=torch.tensor(seq.tokens[seq.m + 1:n], device=dev),
                hidden=seq.feats[seq.m:n - 1], positions=t, past_k=seq.k, past_v=seq.v))
        out, pre, nk, nv = head.forward_batch(items)
        last_out, last_pre = [], []
        for j, (seq, _) in enumerate(batch):
            q = len(items[j].token_ids)
            seq.k = torch.cat([seq.k, nk[j]], dim=1)
            seq.v = torch.cat([seq.v, nv[j]], dim=1)
            seq.m = len(seq.tokens) - 1
            last_out.append(out[j, q - 1])
            last_pre.append(pre[j, q - 1])
        cur = head.to_target_ids(head.logits(torch.stack(last_out)).argmax(-1))
        drafts = [[int(x)] for x in cur.tolist()]
        hidden = torch.stack(last_pre)

        # 4: chain the remaining gamma - 1 drafts, only for sequences that want them.
        for step in range(1, max(g for _, g in batch)):
            idx = [j for j, (_, g) in enumerate(batch) if g > step]
            if not idx:
                break
            items = [StepInput(token_ids=cur[j:j + 1], hidden=hidden[j:j + 1],
                               positions=torch.tensor([batch[j][0].m + step - 1], device=dev),
                               past_k=batch[j][0].k, past_v=batch[j][0].v) for j in idx]
            out, pre, nk, nv = head.forward_batch(items)
            nxt = head.to_target_ids(head.logits(out[:, 0]).argmax(-1))
            for row, j in enumerate(idx):
                seq = batch[j][0]
                seq.k = torch.cat([seq.k, nk[row]], dim=1)
                seq.v = torch.cat([seq.v, nv[row]], dim=1)
                drafts[j].append(int(nxt[row]))
                cur[j] = nxt[row]
                hidden[j] = pre[row, 0]
        return drafts

    def free(self, seq_ids: list[str]) -> None:
        for sid in seq_ids:
            self.state.pop(sid, None)

    def close(self) -> None:
        self.state.clear()
