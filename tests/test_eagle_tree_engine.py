"""KV swap and tree tensor codec. No GPU and no EAGLE checkout."""
from __future__ import annotations

import torch

from dspec_real.draft.eagle_tree_codec import pack_hidden, pack_tree, unpack_hidden, unpack_tree
from dspec_real.draft.engine_eagle_tree import EagleTreeDraftEngine
from dspec_real.proto.messages import SpeculateReply, SpeculateRequest
from dspec_real.target.eagle_tree_remote import RemoteTopK


class _Head:
    def __init__(self) -> None:
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.stable_kv = None
        self.seen: list = []

    def topK_genrate(self, hidden, input_ids, head, logits_processor):
        self.seen.append(self.stable_kv)
        token = int(input_ids[0, -1])
        self.stable_kv = ("kv", token, self.stable_kv)
        n = 3
        draft = torch.tensor([[token, token + 1, token + 2]])
        retrieve = torch.tensor([[0, 1, -1], [0, 1, 2]])
        mask = torch.tril(torch.ones(1, 1, n, n))
        positions = torch.arange(n)
        return draft, retrieve, mask, positions


def _req(seq: str, token: int, hidden: torch.Tensor) -> SpeculateRequest:
    raw, shape, dim, dtype = pack_hidden(hidden)
    return SpeculateRequest(
        seq_id=seq, drafter_id="d0", gamma=0, token_ids=[token],
        hidden_states=raw, hidden_dim=dim, hidden_dtype=dtype, hidden_shape=shape,
    )


def test_kv_is_per_sequence():
    engine = EagleTreeDraftEngine(_Head(), node_id="d0")
    hidden = torch.randn(1, 2, 4, dtype=torch.float32)
    engine.speculate(_req("a", 1, hidden))
    engine.speculate(_req("b", 4, hidden))
    engine.speculate(_req("a", 2, hidden))
    # Third call is seq a again: the head must see a's kv, not b's.
    assert engine.head.seen[2] == ("kv", 1, None)
    assert engine.head.stable_kv[1] == 2
    engine.free(["a"])
    engine.speculate(_req("a", 3, hidden))
    assert engine.head.seen[-1] is None


def test_tree_roundtrip():
    hidden = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    raw, shape, dim, dtype = pack_hidden(hidden)
    back = unpack_hidden(raw, shape, dim, dtype)
    assert back.dtype == torch.bfloat16
    assert torch.equal(back.view(torch.int16), hidden.view(torch.int16))

    draft = torch.tensor([[7, 8, 9]])
    retrieve = torch.tensor([[0, 1, -1]])
    mask = torch.zeros(1, 1, 3, 3)
    positions = torch.tensor([0, 1, 2])
    fields = pack_tree(draft, retrieve, mask, positions)
    reply = SpeculateReply(seq_id="s", **fields)
    tokens, got_retrieve, got_mask, got_pos = unpack_tree(reply, "cpu")
    assert tokens.shape == (1, 3)
    assert int(got_retrieve[0, 2]) == -1
    assert got_mask.shape == (1, 1, 3, 3)
    assert got_pos.tolist() == [0, 1, 2]


class _Client:
    def __init__(self, engine: EagleTreeDraftEngine) -> None:
        self.engine = engine

    def speculate(self, requests):
        return self.engine.speculate_batch(requests)


def test_remote_topk_returns_tree():
    engine = EagleTreeDraftEngine(_Head(), node_id="d0")
    remote = RemoteTopK(_Client(engine), "d0", "seq")
    hidden = torch.randn(1, 2, 4)
    ids = torch.tensor([[1, 2]])
    tokens, retrieve, mask, positions = remote(hidden, ids, None, None)
    assert tokens.shape[0] == 1
    assert retrieve.shape[0] >= 1
    assert mask.shape[-1] == tokens.shape[-1]
    assert remote.last_reply is not None
    assert remote.last_reply.draft_ms >= 0
