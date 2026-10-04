"""Remote EAGLE3 drafter on CPU with a tiny random head (fp32).

The property that matters: after any sequence of rounds with rejections,
the drafter's state must be exactly what it would be had it seen the final
confirmed history in one go -- i.e. rollback + re-prefill leave no trace of
drafted-then-rejected tokens. Checked by comparing drafts against a fresh
engine given the whole history at once.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from dsdreal.draft.eagle3_model import Eagle3Head, StepInput  # noqa: E402
from dsdreal.draft.engine_eagle3 import Eagle3DraftEngine  # noqa: E402
from dsdreal.proto.messages import SpeculateRequest  # noqa: E402

HT = 48  # target hidden size


def _head(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                                   intermediate_size=128, vocab_size=100, num_hidden_layers=1,
                                   rms_norm_eps=1e-6, max_position_embeddings=512)
    cfg.draft_vocab_size = 50
    h = Eagle3Head(cfg, target_hidden_size=HT)
    for p in h.parameters():
        torch.nn.init.normal_(p, std=0.2)
    h.d2t.copy_(torch.arange(50) % 7)  # draft id d -> target id d + d2t[d]
    return h.float().eval()


def _raw(t):
    return t.float().contiguous().numpy().tobytes()


def _req(seq, n_conf, prompt=None, accepted=(), feats=None, gamma=4):
    return SpeculateRequest(seq_id=seq, drafter_id="d0", gamma=gamma, num_confirmed=n_conf,
                            accepted_tokens=list(accepted), prompt_token_ids=prompt,
                            hidden_states=_raw(feats) if feats is not None else None,
                            hidden_dim=3 * HT, hidden_dtype="float32")


def test_chunked_prefill_matches_one_shot():
    h = _head()
    toks = torch.randint(0, 100, (12,))
    hid = torch.randn(12, 64)
    empty = torch.empty(h.nkv, 0, h.hd)
    one, _, _, _ = h.forward_batch([StepInput(toks, hid, torch.arange(12), empty, empty)])
    a, _, ka, va = h.forward_batch([StepInput(toks[:5], hid[:5], torch.arange(5), empty, empty)])
    b, _, _, _ = h.forward_batch([StepInput(toks[5:], hid[5:], torch.arange(5, 12), ka[0], va[0])])
    assert torch.allclose(torch.cat([a[0, :5], b[0, :7]]), one[0], atol=1e-5)


def test_padded_batch_matches_single():
    h = _head()
    empty = torch.empty(h.nkv, 0, h.hd)
    x1 = StepInput(torch.randint(0, 100, (3,)), torch.randn(3, 64), torch.arange(3), empty, empty)
    x2 = StepInput(torch.randint(0, 100, (7,)), torch.randn(7, 64), torch.arange(7), empty, empty)
    both, _, _, _ = h.forward_batch([x1, x2])
    s1, _, _, _ = h.forward_batch([x1])
    s2, _, _, _ = h.forward_batch([x2])
    assert torch.allclose(both[0, :3], s1[0], atol=1e-5)
    assert torch.allclose(both[1, :7], s2[0], atol=1e-5)


def test_rollback_leaves_no_trace_of_rejected_drafts():
    h = _head()
    prompt = torch.randint(0, 100, (6,)).tolist()
    feats = torch.randn(40, 3 * HT)  # target features by position

    inc = Eagle3DraftEngine(h)
    hist = prompt + [17]
    inc.speculate(_req("s", len(hist), prompt=hist, feats=feats[:len(hist) - 1]))
    # three rounds accepting 0, 2, 1 drafts (+ the target's own token each)
    for extra in ([91], [3, 4, 92], [5, 93]):
        prev = len(hist)
        hist = hist + extra
        last = inc.speculate(_req("s", len(hist), accepted=extra,
                                  feats=feats[prev - 1:len(hist) - 1])).draft_tokens
        seq = inc.state["s"]
        assert seq.m == len(hist) - 1
        assert seq.k.shape[1] == seq.m + 3  # confirmed + (gamma - 1) chained drafts

    fresh = Eagle3DraftEngine(h).speculate(
        _req("f", len(hist), prompt=hist, feats=feats[:len(hist) - 1])).draft_tokens
    assert last == fresh


def test_hidden_count_mismatch_is_loud():
    eng = Eagle3DraftEngine(_head())
    with pytest.raises(ValueError, match="hidden-state count"):
        eng.speculate(_req("s", 5, prompt=[1, 2, 3, 4, 5], feats=torch.randn(2, 3 * HT)))


def test_batch_matches_one_by_one_and_variable_gamma():
    h = _head()
    feats = torch.randn(30, 3 * HT)
    reqs = [_req("a", 6, prompt=[1, 2, 3, 4, 5, 6], feats=feats[:5], gamma=2),
            _req("b", 9, prompt=list(range(10, 19)), feats=feats[5:13], gamma=5)]
    together = Eagle3DraftEngine(h).speculate_batch(reqs)
    alone = [Eagle3DraftEngine(h).speculate(r) for r in reqs]
    assert [r.draft_tokens for r in together] == [r.draft_tokens for r in alone]
    assert [len(r.draft_tokens) for r in together] == [2, 5]
    assert all(0 <= t < 100 for r in together for t in r.draft_tokens)
