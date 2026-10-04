"""SpecDec++ gamma policy (port of dsdSim SpecPPGammaPolicy), the acceptance
profile it reads, and per-request gamma in RemoteDraftProposer."""
from __future__ import annotations

import numpy as np
import pytest

from dsdreal.draft.engine_echo import EchoDraftEngine
from dsdreal.policy.window import AcceptanceProfile, SpecPPGamma, StaticGamma
from dsdreal.proto.transport import InProcessDraftClient
from dsdreal.target import proposer


def _profile(*probs, edges=()):
    return AcceptanceProfile(edges=list(edges), probs=[list(p) for p in probs])


def test_specpp_hand_computed():
    # cum accept: .9, .81, .729, .6561, .59049, .531441, .478, .430 -> reject
    # 1-cum stays <= 0.7 all the way -> max
    assert SpecPPGamma(_profile([0.9] * 8)).select_gamma(50) == 8
    # p = .6: cum .6 .36 .216 -> 1-cum .4 .64 .784 > .7 at idx 3 -> 2
    assert SpecPPGamma(_profile([0.6] * 8)).select_gamma(50) == 2
    # p = .8: cum .8 .64 .512 .4096 .32768 .262 -> 1-.262=.738 > .7 at 6 -> 5
    assert SpecPPGamma(_profile([0.8] * 8)).select_gamma(50) == 5


def test_specpp_clamps_and_pads():
    # very low acceptance still gets min_gamma
    assert SpecPPGamma(_profile([0.1] * 8)).select_gamma(10) == 2
    # short profile is padded with its last value (as the simulator does)
    assert SpecPPGamma(_profile([0.9, 0.9])).select_gamma(10) == 8
    # no profile -> fallback
    assert SpecPPGamma(None, fallback_gamma=4).select_gamma(10) == 4


def test_specpp_uses_context_bucket():
    prof = _profile([0.6] * 8, [0.95] * 8, edges=[100])
    pol = SpecPPGamma(prof)
    assert pol.select_gamma(50) == 2
    assert pol.select_gamma(150) == 8


def test_profile_from_steps_counts_only_observed_positions():
    # gamma 4: accepted 1 -> positions 1 (acc), 2 (rejected) observed
    rows = [{"request_id": "a", "n_draft": 4, "accepted": 1}] * 30 \
         + [{"request_id": "a", "n_draft": 4, "accepted": 4}] * 30
    prof = AcceptanceProfile.from_steps(rows, {"a": 10}, min_count=10)
    p = prof.position_probabilities(10)
    assert p[0] == pytest.approx(1.0)          # 60/60
    assert p[1] == pytest.approx(30 / 60)      # 30 accepted of 60 tried
    assert p[2] == pytest.approx(1.0) and p[3] == pytest.approx(1.0)


def test_profile_json_roundtrip(tmp_path):
    prof = _profile([0.5, 0.4], [0.9, 0.8], edges=[64])
    prof.to_json(tmp_path / "p.json")
    back = AcceptanceProfile.from_json(tmp_path / "p.json")
    assert back.edges == [64] and back.probs == prof.probs


class _Cfg:
    def __init__(self, k):
        self.speculative_config = type("S", (), {"num_speculative_tokens": k})()


@pytest.fixture(autouse=True)
def _reset():
    yield
    proposer.configure(None)


def _batch(histories, max_len=64):
    n = len(histories)
    tok = np.zeros((n, max_len), dtype=np.int64)
    for i, h in enumerate(histories):
        tok[i, :len(h)] = h
    return [[1]] * n, np.array([len(h) for h in histories]), tok


def test_proposer_gamma_is_per_request_sticky_and_capped():
    class ByLen:
        def select_gamma(self, context_len):
            return 2 if context_len < 5 else 99  # 99 -> capped at k

    proposer.configure(InProcessDraftClient(EchoDraftEngine()), window=ByLen())
    p = proposer.RemoteDraftProposer(_Cfg(6))
    short, long_ = [1, 2, 3, 4], [5] * 9
    out = p.propose(*_batch([short, long_]))
    assert [len(o) for o in out] == [2, 6]
    out = p.propose(*_batch([short + [7, 8, 9], long_ + [1, 2]]))
    assert [len(o) for o in out] == [2, 6]


def test_static_gamma_policy():
    proposer.configure(InProcessDraftClient(EchoDraftEngine()), window=StaticGamma(3))
    p = proposer.RemoteDraftProposer(_Cfg(8))
    (out,) = p.propose(*_batch([[1, 2, 3]]))
    assert len(out) == 3
