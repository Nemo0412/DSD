"""Thin runtime shim that runs the DSpec simulator (src/) under the real-system
setup. Nothing in src/ is edited; everything here is a
monkeypatch applied after import.

What it does (see sim/README.md for the why):

1. Import plumbing. DSpec's `src/performance/vidur_realtime.py` raises at
   import time when `thirdparty/Sai_speculative_vidur` is missing, and that
   module is imported unconditionally via `performance/__init__.py`, so
   without VIDUR the simulator cannot even be imported. We pre-register a stub
   module under that name. `acceptance/regressor.py` also does
   `from src import sim` (expects the repo root on sys.path); we alias the
   already-imported `sim` module as `src.sim` so it is not imported twice.

2. `CalibratedProvider`: a performance provider driven by per-device
   `dspec_real_calib` blocks that make_config.py writes into the YAML. It covers
   every latency path the simulator asks a provider for:
     - target prefill  (TargetServer._make_phase_request, phase="prefill")
     - target verify   (TargetServer._make_phase_request, phase="decode",
                        and TargetServer._estimate_target_latency, phase="verify")
     - draft generation (DraftServer._predict_generation_latency, phase="decode",
                        target_id = the draft's own id)
   Selected with `performance_model.type: dspec_real_calibrated`.

3. `TargetServer._estimate_batch_latency_from_provider` is replaced so that
   (a) the prefill-part / decode-part combination rule of a mixed batch is
   configurable ("max" = DSpec's native rule, "sum" = additive), and
   (b) "lockstep" mode can add the per-step RTT + drafting time to every
   target batch (emulates our vLLM custom_class proposer, which blocks the
   engine step while drafts are fetched).

4. The acceptance classifier is a RandomForest pickled with n_jobs=-1; its
   per-call thread-pool overhead dominated runtime (~26 ms vs ~4 ms per
   predict_proba). We set n_jobs=1 after loading; predictions are identical.
"""
from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_SIM = None


def dspec_sim_root(explicit: Optional[str] = None) -> Path:
    root = explicit or os.environ.get("DSPEC_SIM_ROOT") or Path(__file__).resolve().parents[1]
    p = Path(root).resolve()
    if not (p / "src" / "sim.py").is_file():
        raise SystemExit(f"{p} does not look like a DSpec checkout (no src/sim.py)")
    return p


def load_sim(root: Optional[str] = None):
    """Import DSpec's `sim` module with all shims applied. Idempotent."""
    global _SIM
    if _SIM is not None:
        return _SIM
    src = str(dspec_sim_root(root) / "src")
    if src not in sys.path:
        # must precede stdlib: DSpec has a top-level package named `trace`
        sys.path.insert(0, src)

    stub = types.ModuleType("performance.vidur_realtime")

    class VidurRealtimeRunner:  # pragma: no cover - never used
        def __init__(self, *a, **k):
            raise RuntimeError("VIDUR is not available (stubbed by dspec_sim_shim)")

    stub.VidurRealtimeRunner = VidurRealtimeRunner
    stub.get_shared_vidur_runner = lambda *a, **k: None
    sys.modules.setdefault("performance.vidur_realtime", stub)

    import sim  # noqa: E402  (DSpec's src/sim.py)

    pkg = sys.modules.get("src") or types.ModuleType("src")
    pkg.sim = sim
    sys.modules["src"] = pkg
    sys.modules["src.sim"] = sim

    _patch_provider_factory(sim)
    _patch_batch_latency(sim)
    _patch_serve_loop_tail(sim)
    _patch_acceptance_loader(sim)
    _patch_acceptance_profile(sim)
    _patch_acceptance_replay(sim)
    _SIM = sim
    return sim


# --------------------------------------------------------------------------
# Calibrated latency provider
# --------------------------------------------------------------------------

class CalibratedProvider:
    """Latency model from the `dspec_real_calib` block of each device.

    target prefill (one batch):
        prefill_base_ms + prefill_ms_per_token * sum(prompt tokens in batch)
    target verify (one batch of B requests, request i verifies g_i drafts):
        verify_base_ms + verify_ms_per_seq * B
        + verify_ms_per_token * sum(g_i + 1)          (+1 = bonus position)
        + verify_ms_per_ctx_token * sum(context_i)
    draft generation (one request, g tokens):
        base_ms + ms_per_token * g           (async mode)
        0                                    (lockstep mode: charged on target)
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, Dict[str, Any]] = {}

    def register_target(self, *, target_id: str, model: str, hardware: str,
                        metadata: Optional[Mapping[str, Any]] = None) -> None:
        self._profiles[str(target_id)] = dict(metadata or {})

    def flush(self) -> None:
        return

    def get_metrics(self, request):
        from performance import PhaseMetrics  # DSpec's dataclass

        meta = self._profiles.get(str(request.target_id or ""))
        if meta is None:
            return None
        calib = meta.get("dspec_real_calib") or {}
        role = meta.get("role", "target")
        if role == "draft":
            if calib.get("mode") == "lockstep":
                return PhaseMetrics(latency_ms=0.0)
            g = max(1, int(request.tokens_to_generate or request.fanout or 1))
            return PhaseMetrics(latency_ms=draft_ms(calib, g))
        t = calib.get("target", {})
        if request.phase == "prefill":
            toks = request.tokens_per_request or [max(1, request.sequence_length)] * max(1, request.batch_size)
            lat = float(t.get("prefill_base_ms", 0.0)) + float(t.get("prefill_ms_per_token", 0.0)) * sum(toks)
            return PhaseMetrics(latency_ms=lat)
        # verify / decode on the target
        toks = request.tokens_per_request or [max(1, request.tokens_to_generate or request.fanout or 1)]
        ctxs = request.context_per_request or [max(0, request.context_length)] * len(toks)
        b = len(toks)
        lat = (float(t.get("verify_base_ms", 0.0))
               + float(t.get("verify_ms_per_seq", 0.0)) * b
               + float(t.get("verify_ms_per_token", 0.0)) * sum(int(x) + 1 for x in toks)
               + float(t.get("verify_ms_per_ctx_token", 0.0)) * sum(ctxs))
        return PhaseMetrics(latency_ms=lat)


def draft_ms(calib: Mapping[str, Any], gamma: int, seqs_on_drafter: int = 1) -> float:
    d = calib.get("draft", {})
    return (float(d.get("base_ms", 0.0))
            + float(d.get("ms_per_token", 0.0)) * gamma
            + float(d.get("ms_per_token_per_seq", 0.0)) * gamma * max(0, seqs_on_drafter - 1))


def _patch_provider_factory(sim) -> None:
    orig = sim.create_performance_provider

    def create(config):
        if str(getattr(config, "type", "") or "").lower() == "dspec_real_calibrated":
            return CalibratedProvider()
        return orig(config)

    sim.create_performance_provider = create


def _patch_batch_latency(sim) -> None:
    """Replace TargetServer._estimate_batch_latency_from_provider.

    Native DSpec: latency = max(provider(prefill jobs), provider(decode jobs)).
    Here: combine by calib.target.mixed_batch_combine ("max" | "sum"), then in
    lockstep mode add rtt_ms + drafting time of the slowest drafter.
    """
    TargetServer = sim.TargetServer
    native = TargetServer._estimate_batch_latency_from_provider

    def patched(self, batch):
        calib = (self.metadata or {}).get("dspec_real_calib")
        if not calib:
            return native(self, batch)
        provider = getattr(self, "performance", None)
        if provider is None or not batch:
            return None
        parts = []
        for phase_name in ("prefill", "decode"):
            jobs = [j for j in batch if (j.job_type == "prefill") == (phase_name == "prefill")]
            if not jobs:
                continue
            req = self._make_phase_request(phase_name, jobs)
            m = provider.get_metrics(req) if req is not None else None
            if m is not None:
                parts.append(max(0.0, m.latency_ms))
        if not parts:
            return None
        combine = str(calib.get("target", {}).get("mixed_batch_combine", "max")).lower()
        lat = sum(parts) if combine == "sum" else max(parts)
        # Lockstep: a vLLM step is [forward: verify + prefill] then [propose:
        # draft + RTT for every running request]. The drafting is a *tail*
        # after the forward -- the batch's jobs are done when the forward ends
        # (see _patch_serve_loop_tail) and the target stays busy for the tail.
        # Charging it as a head instead made a request whose prefill joined a
        # batch wait for other requests' drafting, then pay its own first
        # draft again (~one draft + RTT too much TTFT). Prefill-only batches
        # get the tail too: that propose is the new request's first draft.
        self._dsd_tail_ms = 0.0
        if calib.get("mode") == "lockstep":
            self._dsd_tail_ms = lockstep_extra_ms(calib, batch, self.cfg.gamma)
            lat += self._dsd_tail_ms
        return lat

    TargetServer._estimate_batch_latency_from_provider = patched


def _patch_serve_loop_tail(sim) -> None:
    """TargetServer._serve_loop with the lockstep drafting tail after job
    completion: wait (batch latency - tail), complete the batch's jobs, then
    keep holding the server for the tail before taking the next batch.

    Done by rewriting two lines of DSpec's own loop (each asserted to occur
    exactly once), so the rest of its batching logic is untouched. A no-op
    when the tail is 0 (async mode)."""
    import inspect
    import textwrap

    lines = textwrap.dedent(inspect.getsource(sim.TargetServer._serve_loop)).split("\n")

    def at(pred, what):
        hits = [i for i, l in enumerate(lines) if pred(l.strip())]
        if len(hits) != 1:
            raise RuntimeError(f"DSpec _serve_loop changed, cannot patch ({what}): {hits}")
        i = hits[0]
        return i, lines[i][:len(lines[i]) - len(lines[i].lstrip())]

    # 1. the batch sleep -> sleep only through the forward
    i, ind = at(lambda l: l.startswith("yield self.env.timeout(batch_latency)"), "batch sleep")
    lines[i:i + 1] = [ind + "_tail = getattr(self, '_dsd_tail_ms', 0.0) or 0.0",
                      ind + "yield self.env.timeout(max(0.0, batch_latency - _tail))"]
    # 2. after the completion loop (still inside `with server.request()`) -> hold for the tail
    i, ind = at(lambda l: l.startswith("if self.router and hasattr(self.router, 'mark_idle')"), "post-batch")
    lines[i:i] = [ind + "if _tail > 0:",
                  ind + "    yield self.env.timeout(_tail)",
                  ind + "    self.busy_ms += _tail"]

    ns: Dict[str, Any] = {}
    exec(compile("\n".join(lines), "<dspec_sim _serve_loop + lockstep tail>", "exec"), vars(sim), ns)
    sim.TargetServer._serve_loop = ns["_serve_loop"]
    sim.TargetServer._dsd_patched_serve_loop_src = "\n".join(lines)


def lockstep_extra_ms(calib: Mapping[str, Any], batch, default_gamma: int) -> float:
    """RTT + drafting time added to a lockstep target step.

    Every request in the step needs fresh drafts; requests are spread over the
    N drafters (round-robin), drafters run in parallel, the step waits for the
    slowest one.
    """
    phys: Dict[str, int] = {}
    gmax = 1
    for j in batch:
        key = str(j.draft_id).split("_r")[0]
        phys[key] = phys.get(key, 0) + 1
        g = j.gamma_override if getattr(j, "gamma_override", 0) > 0 else (
            j.token_count if j.job_type != "prefill" else default_gamma)
        gmax = max(gmax, int(g))
    per_drafter = max(phys.values()) if phys else 1
    return float(calib.get("rtt_ms", 0.0)) + draft_ms(calib, gmax, per_drafter)


_PROFILE_CACHE: Dict[str, Any] = {}


def _patch_acceptance_profile(sim) -> None:
    """Optional: `speculation.acceptance.dspec_real_profile: <profile.json>`.

    Replaces the joblib classifier's per-position probabilities with a
    measured DSpec AcceptanceProfile (scripts/build_acceptance_profile.py:
    {"edges": [...], "probs": [[p1..pk] per context bucket]}). The bucket is
    chosen with the context length DSpec passes (prompt length for
    SpecDec++'s gamma choice, current context for verification). Positions
    past the profile depth repeat the last probability. Everything
    downstream (sequential Bernoulli verification, SpecPP selection) is
    unchanged DSpec code.
    """
    import bisect
    import json

    DraftServer = sim.DraftServer
    native = DraftServer._lookup_acceptance

    def patched(self, conn, context_length, tokens):
        path = (self.cfg.acceptance_config or {}).get("dspec_real_profile")
        if not path:
            return native(self, conn, context_length, tokens)
        prof = _PROFILE_CACHE.get(path)
        if prof is None:
            prof = json.loads(Path(path).read_text())
            _PROFILE_CACHE[path] = prof
        row = list(prof["probs"][bisect.bisect_right(prof["edges"], int(context_length))])
        tokens = int(max(1, tokens))
        if not row:
            row = [float(self.cfg.acceptance_config.get("default_rate", 0.7))]
        while len(row) < tokens:
            row.append(row[-1])
        probs = tuple(max(0.0, min(1.0, float(p))) for p in row[:tokens])
        return sum(probs) / len(probs), probs

    DraftServer._lookup_acceptance = patched


_REPLAY_CACHE: Dict[str, Any] = {}


def _patch_acceptance_replay(sim) -> None:
    """Optional: `speculation.acceptance.dspec_real_replay: <replay.json>`
    ({draft_device_id: {"gamma": g, "accepted": [a_0, a_1, ...]}}, written by
    make_config.py --replay-from <real run>).

    Replays a real run's token events instead of sampling acceptance -- the
    paper's acceptance_seq replay, fed from our own steps.jsonl. make_config
    gives every request its own virtual drafter, so the drafter id is the
    sequence: its conversation uses the recorded gamma, and its k-th
    verification accepts exactly the recorded a_k drafts (no bonus, DSpec's
    convention). The trace sets answer = rounds * gamma, so DSpec's
    ceil(answer / gamma) loop runs exactly the recorded rounds.
    """
    import json

    DraftServer = sim.DraftServer
    base_policy = sim.ConstantGammaPolicy
    native_verify = DraftServer._simulate_verification
    native_init = DraftServer.__init__

    def table(self):
        path = (self.cfg.acceptance_config or {}).get("dspec_real_replay")
        if not path:
            return None
        t = _REPLAY_CACHE.get(path)
        if t is None:
            t = _REPLAY_CACHE[path] = json.loads(Path(path).read_text())
        return t.get(self.id)

    class ReplayGamma(base_policy):
        def __init__(self, gamma: int) -> None:
            super().__init__(gamma)

    def __init__(self, *args, **kwargs):
        native_init(self, *args, **kwargs)
        rp = table(self)
        self._dsd_replay_round = 0
        if rp is not None:
            self._gamma_policy = ReplayGamma(int(rp["gamma"]))
            self.gamma = int(rp["gamma"])

    def verify(self, tokens, conn, context_length, acceptance_cursor=None, **kwargs):
        rp = table(self)
        if rp is None:
            return native_verify(
                self, tokens, conn, context_length,
                acceptance_cursor=acceptance_cursor, **kwargs,
            )
        k = self._dsd_replay_round
        self._dsd_replay_round += 1
        tokens = max(0, int(tokens))
        acc = rp["accepted"][k] if k < len(rp["accepted"]) else 0
        acc = max(0, min(int(acc), tokens))
        return sim.VerifyResult(chunk_id=self.chunks_sent, accepted_tokens=acc,
                                rejected_tokens=tokens - acc, total_tokens=tokens)

    DraftServer.__init__ = __init__
    DraftServer._simulate_verification = verify


def _patch_acceptance_loader(sim) -> None:
    Reg = sim.AcceptanceRegressor
    orig = Reg.from_file.__func__

    def from_file(cls, path, *, surrogate_config=None):
        obj = orig(cls, path, surrogate_config=surrogate_config)
        for attr in ("_classifier", "classifier", "_regressor", "regressor"):
            est = getattr(obj, attr, None)
            if est is not None and hasattr(est, "n_jobs"):
                est.n_jobs = 1
        for attr in ("_classifiers", "_regressors"):
            group = getattr(obj, attr, None)
            if isinstance(group, dict):
                for est in group.values():
                    if hasattr(est, "n_jobs"):
                        est.n_jobs = 1
        return obj

    Reg.from_file = classmethod(from_file)
