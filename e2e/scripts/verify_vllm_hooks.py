#!/usr/bin/env python
"""Stage 0 kill-criteria probes for dsd-real.

Each probe is a go/no-go gate on the "plug vLLM, don't fork it" plan.
GPU probes run in their own subprocess so a crash contains itself.

    python scripts/verify_vllm_hooks.py            # run all
    python scripts/verify_vllm_hooks.py --probe k1 # run one
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

MODEL = os.environ.get("DSD_PROBE_MODEL", "Qwen/Qwen3-0.6B")
GAMMA = 4
GPU_FRAC = float(os.environ.get("DSD_PROBE_GPU_FRAC", "0.35"))
MAX_LEN = 1024

PROMPTS = [
    "The capital of France is",
    "In one sentence, explain why the sky is blue:",
    "List three prime numbers:",
    "def fibonacci(n):",
]


# --------------------------------------------------------------------------
# shared engine helpers
# --------------------------------------------------------------------------
def _quiet_env() -> None:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    # Keep engine core in-process so our monkeypatch reaches the model runner.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def patch_drafter(factory):
    """Replace GPUModelRunner.drafter right after construction.

    This is the exact mechanism dsdreal/target/plugin.py will use, so the
    probe validates the install path as well as the behaviour under test.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if getattr(self, "drafter", None) is not None:
            self.drafter = factory(self.vllm_config, self.drafter)

    GPUModelRunner.__init__ = __init__
    return original


def build_llm(spec: bool = True, **overrides):
    from vllm import LLM

    kwargs = dict(
        model=MODEL,
        gpu_memory_utilization=GPU_FRAC,
        max_model_len=MAX_LEN,
        enforce_eager=True,
        dtype="bfloat16",
        disable_log_stats=True,
    )
    if spec:
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": GAMMA,
            "prompt_lookup_min": 1,
            "prompt_lookup_max": GAMMA,
        }
    kwargs.update(overrides)
    return LLM(**kwargs)


def greedy(llm, prompts, max_tokens=32):
    from vllm import SamplingParams

    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False
    )
    return [list(o.outputs[0].token_ids) for o in outs]


def verdict(name, passed, detail, **extra):
    print(json.dumps({"probe": name, "pass": bool(passed), "detail": detail, **extra}))
    return 0 if passed else 1


# --------------------------------------------------------------------------
# K6 - static: does this vLLM expose method="custom_class"?
# --------------------------------------------------------------------------
def probe_k6():
    import typing

    import vllm
    from vllm.config.speculative import SpeculativeMethod

    methods = list(typing.get_args(SpeculativeMethod))
    flat = []
    for m in methods:
        flat.extend(typing.get_args(m) or [m])
    native = "custom_class" in flat
    return verdict(
        "k6",
        True,  # informational: decides which install path compat.py takes
        f"vllm=={vllm.__version__} native custom_class={native} -> "
        f"{'native hook' if native else 'NgramProposer subclass shim'}",
        vllm_version=vllm.__version__,
        native_custom_class=native,
        methods=sorted(set(flat)),
    )


# --------------------------------------------------------------------------
# K1 - can a proposer return variable-length drafts (per request)?
# --------------------------------------------------------------------------
def probe_k1():
    _quiet_env()
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer

    seen = {"lengths": [], "calls": 0}

    class VariableLengthProposer(NgramProposer):
        """Returns a different gamma for each request in the batch."""

        PATTERN = [0, 1, 3, 2]

        def propose(self, num_speculative_tokens, sampled_token_ids, num_tokens_no_spec,
                    token_ids_cpu, slot_mappings=None):
            seen["calls"] += 1
            out = []
            for i, sampled in enumerate(sampled_token_ids):
                if not sampled:
                    out.append([])
                    continue
                n = self.PATTERN[i % len(self.PATTERN)]
                # Arbitrary in-vocab ids; they will simply be rejected.
                out.append([(1000 + i * 7 + j) for j in range(n)])
            seen["lengths"].append([len(x) for x in out])
            return out

    patch_drafter(lambda cfg, old: VariableLengthProposer(cfg))
    llm = build_llm()
    spec_out = greedy(llm, PROMPTS)
    del llm

    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    base = build_llm(spec=False)
    base_out = greedy(base, PROMPTS)
    del base

    widths = sorted({tuple(sorted(set(x))) for x in seen["lengths"]})
    identical = spec_out == base_out
    varied = any(len(set(row)) > 1 for row in seen["lengths"])
    ok = identical and varied and seen["calls"] > 0
    return verdict(
        "k1",
        ok,
        f"calls={seen['calls']} variable_within_batch={varied} "
        f"output_identical_to_nonspec={identical} distinct_len_sets={widths[:3]}",
        calls=seen["calls"],
        varied=varied,
        identical=identical,
    )


# --------------------------------------------------------------------------
# K2 - is propose() on the critical path? (does injected delay reach TPOT?)
# --------------------------------------------------------------------------
def probe_k2():
    _quiet_env()
    import time as _time

    from vllm.v1.spec_decode.ngram_proposer import NgramProposer

    delay_s = float(os.environ.get("DSD_PROBE_DELAY_MS", "5")) / 1000.0

    class SleepProposer(NgramProposer):
        """Proposes nothing, but burns `delay_s` per engine step."""

        def __init__(self, cfg, delay):
            # NgramProposer.__init__ calls self.propose() to trigger the numba
            # JIT (ngram_proposer.py:57), so our state must exist first.
            self.delay = 0.0
            self.calls = 0
            super().__init__(cfg)
            self.delay = delay

        def propose(self, num_speculative_tokens, sampled_token_ids, num_tokens_no_spec,
                    token_ids_cpu, slot_mappings=None):
            self.calls += 1
            if self.delay:
                _time.sleep(self.delay)
            return [[] for _ in sampled_token_ids]

    holder = {}

    def factory(cfg, old):
        p = SleepProposer(cfg, holder["delay"])
        holder["proposer"] = p
        return p

    patch_drafter(factory)

    def run(delay, n_tokens=48):
        holder["delay"] = delay
        llm = build_llm()
        greedy(llm, PROMPTS[:1], max_tokens=8)  # warm up
        t0 = _time.perf_counter()
        toks = greedy(llm, PROMPTS[:1], max_tokens=n_tokens)
        dt = _time.perf_counter() - t0
        n = len(toks[0])
        calls = holder["proposer"].calls
        del llm
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        return (dt / max(n - 1, 1)) * 1000.0, n, calls

    tpot0, n0, _ = run(0.0)
    tpot1, n1, _ = run(delay_s)
    observed = tpot1 - tpot0
    expected = delay_s * 1000.0
    # Sleep lands on the critical path if we recover most of the injected delay.
    ok = observed >= 0.6 * expected
    return verdict(
        "k2",
        ok,
        f"tpot(0ms)={tpot0:.2f}ms tpot({expected:.0f}ms)={tpot1:.2f}ms "
        f"delta={observed:.2f}ms expected~{expected:.0f}ms "
        f"recovered={100*observed/expected:.0f}%",
        tpot_base_ms=round(tpot0, 3),
        tpot_delayed_ms=round(tpot1, 3),
        delta_ms=round(observed, 3),
        expected_ms=expected,
    )


# --------------------------------------------------------------------------
# K3 - is --scheduler-cls a working plug point?
# --------------------------------------------------------------------------
def probe_k3():
    _quiet_env()
    import dsd_probe_sched  # noqa: F401  (module defines the counter)

    llm = build_llm(spec=False, scheduler_cls="dsd_probe_sched.CountingScheduler")
    out = greedy(llm, PROMPTS, max_tokens=16)
    del llm
    calls = dsd_probe_sched.read_count()
    ok = calls > 0 and all(len(o) > 0 for o in out)
    return verdict(
        "k3",
        ok,
        f"CountingScheduler.schedule() called {calls}x; outputs non-empty={all(len(o)>0 for o in out)}",
        schedule_calls=calls,
    )


# --------------------------------------------------------------------------
# K4 - can self.waiting be swapped for a custom RequestQueue?
# --------------------------------------------------------------------------
def probe_k4():
    _quiet_env()
    import dsd_probe_sched  # noqa: F401

    n_req = int(os.environ.get("DSD_PROBE_K4_REQS", "40"))
    prompts = [f"{PROMPTS[i % len(PROMPTS)]} (variant {i})" for i in range(n_req)]

    base = build_llm(spec=False)
    base_out = greedy(base, prompts, max_tokens=16)
    del base
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    llm = build_llm(spec=False, scheduler_cls="dsd_probe_sched.BucketQueueScheduler")
    lab_out = greedy(llm, prompts, max_tokens=16)
    del llm

    pops = dsd_probe_sched.read_pops()
    identical = base_out == lab_out
    ok = identical and pops >= n_req
    return verdict(
        "k4",
        ok,
        f"custom RequestQueue popped {pops} requests; "
        f"outputs identical to FCFS baseline={identical} (n={n_req})",
        pops=pops,
        identical=identical,
        n_requests=n_req,
    )


# --------------------------------------------------------------------------
# K5 - does a vllm.general_plugins entry point load inside worker processes?
# --------------------------------------------------------------------------
def probe_k5():
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    # Deliberately NOT disabling V1 multiprocessing: that is the case we must prove.
    os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    marker = os.path.abspath("runs/k5_plugin_marker.jsonl")
    if os.path.exists(marker):
        os.remove(marker)
    os.environ["DSD_PLUGIN_MARKER"] = marker

    from importlib.metadata import entry_points

    eps = [e for e in entry_points(group="vllm.general_plugins")]
    names = [f"{e.name} -> {e.value}" for e in eps]
    if not any(e.name == "dsdreal_probe" for e in eps):
        return verdict(
            "k5",
            False,
            "entry point 'dsdreal_probe' not registered; run: "
            "pip install -e . in the probe venv. discovered=" + str(names),
            discovered=names,
        )

    llm = build_llm(spec=False)
    greedy(llm, PROMPTS[:1], max_tokens=8)
    del llm

    pids = []
    if os.path.exists(marker):
        with open(marker) as fh:
            pids = [json.loads(line)["pid"] for line in fh if line.strip()]
    ok = len(pids) > 0
    return verdict(
        "k5",
        ok,
        f"register() fired in {len(set(pids))} distinct process(es): {sorted(set(pids))}",
        pids=sorted(set(pids)),
        discovered=names,
    )


# --------------------------------------------------------------------------
# K7 - can we observe the verify forward (L2: logits + SpecDecodeMetadata)?
# --------------------------------------------------------------------------
def probe_k7():
    _quiet_env()
    from vllm.v1.sample.rejection_sampler import RejectionSampler
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer

    gamma = 3
    tap = {"calls": 0, "draft_tokens": 0, "logits_rows": 0, "vocab": None,
           "draft_probs_none": 0, "live": False}
    proposed = {"tokens": 0}

    class FixedGammaProposer(NgramProposer):
        def propose(self, num_speculative_tokens, sampled_token_ids, num_tokens_no_spec,
                    token_ids_cpu, slot_mappings=None):
            if not tap["live"]:
                return [[] for _ in sampled_token_ids]
            out = []
            for i, sampled in enumerate(sampled_token_ids):
                if not sampled:
                    out.append([])
                    continue
                out.append([1000 + i * 7 + j for j in range(gamma)])
            proposed["tokens"] += sum(len(x) for x in out)
            return out

    original_forward = RejectionSampler.forward

    def tapped_forward(self, metadata, draft_probs, logits, sampling_metadata):
        if not tap["live"]:
            # Profiling warm-up (_dummy_sampler_run) — random logits, dummy
            # metadata. Not a real verify forward; must not be counted.
            return original_forward(self, metadata, draft_probs, logits,
                                    sampling_metadata)
        tap["calls"] += 1
        tap["draft_tokens"] += int(sum(metadata.num_draft_tokens))
        tap["logits_rows"] += int(logits.shape[0])
        tap["vocab"] = int(logits.shape[-1])
        tap["draft_probs_none"] += int(draft_probs is None)
        return original_forward(self, metadata, draft_probs, logits, sampling_metadata)

    RejectionSampler.forward = tapped_forward
    patch_drafter(lambda cfg, old: FixedGammaProposer(cfg))

    llm = build_llm()
    tap["live"] = True          # engine built; everything from here is real work
    greedy(llm, PROMPTS, max_tokens=24)
    tap["live"] = False
    del llm
    RejectionSampler.forward = original_forward

    # The sampler sees every draft token we proposed, except the final step's
    # proposal which no subsequent forward consumes.
    ok = (
        tap["calls"] > 0
        and tap["draft_tokens"] > 0
        and tap["draft_tokens"] <= proposed["tokens"]
        and tap["draft_tokens"] >= proposed["tokens"] - gamma * len(PROMPTS)
    )
    return verdict(
        "k7",
        ok,
        f"forward taps={tap['calls']} sum(num_draft_tokens)={tap['draft_tokens']} "
        f"proposed={proposed['tokens']} vocab={tap['vocab']} "
        f"draft_probs_is_None={tap['draft_probs_none']}/{tap['calls']}",
        **{k: v for k, v in tap.items()},
        proposed_tokens=proposed["tokens"],
    )


PROBES = {
    "k1": probe_k1,
    "k2": probe_k2,
    "k3": probe_k3,
    "k4": probe_k4,
    "k5": probe_k5,
    "k6": probe_k6,
    "k7": probe_k7,
}

TITLES = {
    "k1": "variable-length drafts per request",
    "k2": "propose() is on the critical path",
    "k3": "--scheduler-cls plug point",
    "k4": "custom RequestQueue (LAB lives here)",
    "k5": "entry-point plugin loads in workers",
    "k6": "custom_class availability (informational)",
    "k7": "L2 tap on the verify forward",
}


def run_all(only=None, timeout=900):
    order = only or ["k6", "k1", "k2", "k7", "k3", "k4", "k5"]
    results = {}
    for name in order:
        print(f"\n=== {name.upper()}  {TITLES[name]} " + "=" * 28, flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--probe", name],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        payload = None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith('{"probe"'):
                payload = json.loads(line)
        if payload is None:
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
            payload = {"probe": name, "pass": False,
                       "detail": "no verdict emitted; tail:\n  " + "\n  ".join(tail)}
        payload["seconds"] = round(time.time() - t0, 1)
        results[name] = payload
        mark = "PASS" if payload["pass"] else "FAIL"
        print(f"[{mark}] {payload['detail']}  ({payload['seconds']}s)", flush=True)

    os.makedirs("runs", exist_ok=True)
    with open("runs/stage0_probes.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 72)
    for name in order:
        r = results[name]
        print(f"  {name.upper()}  {'PASS' if r['pass'] else 'FAIL'}  {TITLES[name]}")
    blockers = [n for n in ("k1", "k2") if not results.get(n, {}).get("pass")]
    print("=" * 72)
    if blockers:
        print(f"BLOCKING: {', '.join(b.upper() for b in blockers)} failed -> "
              "the plugin path cannot control gamma. Pivot to a pinned vLLM fork "
              "(patch gpu_model_runner.py dispatch), NOT nano-vllm.")
    else:
        print("K1+K2 green: the plug-don't-fork plan holds. Proceed to Stage 1.")
    print("Wrote runs/stage0_probes.json")
    return 0 if not blockers else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=sorted(PROBES) + ["all"], default="all")
    args = ap.parse_args()
    if args.probe == "all":
        sys.exit(run_all())
    sys.exit(PROBES[args.probe]())
