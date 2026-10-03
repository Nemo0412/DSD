"""Scheduler plug-points under test for K3 / K4.

K3 proves `--scheduler-cls` is a live extension point.
K4 proves `Scheduler.waiting` can be swapped for a custom RequestQueue —
which is where LAB batching will live (admission-time grouping, never a
reorder of `self.running`).

Counters go through files so the probe works whether or not the engine core
runs in this process.
"""
from __future__ import annotations

import os

from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler

_COUNT = os.environ.get("DSD_PROBE_SCHED_COUNT", "runs/k3_schedule_calls")
_POPS = os.environ.get("DSD_PROBE_POP_COUNT", "runs/k4_queue_pops")


def _bump(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write("x")


def _read(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_count() -> int:
    return _read(_COUNT)


def read_pops() -> int:
    return _read(_POPS)


def reset() -> None:
    for p in (_COUNT, _POPS):
        if os.path.exists(p):
            os.remove(p)


class CountingScheduler(Scheduler):
    """Delegates entirely to vLLM; only records that it was used."""

    def schedule(self, throttle_prefills: bool = False):
        _bump(_COUNT)
        return super().schedule(throttle_prefills)


class BucketQueueScheduler(Scheduler):
    """Installs a length-bucketed waiting queue in place of FCFS."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waiting = LABRequestQueue()


class LABRequestQueue(FCFSRequestQueue):
    """Length-aware admission queue.

    The head-of-line request fixes a length bucket; subsequent pops prefer
    requests in that same bucket so the batch is composed of similar-length
    work. Falls back to strict FCFS whenever the bucket is exhausted, so the
    queue can never starve a request.
    """

    EDGES = (32, 64, 128, 256, 512, 1024)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bucket: int | None = None

    @classmethod
    def bucket_of(cls, request) -> int:
        n = request.num_prompt_tokens
        for i, edge in enumerate(cls.EDGES):
            if n <= edge:
                return i
        return len(cls.EDGES)

    def pop_request(self):
        _bump(_POPS)
        if not self:
            return super().pop_request()
        if self._bucket is None:
            self._bucket = self.bucket_of(self[0])
        for i, req in enumerate(self):
            if self.bucket_of(req) == self._bucket:
                if i == 0:
                    return super().pop_request()
                self.remove(req)
                return req
        # Bucket exhausted: re-anchor on the new head.
        self._bucket = self.bucket_of(self[0])
        return super().pop_request()
