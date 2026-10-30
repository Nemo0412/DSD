"""Length-aware admission queue for the vLLM target.

vLLM's default waiting queue is FCFS. This scheduler keeps that for the
running batch and only changes who is admitted next: the head-of-line
request fixes a prompt-length bucket, and later pops prefer the same
bucket. An empty bucket falls back to FCFS, so a request cannot starve.
"""
from __future__ import annotations

from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler


class LABScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waiting = LABRequestQueue()


class LABRequestQueue(FCFSRequestQueue):
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
        self._bucket = self.bucket_of(self[0])
        return super().pop_request()
