"""One long-lived worker per physical GPU, sharing a dynamically claimed job queue."""

from __future__ import annotations

import concurrent.futures
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar


Job = TypeVar("Job")
Result = TypeVar("Result")


@dataclass(frozen=True)
class GPUJobOutcome(Generic[Job, Result]):
    index: int
    job: Job
    gpu: str
    result: Result | None = None
    error: Exception | None = None


@dataclass(frozen=True)
class GPUScheduleSummary(Generic[Job, Result]):
    outcomes: tuple[GPUJobOutcome[Job, Result], ...]
    not_started: tuple[tuple[int, Job], ...]

    @property
    def failures(self) -> tuple[GPUJobOutcome[Job, Result], ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.error is not None)


def run_gpu_jobs(
    jobs: Sequence[Job],
    gpu_ids: Sequence[str],
    run_job: Callable[[Job, str], Result],
    on_finish: Callable[[GPUJobOutcome[Job, Result]], None] | None = None,
) -> GPUScheduleSummary[Job, Result]:
    """Run at most one job on each GPU and give freed GPUs the next queued job.

    A job failure stops workers from claiming more jobs. Jobs already running on
    other GPU tokens are allowed to finish and are included in the summary.
    """
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    pending: queue.Queue[tuple[int, Job]] = queue.Queue()
    for index, job in enumerate(jobs):
        pending.put((index, job))

    claim_lock = threading.Lock()
    stop_claiming = threading.Event()
    claimed: set[int] = set()
    outcome_lock = threading.Lock()
    outcomes: list[GPUJobOutcome[Job, Result]] = []

    def worker(gpu: str) -> None:
        while True:
            with claim_lock:
                if stop_claiming.is_set():
                    return
                try:
                    index, job = pending.get_nowait()
                except queue.Empty:
                    return
                claimed.add(index)
            try:
                outcome = GPUJobOutcome(index=index, job=job, gpu=gpu, result=run_job(job, gpu))
            except Exception as exc:
                outcome = GPUJobOutcome(index=index, job=job, gpu=gpu, error=exc)
                with claim_lock:
                    stop_claiming.set()
            with outcome_lock:
                outcomes.append(outcome)
            if on_finish is not None:
                on_finish(outcome)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(worker, str(gpu)) for gpu in gpu_ids]
        for future in futures:
            future.result()

    outcomes.sort(key=lambda outcome: outcome.index)
    not_started = tuple((index, job) for index, job in enumerate(jobs) if index not in claimed)
    return GPUScheduleSummary(tuple(outcomes), not_started)
