from __future__ import annotations

import threading
import time

from scripts.gpu_scheduler import run_gpu_jobs


def test_gpu_tokens_dynamically_give_fast_gpu_the_next_job() -> None:
    both_started = threading.Barrier(2)
    followup_started = threading.Event()
    lock = threading.Lock()
    active_by_gpu = {"0": 0, "1": 0}
    maximum_by_gpu = {"0": 0, "1": 0}
    slow_job_still_active_at_followup: list[bool] = []
    assignments: dict[str, str] = {}
    jobs = ("fast", "slow", "followup", "last")

    def fake_job(job: str, gpu: str) -> str:
        with lock:
            active_by_gpu[gpu] += 1
            maximum_by_gpu[gpu] = max(maximum_by_gpu[gpu], active_by_gpu[gpu])
            assignments[job] = gpu
        try:
            if job in {"fast", "slow"}:
                both_started.wait(timeout=2)
            if job == "fast":
                time.sleep(0.03)
            elif job == "slow":
                assert followup_started.wait(timeout=2)
            elif job == "followup":
                with lock:
                    slow_job_still_active_at_followup.append(
                        any(name == "slow" and assigned_gpu != gpu for name, assigned_gpu in assignments.items())
                    )
                followup_started.set()
                time.sleep(0.005)
            else:
                time.sleep(0.005)
            return job
        finally:
            with lock:
                active_by_gpu[gpu] -= 1

    summary = run_gpu_jobs(jobs, ["0", "1"], fake_job)

    assert not summary.failures
    assert not summary.not_started
    assert [row.job for row in summary.outcomes] == list(jobs)
    assert maximum_by_gpu == {"0": 1, "1": 1}
    assert slow_job_still_active_at_followup == [True]
    assert assignments["followup"] == assignments["fast"]
    assert assignments["followup"] != assignments["slow"]
