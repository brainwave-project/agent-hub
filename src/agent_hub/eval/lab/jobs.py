"""In-process background jobs — indexing and batch runs outlive an HTTP request.

In-memory on purpose: a job is progress reporting, not a record. Everything a
job produces that matters (index state, runs) is persisted by the job itself,
so losing the registry on restart loses a progress bar, never a result.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from agent_hub.eval.lab.models import now_iso

__all__ = ["Job", "JobRegistry"]


@dataclass
class Job:
    id: str
    kind: str
    label: str = ""
    status: str = "running"  # running | done | failed
    total: int = 0
    done: int = 0
    message: str = ""
    error: str | None = None
    result: dict = field(default_factory=dict)
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None

    def progress(self, done: int, total: int | None = None, message: str = "") -> None:
        self.done = done
        if total is not None:
            self.total = total
        if message:
            self.message = message

    def as_dict(self) -> dict:
        return asdict(self)


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, label: str, work: Callable[[Job], dict]) -> Job:
        job = Job(id=f"job_{uuid.uuid4().hex[:12]}", kind=kind, label=label)
        with self._lock:
            self._jobs[job.id] = job

        def target() -> None:
            try:
                job.result = work(job) or {}
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI verbatim
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = traceback.format_exc(limit=3)
            finally:
                job.finished_at = now_iso()

        threading.Thread(target=target, name=job.id, daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)
