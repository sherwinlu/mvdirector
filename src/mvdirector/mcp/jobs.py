# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Thread-safe background job manager for long-running MCP tool executions."""

from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mvdirector.errors import MVDError, UnknownIdError
from mvdirector.generation.runner import RunEvent

log = logging.getLogger(__name__)

_MAX_EVENTS_PER_JOB = 200
_MAX_RETAINED_JOBS = 100
_MAX_CONCURRENT_JOBS = 8
_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})


class JobCancelled(MVDError):
    """Raised inside a cooperative progress callback when a background job is stopped."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_run_event(event: RunEvent) -> dict[str, Any]:
    """Convert a `RunEvent` into a compact, JSON-serializable dictionary."""
    data: dict[str, Any] = {"kind": event.kind}
    if event.asset:
        data["asset"] = event.asset
    if event.shot_id:
        data["shot_id"] = event.shot_id
    if event.attempt:
        data["attempt"] = event.attempt
    if event.strategy:
        data["strategy"] = event.strategy
    if event.passed is not None:
        data["passed"] = event.passed
    if event.score is not None:
        data["score"] = round(event.score, 4)
    if event.issues:
        data["issues"] = list(event.issues)
    if event.error:
        data["error"] = event.error
    if event.generations_used:
        data["generations_used"] = event.generations_used
    if event.message:
        data["message"] = event.message
    if event.active_checks:
        data["active_checks"] = list(event.active_checks)
    if event.skipped_checks:
        data["skipped_checks"] = [
            {"check": name, "reason": reason} for name, reason in event.skipped_checks
        ]
    return data


@dataclass
class JobRecord:
    """Mutable state for one background MCP job (always accessed under JobManager._lock)."""

    job_id: str
    stage: str
    project_dir: str
    status: str = "queued"  # queued | running | succeeded | failed | cancelled
    created_at: str = field(default_factory=_utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    cancel_requested: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=_MAX_EVENTS_PER_JOB), repr=False
    )
    event_count: int = 0
    generations_used: int = 0
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None

    def snapshot(self, *, event_limit: int = 20) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the job's current state."""
        recent = list(self.events)[-event_limit:] if event_limit > 0 else []
        data: dict[str, Any] = {
            "job_id": self.job_id,
            "stage": self.stage,
            "project_dir": self.project_dir,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancel_requested": self.cancel_requested,
            "event_count": self.event_count,
            "generations_used": self.generations_used,
        }
        if self.progress:
            data["progress"] = dict(self.progress)
        if recent:
            data["recent_events"] = recent
        if self.result is not None:
            data["result"] = self.result
        if self.error is not None:
            data["error"] = self.error
        return data


class JobContext:
    """Handle passed to a background worker so it can emit events and check cancellation."""

    def __init__(self, manager: JobManager, job_id: str, cancel_event: threading.Event) -> None:
        self._manager = manager
        self._job_id = job_id
        self._cancel_event = cancel_event

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise JobCancelled("Job cancelled by request.")

    def on_run_event(self, event: RunEvent) -> None:
        self._manager._record_event(self._job_id, event)
        # Check cancellation only before starting work, never during `run_finished` cleanup.
        if event.kind in ("phase_started", "attempt_started") and self._cancel_event.is_set():
            raise JobCancelled("Job cancelled by request.")

    def set_progress(self, **fields: Any) -> None:
        self._manager._update_progress(self._job_id, fields)


class JobManager:
    """Runs long pipeline stages on daemon worker threads with progress & cooperative cancel."""

    def __init__(
        self,
        *,
        max_retained: int = _MAX_RETAINED_JOBS,
        max_concurrent: int = _MAX_CONCURRENT_JOBS,
    ) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._order: deque[str] = deque()
        self._threads: dict[str, threading.Thread] = {}
        self._max_retained = max(1, max_retained)
        self._max_concurrent = max(1, max_concurrent)

    def submit(
        self,
        stage: str,
        project_dir: str,
        work: Callable[[JobContext], Any],
    ) -> dict[str, Any]:
        """Start ``work(ctx)`` on a background thread and return the initial job snapshot."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        record = JobRecord(job_id=job_id, stage=stage, project_dir=project_dir)
        ctx = JobContext(self, job_id, record.cancel_event)

        def _runner() -> None:
            with self._lock:
                if record.cancel_event.is_set():
                    record.status = "cancelled"
                    record.finished_at = _utc_now_iso()
                    return
                record.status = "running"
                record.started_at = _utc_now_iso()
            try:
                outcome = work(ctx)
            except JobCancelled as exc:
                with self._lock:
                    record.status = "cancelled"
                    record.error = str(exc)
                    record.finished_at = _utc_now_iso()
            except MVDError as exc:
                log.warning("Background MCP job %s (%s) failed: %s", job_id, stage, exc)
                with self._lock:
                    record.status = "failed"
                    record.error = str(exc)
                    record.finished_at = _utc_now_iso()
            except Exception:
                log.exception("Background MCP job %s (%s) failed", job_id, stage)
                with self._lock:
                    record.status = "failed"
                    record.error = "Internal server error."
                    record.finished_at = _utc_now_iso()
            else:
                with self._lock:
                    record.status = "cancelled" if record.cancel_event.is_set() else "succeeded"
                    record.result = outcome
                    record.finished_at = _utc_now_iso()

        thread = threading.Thread(target=_runner, name=f"mvd-mcp-{job_id}", daemon=True)
        with self._lock:
            active = sum(1 for rec in self._jobs.values() if rec.status not in _TERMINAL_STATES)
            if active >= self._max_concurrent:
                raise MVDError(
                    f"Too many active background jobs ({active}/{self._max_concurrent}). "
                    "Wait for running jobs to finish or stop one with mvd_task_stop."
                )
            self._jobs[job_id] = record
            self._order.append(job_id)
            self._threads[job_id] = thread
            self._prune_locked()
            initial = record.snapshot(event_limit=0)
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._jobs.pop(job_id, None)
                self._threads.pop(job_id, None)
                self._order = deque(jid for jid in self._order if jid != job_id)
            raise
        return initial

    def get(self, job_id: str, *, event_limit: int = 20) -> dict[str, Any]:
        """Return a snapshot of one job, raising ``KeyError`` if unknown."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise UnknownIdError(f"Unknown job_id: {job_id!r}")
            return record.snapshot(event_limit=event_limit)

    def list_jobs(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """Return compact summaries of all retained jobs, newest first."""
        with self._lock:
            items = [self._jobs[jid] for jid in reversed(self._order) if jid in self._jobs]
            if status is not None:
                items = [j for j in items if j.status == status]
            return [j.snapshot(event_limit=0) for j in items]

    def stop(self, job_id: str) -> dict[str, Any]:
        """Request cooperative cancellation of a queued or running job."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise UnknownIdError(f"Unknown job_id: {job_id!r}")
            if record.status not in _TERMINAL_STATES:
                record.cancel_requested = True
                record.cancel_event.set()
            return record.snapshot(event_limit=5)

    def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Block until ``job_id`` finishes (or ``timeout`` elapses) and return its snapshot."""
        with self._lock:
            thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout=timeout)
        return self.get(job_id)

    def _record_event(self, job_id: str, event: RunEvent) -> None:
        serialized = serialize_run_event(event)
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.events.append(serialized)
            record.event_count += 1
            if event.generations_used or event.kind == "run_finished":
                # run_finished carries the final count, after slots refunded by a cancel.
                record.generations_used = event.generations_used

    def _update_progress(self, job_id: str, fields: dict[str, Any]) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.progress.update(fields)

    def _prune_locked(self) -> None:
        if len(self._order) <= self._max_retained:
            return
        excess = len(self._order) - self._max_retained
        kept: deque[str] = deque()
        for jid in self._order:
            record = self._jobs.get(jid)
            if excess > 0 and (record is None or record.status in _TERMINAL_STATES):
                self._jobs.pop(jid, None)
                self._threads.pop(jid, None)
                excess -= 1
            else:
                kept.append(jid)
        self._order = kept
