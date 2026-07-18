from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import random
import time
from typing import Any, Callable
from uuid import UUID

import psycopg


# Represent a leased unit of resumable Wikimedia acquisition work.
@dataclass(frozen=True)
class EvidenceJob:
    id: UUID
    run_id: str
    operation: str
    subject: str
    status: str
    attempts: int
    claimed_by: str | None
    claim_token: UUID | None
    evidence: Any | None = None
    error: str | None = None


# Record the payload produced by a completed evidence job.
@dataclass(frozen=True)
class CompletedEvidence:
    operation: str
    subject: str
    evidence: Any
    attempts: int


# Record a terminal job failure without losing its audit trail.
@dataclass(frozen=True)
class FailedEvidence:
    operation: str
    subject: str
    attempts: int
    reason: str


TerminalEvidence = CompletedEvidence | FailedEvidence
COUNTRY_DAY_OPERATION = "country-day"
METADATA_BATCH_OPERATION = "metadata-batch"


# Persist idempotent jobs, leases, retries, and completion barriers in PostgreSQL.
class EvidenceJobStore:
    # Initialize the EvidenceJobStore.
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    # Create or upgrade the evidence job tables.
    def migrate(self) -> None:
        # Migrations are additive so older resumable databases remain usable.
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_jobs (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    run_id text NOT NULL,
                    operation text NOT NULL,
                    subject text NOT NULL,
                    status text NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','claimed','completed','failed')),
                    attempts integer NOT NULL DEFAULT 0,
                    claimed_by text,
                    claim_token uuid,
                    lease_expires_at timestamptz,
                    evidence jsonb,
                    error text,
                    UNIQUE (run_id, operation, subject)
                )
                """
            )
            connection.execute(
                "ALTER TABLE evidence_jobs ADD COLUMN IF NOT EXISTS claim_token uuid"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS acquisition_runs (
                    run_id text PRIMARY KEY,
                    configuration jsonb NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    published_path text,
                    publication_complete boolean NOT NULL DEFAULT false
                )"""
            )
            connection.execute(
                "ALTER TABLE acquisition_runs ADD COLUMN IF NOT EXISTS published_path text"
            )
            connection.execute(
                """ALTER TABLE acquisition_runs ADD COLUMN IF NOT EXISTS
                   publication_complete boolean NOT NULL DEFAULT false"""
            )

    # Create a run once or reject effective-configuration drift on resume.
    def ensure_run(self, run_id: str, configuration: dict[str, str]) -> None:
        # Conflict updates return the original configuration for an exact drift check.
        with psycopg.connect(self.database_url) as connection:
            row = connection.execute(
                """INSERT INTO acquisition_runs (run_id, configuration)
                   VALUES (%s, %s::jsonb)
                   ON CONFLICT (run_id) DO UPDATE SET run_id = EXCLUDED.run_id
                   RETURNING configuration""",
                (run_id, json.dumps(configuration)),
            ).fetchone()
        if row[0] != configuration:
            raise ValueError("resumed run configuration does not match recorded facts")

    # Schedule country days.
    def schedule_country_days(self, run_id: str, subjects: tuple[str, ...]) -> None:
        self._schedule(run_id, COUNTRY_DAY_OPERATION, subjects)

    # Schedule metadata batches.
    def schedule_metadata_batches(self, run_id: str, subjects: tuple[str, ...]) -> None:
        self._schedule(run_id, METADATA_BATCH_OPERATION, subjects)

    # Insert missing jobs without duplicating existing work.
    def _schedule(
        self, run_id: str, operation: str, subjects: tuple[str, ...]
    ) -> None:
        # The unique run/operation/subject key makes repeated scheduling idempotent.
        with psycopg.connect(self.database_url) as connection:
            connection.cursor().executemany(
                """INSERT INTO evidence_jobs (run_id, operation, subject)
                   VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
                ((run_id, operation, subject) for subject in subjects),
            )

    # Reserve publication path.
    def reserve_publication_path(self, run_id: str, path: str) -> None:
        with psycopg.connect(self.database_url) as connection:
            row = connection.execute(
                """UPDATE acquisition_runs
                   SET published_path = COALESCE(published_path, %s)
                   WHERE run_id = %s RETURNING published_path""",
                (path, run_id),
            ).fetchone()
        if row is None or row[0] != path:
            raise ValueError("run was already published to a different path")

    # Mark publication complete.
    def mark_publication_complete(self, run_id: str, path: str) -> None:
        with psycopg.connect(self.database_url) as connection:
            updated = connection.execute(
                """UPDATE acquisition_runs SET publication_complete = true
                   WHERE run_id = %s AND published_path = %s""",
                (run_id, path),
            ).rowcount
        if updated != 1:
            raise ValueError("publication path was not reserved for this run")

    # Atomically claim the next available job.
    def claim(
        self,
        worker: str,
        *,
        lease_seconds: int,
        run_id: str | None = None,
        operations: tuple[str, ...] | None = None,
        max_attempts: int = 3,
    ) -> EvidenceJob | None:
        # Expired leases at the attempt cap become terminal before new work is chosen.
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """UPDATE evidence_jobs SET status = 'failed',
                          error = 'lease attempts exhausted', claimed_by = NULL,
                          claim_token = NULL, lease_expires_at = NULL
                   WHERE status = 'claimed' AND lease_expires_at <= now()
                     AND attempts >= %s""",
                (max_attempts,),
            )
            # SKIP LOCKED lets concurrent workers claim distinct jobs without blocking.
            row = connection.execute(
                """
                WITH available AS (
                    SELECT id FROM evidence_jobs
                    WHERE (status = 'pending'
                       OR (status = 'claimed' AND lease_expires_at <= now()))
                      AND (%s::text IS NULL OR run_id = %s::text)
                      AND (%s::text[] IS NULL OR operation = ANY(%s::text[]))
                    ORDER BY run_id, operation, subject
                    FOR UPDATE SKIP LOCKED LIMIT 1
                )
                UPDATE evidence_jobs AS jobs
                SET status = 'claimed', claimed_by = %s,
                    lease_expires_at = now() + (%s * interval '1 second'),
                    attempts = attempts + 1, claim_token = gen_random_uuid()
                FROM available WHERE jobs.id = available.id
                RETURNING jobs.id, jobs.run_id, jobs.operation, jobs.subject,
                          jobs.status, jobs.attempts, jobs.claimed_by, jobs.claim_token
                """,
                (run_id, run_id, list(operations) if operations else None,
                 list(operations) if operations else None, worker, lease_seconds),
            ).fetchone()
        # A fresh claim token prevents stale workers from completing reclaimed work.
        return EvidenceJob(*row) if row else None

    # Mark a claimed job as completed.
    def complete(self, job: EvidenceJob, evidence: object) -> None:
        self._finish(job, "completed", evidence=evidence)

    # Mark a claimed job as failed or retryable.
    def fail(self, job: EvidenceJob, reason: str, *, terminal: bool) -> None:
        # Retryable failures return to pending; terminal failures remain audit evidence.
        with psycopg.connect(self.database_url) as connection:
            updated = connection.execute(
                """UPDATE evidence_jobs SET status = %s, error = %s,
                          claimed_by = NULL, claim_token = NULL, lease_expires_at = NULL
                   WHERE id = %s AND claim_token = %s AND status = 'claimed'""",
                ("failed" if terminal else "pending", reason, job.id, job.claim_token),
            ).rowcount
        if updated != 1:
            raise RuntimeError("evidence job lease was lost before failure update")

    # Persist a terminal job result for the active claim.
    def _finish(self, job: EvidenceJob, status: str, *, evidence: object) -> None:
        # Completion succeeds only for the worker holding the current lease token.
        with psycopg.connect(self.database_url) as connection:
            updated = connection.execute(
                """UPDATE evidence_jobs SET status = %s, evidence = %s::jsonb,
                          claimed_by = NULL, claim_token = NULL, lease_expires_at = NULL
                   WHERE id = %s AND claim_token = %s AND status = 'claimed'""",
                (status, json.dumps(evidence), job.id, job.claim_token),
            ).rowcount
        if updated != 1:
            raise RuntimeError("evidence job lease was lost before completion")

    # Check whether all jobs of a kind reached a terminal state.
    def barrier_reached(self, run_id: str, operations: tuple[str, ...]) -> bool:
        with psycopg.connect(self.database_url) as connection:
            unfinished = connection.execute(
                """SELECT count(*) FROM evidence_jobs
                   WHERE run_id = %s AND operation = ANY(%s::text[])
                     AND status NOT IN ('completed','failed')""",
                (run_id, list(operations)),
            ).fetchone()[0]
        return unfinished == 0

    # Return terminal results once a job-kind barrier is reached.
    def results_at_barrier(
        self, run_id: str, operations: tuple[str, ...]
    ) -> tuple[TerminalEvidence, ...]:
        if not self.barrier_reached(run_id, operations):
            raise RuntimeError("evidence results requested before the run barrier")
        return self._terminal_evidence(run_id, operations)

    # Return all terminal job results of a kind.
    def terminal_results(
        self, run_id: str, operations: tuple[str, ...]
    ) -> tuple[TerminalEvidence, ...]:
        return self._terminal_evidence(run_id, operations)

    # Load terminal evidence for a job kind.
    def _terminal_evidence(
        self,
        run_id: str,
        operations: tuple[str, ...],
        *,
        subject: str | None = None,
    ) -> tuple[TerminalEvidence, ...]:
        with psycopg.connect(self.database_url) as connection:
            rows = connection.execute(
                """SELECT operation, subject, status, attempts, evidence, error
                   FROM evidence_jobs
                   WHERE run_id = %s AND operation = ANY(%s::text[])
                     AND status IN ('completed', 'failed')
                     AND (%s::text IS NULL OR subject = %s::text)
                   ORDER BY operation, subject""",
                (run_id, list(operations), subject, subject),
            ).fetchall()
        return tuple(
            CompletedEvidence(operation, item_subject, evidence, attempts)
            if status == "completed"
            else FailedEvidence(operation, item_subject, attempts, error or "failed")
            for operation, item_subject, status, attempts, evidence, error in rows
        )

    # Clear for tests.
    def clear_for_tests(self) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute("TRUNCATE evidence_jobs")
            connection.execute("TRUNCATE acquisition_runs")


# Execute claimed jobs and report bounded progress at barriers.
class EvidenceJobExecution:
    """Run fetch Evidence Jobs while hiding lifecycle policy from callers."""

    # Initialize the EvidenceJobExecution.
    def __init__(
        self,
        store: EvidenceJobStore,
        *,
        max_attempts: int = 3,
        lease_seconds: int = 60,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        honor_retry_after: bool = False,
    ) -> None:
        self.store = store
        self.max_attempts = max_attempts
        self.lease_seconds = lease_seconds
        self.sleep = sleep
        self.jitter = jitter
        self.honor_retry_after = honor_retry_after

    # Process jobs until the selected barrier is reached.
    def drain(
        self,
        run_id: str,
        operations: tuple[str, ...],
        handler: Callable[[EvidenceJob], object],
        *,
        workers: int,
        is_terminal_error: Callable[[Exception], bool],
        on_terminal: Callable[[EvidenceJob], None] | None = None,
    ) -> None:
        # Workers stop only when every selected operation reaches a terminal barrier.
        # Execute one claimed evidence job.
        def work(worker_number: int) -> None:
            while not self.store.barrier_reached(run_id, operations):
                job = self.store.claim(
                    f"fetcher-{worker_number}",
                    lease_seconds=self.lease_seconds,
                    run_id=run_id,
                    operations=operations,
                    max_attempts=self.max_attempts,
                )
                # A temporary empty claim can occur while another worker holds a lease.
                if job is None:
                    self.sleep(0.01)
                    continue
                try:
                    self.store.complete(job, handler(job))
                except Exception as error:
                    # Retry policy is centralized here so handlers contain domain work only.
                    terminal = (
                        is_terminal_error(error) or job.attempts >= self.max_attempts
                    )
                    self.store.fail(job, str(error), terminal=terminal)
                    if terminal and on_terminal:
                        on_terminal(job)
                    if not terminal and not getattr(error, "retry_immediately", False):
                        retry_after = getattr(error, "retry_after_seconds", None)
                        if self.honor_retry_after and retry_after is not None:
                            self.sleep(retry_after)
                        else:
                            delay = 2 ** (job.attempts - 1)
                            self.sleep(delay + self.jitter(0, delay))
                else:
                    if on_terminal:
                        on_terminal(job)

        # The durable claim query, not thread identity, coordinates parallel workers.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tuple(executor.map(work, range(workers)))
