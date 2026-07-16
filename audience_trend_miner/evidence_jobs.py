from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from uuid import UUID

import psycopg


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


@dataclass(frozen=True)
class CompletedEvidence:
    operation: str
    subject: str
    evidence: Any
    attempts: int


@dataclass(frozen=True)
class FailedEvidence:
    operation: str
    subject: str
    attempts: int
    reason: str


TerminalEvidence = CompletedEvidence | FailedEvidence


@dataclass(frozen=True)
class ReadyTransformation:
    job: EvidenceJob
    pageviews: TerminalEvidence
    metadata: TerminalEvidence


class EvidenceJobStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def migrate(self) -> None:
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

    def ensure_run(self, run_id: str, configuration: dict[str, str]) -> None:
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

    def schedule_discovery(self, run_id: str, subjects: tuple[str, ...]) -> None:
        self._schedule(run_id, "discovery", subjects)

    def schedule_alias_evidence(
        self, run_id: str, subjects: tuple[str, ...]
    ) -> None:
        self._schedule(run_id, "pageviews", subjects)
        self._schedule(run_id, "metadata", subjects)

    def _schedule(
        self, run_id: str, operation: str, subjects: tuple[str, ...]
    ) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.cursor().executemany(
                """INSERT INTO evidence_jobs (run_id, operation, subject)
                   VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
                ((run_id, operation, subject) for subject in subjects),
            )

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

    def mark_publication_complete(self, run_id: str, path: str) -> None:
        with psycopg.connect(self.database_url) as connection:
            updated = connection.execute(
                """UPDATE acquisition_runs SET publication_complete = true
                   WHERE run_id = %s AND published_path = %s""",
                (run_id, path),
            ).rowcount
        if updated != 1:
            raise ValueError("publication path was not reserved for this run")

    def claim(
        self,
        worker: str,
        *,
        lease_seconds: int,
        run_id: str | None = None,
        operations: tuple[str, ...] | None = None,
    ) -> EvidenceJob | None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """UPDATE evidence_jobs SET status = 'failed',
                          error = 'lease attempts exhausted', claimed_by = NULL,
                          claim_token = NULL, lease_expires_at = NULL
                   WHERE status = 'claimed' AND lease_expires_at <= now()
                     AND attempts >= 3"""
            )
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
        return EvidenceJob(*row) if row else None

    def claim_ready_transformation(
        self, worker: str, *, lease_seconds: int, run_id: str
    ) -> ReadyTransformation | None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """UPDATE evidence_jobs SET status = 'failed',
                          error = 'lease attempts exhausted', claimed_by = NULL,
                          claim_token = NULL, lease_expires_at = NULL
                   WHERE run_id = %s AND operation = 'transform'
                     AND status = 'claimed' AND lease_expires_at <= now()
                     AND attempts >= 3""",
                (run_id,),
            )
            connection.execute(
                """INSERT INTO evidence_jobs (run_id, operation, subject)
                   SELECT pageviews.run_id, 'transform', pageviews.subject
                   FROM evidence_jobs AS pageviews
                   JOIN evidence_jobs AS metadata
                     ON metadata.run_id = pageviews.run_id
                    AND metadata.subject = pageviews.subject
                    AND metadata.operation = 'metadata'
                   WHERE pageviews.run_id = %s
                     AND pageviews.operation = 'pageviews'
                     AND pageviews.status IN ('completed', 'failed')
                     AND metadata.status IN ('completed', 'failed')
                   ON CONFLICT DO NOTHING""",
                (run_id,),
            )
            row = connection.execute(
                """WITH available AS (
                       SELECT transformation.id
                       FROM evidence_jobs AS transformation
                       JOIN evidence_jobs AS pageviews
                         ON pageviews.run_id = transformation.run_id
                        AND pageviews.subject = transformation.subject
                        AND pageviews.operation = 'pageviews'
                       JOIN evidence_jobs AS metadata
                         ON metadata.run_id = transformation.run_id
                        AND metadata.subject = transformation.subject
                        AND metadata.operation = 'metadata'
                       WHERE transformation.run_id = %s
                         AND transformation.operation = 'transform'
                         AND (transformation.status = 'pending'
                           OR (transformation.status = 'claimed'
                             AND transformation.lease_expires_at <= now()))
                         AND pageviews.status IN ('completed', 'failed')
                         AND metadata.status IN ('completed', 'failed')
                       ORDER BY transformation.subject
                       FOR UPDATE OF transformation SKIP LOCKED LIMIT 1
                   )
                   UPDATE evidence_jobs AS jobs
                   SET status = 'claimed', claimed_by = %s,
                       lease_expires_at = now() + (%s * interval '1 second'),
                       attempts = attempts + 1, claim_token = gen_random_uuid()
                   FROM available WHERE jobs.id = available.id
                   RETURNING jobs.id, jobs.run_id, jobs.operation, jobs.subject,
                             jobs.status, jobs.attempts, jobs.claimed_by,
                             jobs.claim_token""",
                (run_id, worker, lease_seconds),
            ).fetchone()
        job = EvidenceJob(*row) if row else None
        if job is None:
            return None
        dependencies = self._terminal_evidence(
            run_id, ("pageviews", "metadata"), subject=job.subject
        )
        indexed = {item.operation: item for item in dependencies}
        return ReadyTransformation(job, indexed["pageviews"], indexed["metadata"])

    def complete(self, job: EvidenceJob, evidence: object) -> None:
        self._finish(job, "completed", evidence=evidence)

    def fail(self, job: EvidenceJob, reason: str, *, terminal: bool) -> None:
        with psycopg.connect(self.database_url) as connection:
            updated = connection.execute(
                """UPDATE evidence_jobs SET status = %s, error = %s,
                          claimed_by = NULL, claim_token = NULL, lease_expires_at = NULL
                   WHERE id = %s AND claim_token = %s AND status = 'claimed'""",
                ("failed" if terminal else "pending", reason, job.id, job.claim_token),
            ).rowcount
        if updated != 1:
            raise RuntimeError("evidence job lease was lost before failure update")

    def recover_incomplete_pageviews(
        self,
        transformation: EvidenceJob,
        reason: str,
    ) -> None:
        """Return a transformation and its invalid upstream evidence to schedulable states."""
        with psycopg.connect(self.database_url) as connection:
            transformed = connection.execute(
                """UPDATE evidence_jobs SET status = 'pending', error = NULL,
                          claimed_by = NULL, claim_token = NULL, lease_expires_at = NULL
                   WHERE id = %s AND claim_token = %s AND status = 'claimed'""",
                (transformation.id, transformation.claim_token),
            ).rowcount
            evidence = connection.execute(
                """UPDATE evidence_jobs
                   SET status = CASE WHEN attempts >= 3 THEN 'failed' ELSE 'pending' END,
                       evidence = NULL, error = %s,
                       claimed_by = NULL, claim_token = NULL, lease_expires_at = NULL
                   WHERE run_id = %s AND operation = 'pageviews' AND subject = %s
                     AND status = 'completed'""",
                (reason, transformation.run_id, transformation.subject),
            ).rowcount
            if transformed != 1 or evidence != 1:
                raise RuntimeError(
                    "incomplete evidence changed before retry transition"
                )

    def _finish(self, job: EvidenceJob, status: str, *, evidence: object) -> None:
        with psycopg.connect(self.database_url) as connection:
            updated = connection.execute(
                """UPDATE evidence_jobs SET status = %s, evidence = %s::jsonb,
                          claimed_by = NULL, claim_token = NULL, lease_expires_at = NULL
                   WHERE id = %s AND claim_token = %s AND status = 'claimed'""",
                (status, json.dumps(evidence), job.id, job.claim_token),
            ).rowcount
        if updated != 1:
            raise RuntimeError("evidence job lease was lost before completion")

    def barrier_reached(self, run_id: str, operations: tuple[str, ...]) -> bool:
        with psycopg.connect(self.database_url) as connection:
            unfinished = connection.execute(
                """SELECT count(*) FROM evidence_jobs
                   WHERE run_id = %s AND operation = ANY(%s::text[])
                     AND status NOT IN ('completed','failed')""",
                (run_id, list(operations)),
            ).fetchone()[0]
            missing_transformations = 0
            if "transform" in operations:
                missing_transformations = connection.execute(
                    """SELECT
                           (SELECT count(*) FROM evidence_jobs
                            WHERE run_id = %s AND operation = 'pageviews')
                         - (SELECT count(*) FROM evidence_jobs
                            WHERE run_id = %s AND operation = 'transform')""",
                    (run_id, run_id),
                ).fetchone()[0]
        return unfinished == 0 and missing_transformations == 0

    def results_at_barrier(
        self, run_id: str, operations: tuple[str, ...]
    ) -> tuple[TerminalEvidence, ...]:
        if not self.barrier_reached(run_id, operations):
            raise RuntimeError("evidence results requested before the run barrier")
        return self._terminal_evidence(run_id, operations)

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

    def clear_for_tests(self) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute("TRUNCATE evidence_jobs")
            connection.execute("TRUNCATE acquisition_runs")
