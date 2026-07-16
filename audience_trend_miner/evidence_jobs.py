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

    def enqueue(self, run_id: str, operation: str, subject: str) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """INSERT INTO evidence_jobs (run_id, operation, subject)
                   VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
                (run_id, operation, subject),
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

    def jobs(self, run_id: str) -> tuple[EvidenceJob, ...]:
        with psycopg.connect(self.database_url) as connection:
            rows = connection.execute(
                """SELECT id, run_id, operation, subject, status, attempts,
                          claimed_by, claim_token, evidence, error
                   FROM evidence_jobs WHERE run_id = %s
                   ORDER BY operation, subject""",
                (run_id,),
            ).fetchall()
        return tuple(EvidenceJob(*row) for row in rows)

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

    def run_jobs_terminal(self, run_id: str, operation: str) -> bool:
        with psycopg.connect(self.database_url) as connection:
            count = connection.execute(
                """SELECT count(*) FROM evidence_jobs
                   WHERE run_id = %s AND operation = %s
                     AND status NOT IN ('completed','failed')""",
                (run_id, operation),
            ).fetchone()[0]
        return count == 0

    def completed_evidence(
        self, run_id: str, operation: str
    ) -> tuple[tuple[str, Any], ...]:
        with psycopg.connect(self.database_url) as connection:
            rows = connection.execute(
                """SELECT subject, evidence FROM evidence_jobs
                   WHERE run_id = %s AND operation = %s AND status = 'completed'
                   ORDER BY subject""",
                (run_id, operation),
            ).fetchall()
        return tuple((subject, evidence) for subject, evidence in rows)

    def clear_for_tests(self) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute("TRUNCATE evidence_jobs")
            connection.execute("TRUNCATE acquisition_runs")
