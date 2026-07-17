# ADR-0001: PostgreSQL-backed resumable Wikimedia evidence pipeline

- Status: Superseded by ADR-0002
- Date: 2026-07-16

This is a historical record of the retired V1 implementation. It does not
describe supported application code, schemas, configuration, or CLI behavior.
The active product architecture is defined by
[ADR-0002](0002-audience-intelligence-v2-cluster-first-country-trends.md).

## Context

Wikimedia Attention Acquisition currently fetches remote evidence and transforms
it into Alias Traffic and Canonical Articles within one synchronous operation.
Fetching already performs concurrent requests, but an interrupted run cannot
resume completed work. Fetching failures and transformation failures are also
harder to isolate because both responsibilities share one implementation.

The pipeline needs stronger debugging locality and crash resumability without
making ordinary callers orchestrate acquisition phases. Canonical Article
formation must never run against an incomplete Candidate Universe or an
unfinished set of aliases.

## Decision

Split the internal Wikimedia pipeline into two deep modules with distinct
interfaces:

1. **Wikimedia Evidence Fetching** completes discovery and retrieves raw
   Pageviews and metadata evidence, returning an immutable typed terminal
   evidence set only after all required fetching work is terminal.
2. **Wikimedia Attention Transformation** synchronously consumes that terminal
   evidence set in memory, derives Alias Traffic, and forms Canonical Articles
   deterministically without database access or Transformation Jobs.

Keep Wikimedia Attention Acquisition as the preferred interface for ordinary
callers. It hides Fetching followed by synchronous Transformation from callers;
only Fetching launches an in-process worker pool.

Use PostgreSQL as the durable source of fetched evidence:

- Evidence Job state and raw Wikimedia evidence are stored in PostgreSQL.
- Raw response payloads use `JSONB`.
- Jobs are scoped to a stable `run_id`.
- Each job has an idempotent key based on its run, operation, and subject.
- Workers claim jobs atomically with expiring leases.
- Expired claims can be recovered after worker failure.
- Attempts, terminal failures, and completion are persisted.
- Resuming a run reuses its recorded Effective Run Configuration and skips
  completed work.

Evidence Jobs represent fetching work only: discovery, Pageviews, and metadata.
Their execution owns atomic claims, expiring leases, bounded retries, recovery,
and terminal barriers. Wikimedia Evidence Fetching owns phase ordering and runs
an in-process worker pool. It aborts on terminal discovery failure because that
would leave an incomplete Candidate Universe; terminal Pageviews or metadata
failures remain typed evidence and produce publishable degradation.

After every alias in the complete Candidate Universe has terminal Pageviews and
metadata evidence, fetching loads an immutable typed projection from PostgreSQL.
Wikimedia Attention Transformation consumes that value synchronously and in
memory. If the process stops before publication, transformation is replayed from
the persisted fetched evidence rather than resumed through Transformation Jobs.

Run Publication owns final artifact paths and projections. Fetching retains
logical evidence identity without embedding filesystem layout, and
Transformation preserves that identity in its in-memory result. Publication is
idempotent for a `run_id` and exposes at most one completed artifact directory
for that run.

The PostgreSQL queue covers Wikimedia fetching only. Deterministic Wikimedia
Attention Transformation and LLM classification remain synchronous until cost,
throughput, or recovery demonstrates a need for another job mechanism.

## Configuration

Effective Run Configuration is resolved exactly once at startup.

- Exported shell values take precedence over `.env` values.
- `.env` values take precedence over global defaults.
- Normal CLI runs require live LLM credentials and fail at startup when they are
  absent.
- Explicit fixture mode is limited to tests and CI.
- `run.py` remains the composition root and constructs concrete adapters from
  resolved configuration.
- Artifacts record only safe normalized provenance, such as model name and
  adapter modes. Secrets, local fixture paths, `.env` paths, and unsafe URL
  details are never published.

## Testing

- Evidence Job claiming, lease expiry, uniqueness, barriers, recovery, and
  resume behavior are tested against real PostgreSQL in an isolated test
  database or schema.
- Wikimedia Evidence Fetching is tested through its interface with real
  PostgreSQL and the fixture Wikimedia adapter.
- Wikimedia Attention Transformation is tested in memory through its interface
  using typed terminal evidence; it has no database or worker test dependency.
- Existing live and fixture Wikimedia adapters remain a real adapter seam.
- End-to-end tests retain the preferred Wikimedia Attention Acquisition
  interface.
- Configuration precedence and validation are tested through the Effective Run
  Configuration interface rather than adapter internals.

## Consequences

### Positive

- Interrupted runs resume instead of refetching completed evidence.
- Fetching and transformation bugs have separate locality.
- Deterministic transformation can be replayed without persisted job state.
- Raw evidence can be inspected and replayed without another Wikimedia request.
- Worker deployment can later become independent without changing the
  PostgreSQL seam.
- Canonical Article formation retains completeness guarantees.

### Costs

- PostgreSQL becomes required runtime infrastructure.
- Job leases, recovery, migrations, and run lifecycle require explicit
  operational policy.
- Idempotent publication must coordinate filesystem state with persisted run
  state.
- The immutable typed terminal evidence representation becomes a maintained
  contract between the two modules.

## Alternatives considered

### Keep the synchronous module

Rejected because it does not provide crash resumability or a durable debugging
seam between remote evidence and deterministic transformation.

### In-memory queue

Rejected because process failure loses job state and completed work.

### SQLite

Rejected because PostgreSQL is the selected persistence adapter and better
supports multiple workers claiming jobs concurrently.

### External message broker

Rejected for now because it adds infrastructure without improving the required
run-scoped evidence persistence. PostgreSQL provides both durable jobs and
transactional evidence storage.

### Durable versioned evidence journal as a public format

Deferred. The intermediate evidence stays internal until saved-run replay or an
external consumer proves the need for a public versioned format.
