# AudienceIntelligence

Audience Trend Miner turns public attention signals into an Emerging Audience
Portfolio. A run discovers the complete current-window candidate universe from
English Wikipedia, retrieves exact traffic across both analysis windows, and
canonicalizes aliases without fabricating audiences.
It then applies auditable traffic, growth, and trend-score gates and removes only
explicit technical or navigational noise. Qualified articles then pass through a
strict, fail-closed commercial-relevance and brand-safety judgment before they
remain attention signals for later clustering; the CLI does not present them as
audiences.

## Install on macOS

Install Python 3.12, create an isolated environment, and install the project:

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run

Supply an analysis date for a reproducible run:

```bash
audience-trend-miner --as-of 2026-07-16 --output-dir runs
```

Use a stable run identifier to resume interrupted Wikimedia work without
refetching completed evidence:

```bash
audience-trend-miner --as-of 2026-07-16 --run-id july-16 --output-dir runs
```

`--as-of` defaults to the current UTC date. Each invocation creates a
timestamped directory containing:

- `manifest.json` — supplied and effective dates plus resolved windows
- `portfolio.json` — schema-valid Emerging Audience Portfolio
- `report.html` — static report, including a valid empty state
- `audit.json` — run status, decisions, and failures
- `wikimedia/` — saved discovery, Pageviews, metadata, and canonical artifacts
- `classification/article_judgments.json` — prompts, raw model outputs,
  validation results, decisions, and complete attempt histories (when articles
  reach classification)
- `clustering/candidate_clusters.json` — semantic representations, embeddings,
  pairwise cosine values, graph edges, and connected-component membership
- `clustering/refinement.json` — validate/split/reject decisions, exclusive
  accepted membership, alternatives, rejected singletons, retry attempts, and
  independent cluster-level safety vetoes

Run Publication is atomic: artifacts are assembled and validated before being
staged, and the timestamped directory appears only after every file is complete.
Publication failures leave no partial run directory.

Partially successful runs remain publishable and use `status: "degraded"` in
the manifest, audit, and portfolio. Each records the failure count and reasons,
and the HTML report displays the same degradation notice while unaffected
articles and clusters continue through the pipeline.

Discovery uses Wikimedia's public APIs by default. If any daily top-page
response is still unavailable after three attempts, the command exits without
creating a partial run directory.

Wikimedia discovery, Pageviews, and metadata fetching use leased, idempotent
PostgreSQL Evidence Jobs. Set `DATABASE_URL` to the PostgreSQL database used for
run state and raw `JSONB` evidence. After every Candidate Universe alias reaches
a terminal fetching state, deterministic transformation runs synchronously in
memory to form Alias Traffic and Canonical Articles. Interrupted transformation
is safely replayed from the persisted fetched evidence.

Deterministic integration runs can select the fixture adapter with
`AUDIENCE_TREND_MINER_WIKIMEDIA_FIXTURE=/path/to/fixture.json`. The fixture is
one logical dataset containing `discovery`, `pageviews`, and `metadata`; it does
not reproduce Wikimedia URL paths.

The current and previous analysis windows are complete seven-day periods. The
current window ends two days before the effective as-of date.

Article classification calls Groq directly. Set `GROQ_API_KEY` for production
runs. The default model is `openai/gpt-oss-120b`; override it with
`AUDIENCE_TREND_MINER_MODEL`. Deterministic integration runs can instead set
`AUDIENCE_TREND_MINER_CLASSIFICATION_FIXTURE` to a JSON file containing a
`responses` array. Invalid output and request failures are retried three total
times with exponential backoff and jitter, then rejected with their evidence
preserved in the audit.

Surviving classified signals are embedded locally exactly once using
`sentence-transformers/all-mpnet-base-v2`. Override the model with
`AUDIENCE_TREND_MINER_EMBEDDING_MODEL` or the inclusive graph-edge threshold
with `AUDIENCE_TREND_MINER_SIMILARITY_THRESHOLD` (default `0.62`). Connected
components are candidate clusters only. Multi-article components receive a
schema-valid validate, split, or reject decision followed by a separate safety
veto; singleton components remain auditable standalone signals. Refined
audiences remain outside `portfolio.json` until the later scoring and portfolio
assembly stages.

For local development, copy `.env.example` to `.env` and set both values there:

```dotenv
GROQ_API_KEY=your-real-key
AUDIENCE_TREND_MINER_MODEL=openai/gpt-oss-120b
DATABASE_URL=postgresql://localhost/audience_intelligence
```

The `.env` file is ignored by Git. Values exported in the shell take precedence,
and the global `DEFAULT_MODEL` in `configuration.py` remains the final fallback.

## Test

```bash
python -m unittest discover -v
```

Publication tests exercise the complete artifact contract and atomic failure
behavior through the Run Publication interface. Focused CLI subprocess tests
cover argument and date wiring, adapter selection, and Candidate Universe aborts.

The frozen V1 evaluation set lives at
`tests/fixtures/v1_quality_evaluation.json`. `audience_trend_miner.quality`
matches produced decisions to its editor labels and enforces the 80%
commercial-relevance threshold, cluster coherence and safety gates, and
four-of-five top-audience approval gate with recorded reviewer provenance.
The same module independently verifies exact Size Index allocation and complete
alias-to-final-membership lineage in published audit data.
