# AudienceIntelligence

Audience Trend Miner builds a country-specific Audience Portfolio from public
attention signals. The supported application is the V2 cluster-first pipeline:
it groups canonical English Wikipedia pages semantically, adjudicates each
preliminary cluster for commercial meaning and brand safety, attaches censored
United States traffic after membership is final, and publishes robust growing
and shrinking audience trends.

## Install on macOS

Install Python 3.12, create an isolated environment, and install the project:

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Production runs require PostgreSQL-backed Wikimedia Evidence Jobs and Groq:

```bash
export DATABASE_URL=postgresql://localhost/audience_intelligence
export GROQ_API_KEY=your-real-key
```

## Run the pipeline

The default CLI runs Wikimedia Evidence, Semantic Audience Formation, Cluster
Adjudication, Trend Portfolio, and Run Publication in dependency order:

```bash
audience-trend-miner \
  --run-id july-16 \
  --as-of 2026-07-16 \
  --output-dir runs \
  --progress-format human
```

The As-of Date anchors two adjacent seven-day Nominal Windows ending two days
before it. Reusing the same run ID and compatible arguments resumes completed
stage artifacts; configuration drift fails closed. Use
`--progress-format json` for a flushed, schema-versioned event stream.

Each run lives under `runs/<run-id>/`. Run Publication atomically exposes only:

- `publication/portfolio.json` — the UI-facing Audience Portfolio
- `publication/audit.json` — membership, trend, narrative, and integrity evidence
- `publication/manifest.json` — run identity, dates, coverage, and artifact hashes

No static HTML report is produced. The interactive UI renders
`portfolio.json` after validating the complete publication contract.

The five stage commands remain available for manual recovery and experiments:
`v2-wikimedia-evidence`, `v2-semantic-audience-formation`,
`v2-cluster-adjudication`, `v2-trend-portfolio`, and `v2-run-publication`.

## Run the interactive UI

Start the loopback FastAPI server (default `http://127.0.0.1:8000`):

```bash
audience-trend-miner v2-ui --output-dir runs
```

Choose only an As-of Date in the browser. The UI derives the stable run ID as
`run-<YYYY-MM-DD>` and starts or resumes the default global CLI. The backend
owns the subprocess independently of the browser connection, records
validated progress before delivery, and replays only missing event sequences
after reconnect. A confirmed cancellation terminates only that run's owned
subprocess and retains artifacts and event history. Failed or cancelled runs can
resume with the same ID.

## Configuration

Copy `.env.example` to `.env` for local values. Exported shell values take
precedence. Important settings include:

- `DATABASE_URL` and `GROQ_API_KEY`
- `AUDIENCE_TREND_MINER_EMBEDDING_MODEL` and
  `AUDIENCE_TREND_MINER_EMBEDDING_BATCH_SIZE`
- `AUDIENCE_TREND_MINER_SIMILARITY_THRESHOLD` (selected production value `0.76`)
- `AUDIENCE_TREND_MINER_MAX_LLM_CLUSTERS` (default `10`, or `all`)
- `AUDIENCE_TREND_MINER_CLUSTER_MODEL` and
  `AUDIENCE_TREND_MINER_NARRATIVE_MODEL`

The global command also accepts fixture files at each external boundary for a
deterministic integration run. Run `audience-trend-miner --help` for all options.

## Test

```bash
python3 -m unittest discover -v
```

Browser workflow tests additionally require Playwright and Chromium:

```bash
python3 -m pip install -e '.[browser-test]'
python3 -m playwright install chromium
python3 -m unittest tests.v2.ui.test_browser
```

Tests mirror the five module boundaries under `tests/v2/` and cover schema
contracts, installed-package assets, global orchestration and resume, structured
publication, and the interactive UI workflow.

## Architecture history

[ADR-0002](docs/adr/0002-audience-intelligence-v2-cluster-first-country-trends.md)
defines the active product architecture. ADR-0001 and the V1 research records
are retained only as historical evidence; they do not describe supported code or
CLI behavior.
