# AudienceIntelligence

Audience Trend Miner turns public attention signals into an Emerging Audience
Portfolio. The first executable slice establishes a reproducible CLI run and
produces a valid empty portfolio without fabricating audiences.

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

`--as-of` defaults to the current UTC date. Each invocation creates a
timestamped directory containing:

- `manifest.json` — supplied and effective dates plus resolved windows
- `portfolio.json` — schema-valid Emerging Audience Portfolio
- `report.html` — static report, including a valid empty state
- `audit.json` — run status, decisions, and failures

The current and previous analysis windows are complete seven-day periods. The
current window ends two days before the effective as-of date.

## Test

```bash
python -m unittest discover -v
```

The end-to-end test invokes the public CLI in a subprocess and checks its exit
status, observable files, schemas, and resolved dates.
