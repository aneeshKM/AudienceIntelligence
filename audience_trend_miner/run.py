from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import jsonschema

from audience_trend_miner.wikimedia import (
    AnalysisWindows,
    FixtureWikimediaAdapter,
    HttpWikimediaAdapter,
    acquire_wikimedia_attention,
)


SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")


def execute_run(as_of_argument: date | None, output_directory: Path) -> Path:
    """Acquire attention signals and publish an Audience Trend Miner run."""
    started_at = datetime.now(timezone.utc)
    as_of = as_of_argument or started_at.date()
    current_end = as_of - timedelta(days=2)
    current_start = current_end - timedelta(days=6)
    previous_end = current_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=6)

    manifest = {
        "as_of_argument": as_of_argument.isoformat() if as_of_argument else None,
        "as_of": as_of.isoformat(),
        "current_window": {
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
        },
        "previous_window": {
            "start": previous_start.isoformat(),
            "end": previous_end.isoformat(),
        },
    }
    portfolio = {
        "schema_version": "1.0",
        "as_of": as_of.isoformat(),
        "audiences": [],
    }
    audit = {
        "schema_version": "1.0",
        "status": "success",
        "degraded": False,
        "run": manifest,
        "decisions": [],
        "failures": [],
    }
    wikimedia_artifacts: dict[str, object] = {}
    fixture_path = os.environ.get("AUDIENCE_TREND_MINER_WIKIMEDIA_FIXTURE")
    rest_base_url = os.environ.get("AUDIENCE_TREND_MINER_WIKIMEDIA_BASE_URL")
    if fixture_path:
        adapter = FixtureWikimediaAdapter.from_file(Path(fixture_path))
    elif rest_base_url != "":
        adapter = (
            HttpWikimediaAdapter(rest_base_url=rest_base_url)
            if rest_base_url
            else HttpWikimediaAdapter()
        )
    else:
        adapter = None

    if adapter is not None:
        attention = acquire_wikimedia_attention(
            AnalysisWindows(
                previous_start=previous_start,
                previous_end=previous_end,
                current_start=current_start,
                current_end=current_end,
            ),
            adapter,
        )
        audit.update(attention.audit_data())
        wikimedia_artifacts = {
            artifact.name: artifact.payload for artifact in attention.raw_artifacts
        }

    _validate("portfolio.schema.json", portfolio)
    _validate("audit.schema.json", audit)

    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    run_directory = output_directory / timestamp
    run_directory.mkdir(parents=True)
    _write_json(run_directory / "manifest.json", manifest)
    _write_json(run_directory / "portfolio.json", portfolio)
    _write_json(run_directory / "audit.json", audit)
    if "canonical_articles" in audit:
        wikimedia_directory = run_directory / "wikimedia"
        wikimedia_directory.mkdir()
        _write_json(
            wikimedia_directory / "canonical_articles.json",
            audit["canonical_articles"],
        )
        for relative_path, artifact in wikimedia_artifacts.items():
            artifact_path = wikimedia_directory / relative_path
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(artifact_path, artifact)
    (run_directory / "report.html").write_text(_empty_report(), encoding="utf-8")
    return run_directory


def _validate(schema_name: str, artifact: object) -> None:
    schema = json.loads((SCHEMA_DIRECTORY / schema_name).read_text(encoding="utf-8"))
    jsonschema.validate(artifact, schema)


def _write_json(path: Path, artifact: object) -> None:
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def _empty_report() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Emerging Audience Portfolio</title>
  <style>
    body { background: #f5f1e8; color: #17211c; font: 18px/1.6 Georgia, serif; margin: 0; }
    main { margin: 10vh auto; max-width: 760px; padding: 3rem; background: #fff; border-top: 8px solid #c65d36; }
    h1 { font-size: clamp(2rem, 6vw, 4rem); line-height: 1; margin-top: 0; }
    .empty { border-left: 3px solid #c65d36; padding-left: 1rem; }
  </style>
</head>
<body><main>
  <p>Audience Trend Miner</p>
  <h1>Emerging Audience Portfolio</h1>
  <p class="empty">No emerging audiences qualified for this run.</p>
</main></body>
</html>
"""
