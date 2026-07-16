from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from html import escape
import json
from pathlib import Path, PurePosixPath
import tempfile

import jsonschema

from audience_trend_miner.classification import ArticleClassificationResult
from audience_trend_miner.trends import TrendDecision, TrendQualificationResult
from audience_trend_miner.wikimedia import AnalysisWindows, WikimediaAttentionResult


SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")


@dataclass(frozen=True)
class PublicationInput:
    output_root: Path
    started_at: datetime
    as_of_argument: date | None
    as_of: date
    windows: AnalysisWindows
    attention: WikimediaAttentionResult
    qualification: TrendQualificationResult
    classification: ArticleClassificationResult


def publish_run(publication: PublicationInput) -> Path:
    """Atomically publish one complete run from finished domain results."""
    artifacts = _assemble_artifacts(publication)
    publication.output_root.mkdir(parents=True, exist_ok=True)
    final_directory = publication.output_root / publication.started_at.strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    with tempfile.TemporaryDirectory(
        dir=publication.output_root,
        prefix=".publication-",
    ) as staging_name:
        staging_directory = Path(staging_name)
        for relative_name, content in artifacts.items():
            artifact_path = staging_directory / relative_name
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(content, encoding="utf-8")
        staging_directory.rename(final_directory)
    return final_directory


def _assemble_artifacts(publication: PublicationInput) -> dict[str, str]:
    manifest = {
        "as_of_argument": (
            publication.as_of_argument.isoformat()
            if publication.as_of_argument is not None
            else None
        ),
        "as_of": publication.as_of.isoformat(),
        "current_window": {
            "start": publication.windows.current_start.isoformat(),
            "end": publication.windows.current_end.isoformat(),
        },
        "previous_window": {
            "start": publication.windows.previous_start.isoformat(),
            "end": publication.windows.previous_end.isoformat(),
        },
    }
    portfolio = {
        "schema_version": "1.0",
        "as_of": publication.as_of.isoformat(),
        "audiences": [],
    }
    audit = {
        "schema_version": "1.0",
        "status": "success",
        "degraded": False,
        "run": manifest,
        "decisions": [
            _decision_audit_record(decision, publication.classification)
            for decision in publication.qualification.decisions
        ],
        "qualified_signals": _accepted_signal_records(publication),
        "article_classifications": [
            decision.audit_data() for decision in publication.classification.decisions
        ],
        "failures": [],
    }
    audit.update(publication.attention.audit_data())
    _validate("portfolio.schema.json", portfolio)
    _validate("audit.schema.json", audit)

    artifacts = {
        "manifest.json": _json_text(manifest),
        "portfolio.json": _json_text(portfolio),
        "audit.json": _json_text(audit),
        "report.html": _report(publication),
        "wikimedia/canonical_articles.json": _json_text(
            audit["canonical_articles"]
        ),
    }
    if publication.classification.decisions:
        artifacts["classification/article_judgments.json"] = _json_text(
            audit["article_classifications"]
        )
    for raw_artifact in publication.attention.raw_artifacts:
        relative_name = _raw_artifact_path(raw_artifact.name)
        if relative_name in artifacts:
            raise ValueError(f"duplicate run artifact path: {relative_name}")
        artifacts[relative_name] = _json_text(raw_artifact.payload)
    return artifacts


def _accepted_signal_records(publication: PublicationInput) -> list[dict[str, object]]:
    accepted_page_ids = {
        decision.page_id for decision in publication.classification.accepted
    }
    return [
        _qualified_signal_record(decision)
        for decision in publication.qualification.qualified
        if decision.article.page_id in accepted_page_ids
    ]


def _raw_artifact_path(name: str) -> str:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ValueError(f"invalid raw artifact path: {name}")
    return str(PurePosixPath("wikimedia") / relative)


def _validate(schema_name: str, artifact: object) -> None:
    schema = json.loads((SCHEMA_DIRECTORY / schema_name).read_text(encoding="utf-8"))
    jsonschema.validate(artifact, schema)


def _json_text(artifact: object) -> str:
    return json.dumps(artifact, indent=2) + "\n"


def _decision_audit_record(
    decision: TrendDecision,
    classification: ArticleClassificationResult,
) -> dict[str, object]:
    classified = next(
        (
            item
            for item in classification.decisions
            if item.page_id == decision.article.page_id
        ),
        None,
    )
    outcome = (
        "rejected_noise"
        if decision.exclusion_reason is not None
        else "classified_signal"
        if classified is not None and classified.accepted
        else "classification_rejected"
        if classified is not None
        else "qualified_signal"
        if decision.included
        else "failed_qualification"
    )
    reasons = []
    if decision.exclusion_reason is not None:
        reasons.append(decision.exclusion_reason)
    if not decision.gates.minimum_traffic:
        reasons.append("minimum_traffic_failed")
    if not decision.gates.growth:
        reasons.append("growth_failed")
    if not decision.gates.positive_score:
        reasons.append("positive_score_failed")
    if decision.included and classified is None:
        reasons.append("all_qualification_gates_passed")
    elif classified is not None:
        reasons.append(
            "classification_accepted"
            if classified.accepted
            else f"classification_rejected:{classified.decision_reason}"
        )
    return {
        "page_id": decision.article.page_id,
        "canonical_title": decision.article.canonical_title,
        "previous_window_views": decision.article.previous_window_views,
        "current_window_views": decision.article.current_window_views,
        "trend_score": decision.score,
        "gates": {
            "minimum_traffic": decision.gates.minimum_traffic,
            "growth": decision.gates.growth,
            "positive_score": decision.gates.positive_score,
        },
        "outcome": outcome,
        "reasons": reasons,
        "exclusion_reason": decision.exclusion_reason,
    }


def _qualified_signal_record(decision: TrendDecision) -> dict[str, object]:
    return {
        "page_id": decision.article.page_id,
        "canonical_title": decision.article.canonical_title,
        "alias_titles": [alias.raw_title for alias in decision.article.aliases],
        "previous_window_views": decision.article.previous_window_views,
        "current_window_views": decision.article.current_window_views,
        "trend_score": decision.score,
    }


def _report(publication: PublicationInput) -> str:
    qualification = publication.qualification
    accepted_page_ids = {
        decision.page_id for decision in publication.classification.accepted
    }
    qualified_items = "".join(
        f"<li><strong>{escape(decision.article.canonical_title)}</strong> — "
        f"{decision.article.current_window_views:,} current views; "
        f"trend score {decision.score:.2f}</li>"
        for decision in qualification.qualified
        if decision.article.page_id in accepted_page_ids
    ) or "<li>No attention signals qualified for this run.</li>"
    classification_rejections = "".join(
        f"<li><strong>{escape(decision.canonical_title)}</strong> — "
        f"{escape(decision.decision_reason)}</li>"
        for decision in publication.classification.rejected
    ) or "<li>No article classifications were rejected.</li>"
    noise_items = "".join(
        f"<li><strong>{escape(decision.article.canonical_title)}</strong> — "
        f"{escape(decision.exclusion_reason or '')}</li>"
        for decision in qualification.rejected_noise
    ) or "<li>No deterministic noise was rejected.</li>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Emerging Audience Portfolio</title>
  <style>
    body {{ background: #f5f1e8; color: #17211c; font: 18px/1.6 Georgia, serif; margin: 0; }}
    main {{ margin: 10vh auto; max-width: 760px; padding: 3rem; background: #fff; border-top: 8px solid #c65d36; }}
    h1 {{ font-size: clamp(2rem, 6vw, 4rem); line-height: 1; margin-top: 0; }}
    .notice {{ border-left: 3px solid #c65d36; padding-left: 1rem; }}
  </style>
</head>
<body><main>
  <p>Audience Trend Miner</p>
  <h1>Emerging Audience Portfolio</h1>
  <p class="notice">These are qualified attention signals, not yet accepted audiences.</p>
  <h2>Qualified attention signals</h2>
  <ul>{qualified_items}</ul>
  <h2>Rejected deterministic noise</h2>
  <ul>{noise_items}</ul>
  <h2>Rejected classifications</h2>
  <ul>{classification_rejections}</ul>
  <p>No emerging audiences qualified for this run.</p>
</main></body>
</html>
"""
