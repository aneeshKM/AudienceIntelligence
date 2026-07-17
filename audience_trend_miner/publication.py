from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from html import escape
import json
import re
from pathlib import Path, PurePosixPath
import tempfile
from urllib.parse import quote

import jsonschema

from audience_trend_miner.classification import ArticleClassification, ArticleClassificationResult
from audience_trend_miner.clustering import CandidateClusteringResult
from audience_trend_miner.refinement import ClusterRefinementResult
from audience_trend_miner.portfolio import PortfolioAudience, PortfolioResult
from audience_trend_miner.quality import verify_publication_quality
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
    clustering: CandidateClusteringResult
    refinement: ClusterRefinementResult
    configuration: dict[str, str]
    run_id: str | None
    portfolio: PortfolioResult = field(default_factory=lambda: PortfolioResult((), ()))


@dataclass(frozen=True)
class DegradationSummary:
    status: str
    degraded: bool
    failure_count: int
    failure_reasons: tuple[str, ...]


def publish_run(publication: PublicationInput) -> Path:
    """Atomically publish one complete run from finished domain results."""
    artifacts = _assemble_artifacts(publication)
    publication.output_root.mkdir(parents=True, exist_ok=True)
    directory_name = publication.run_id or publication.started_at.strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    if not directory_name or directory_name in {".", ".."} or "/" in directory_name:
        raise ValueError("run_id must be one safe path segment")
    final_directory = publication.output_root / directory_name
    if final_directory.is_dir():
        _verify_existing_publication(final_directory, publication)
        return final_directory
    with tempfile.TemporaryDirectory(
        dir=publication.output_root,
        prefix=".publication-",
    ) as staging_name:
        staging_directory = Path(staging_name)
        for relative_name, content in artifacts.items():
            artifact_path = staging_directory / relative_name
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(content, encoding="utf-8")
        (staging_directory / ".complete").write_text("complete\n", encoding="utf-8")
        try:
            staging_directory.rename(final_directory)
        except FileExistsError:
            _verify_existing_publication(final_directory, publication)
    return final_directory


def _verify_existing_publication(
    directory: Path, publication: PublicationInput
) -> None:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("existing run directory is incomplete") from error
    required = {"manifest.json", "audit.json", "portfolio.json", "report.html", ".complete"}
    if not all((directory / name).is_file() for name in required):
        raise ValueError("existing run directory is incomplete")
    if (
        manifest.get("run_id") != publication.run_id
        or manifest.get("configuration") != publication.configuration
        or manifest.get("as_of") != publication.as_of.isoformat()
    ):
        raise ValueError("existing run directory belongs to different run facts")


def _assemble_artifacts(publication: PublicationInput) -> dict[str, str]:
    failures = _run_failure_records(publication)
    degradation = _degradation_summary(failures)
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
        "configuration": publication.configuration,
        "run_id": publication.run_id,
        **_degradation_record(degradation),
    }
    portfolio = {
        "schema_version": "1.0",
        "as_of": publication.as_of.isoformat(),
        **_degradation_record(degradation),
        "audiences": [_portfolio_record(item) for item in publication.portfolio.audiences],
    }
    audit = {
        "schema_version": "1.0",
        "status": degradation.status,
        "degraded": degradation.degraded,
        "failure_count": degradation.failure_count,
        "run": manifest,
        "decisions": [
            _decision_audit_record(decision, publication.classification)
            for decision in publication.qualification.decisions
        ],
        "qualified_signals": _accepted_signal_records(publication),
        "article_classifications": [
            _classification_audit_record(decision)
            for decision in publication.classification.decisions
        ],
        "candidate_clustering": json.loads(
            json.dumps(asdict(publication.clustering))
        ),
        "cluster_refinement": json.loads(json.dumps(asdict(publication.refinement))),
        "portfolio_assessments": json.loads(
            json.dumps([asdict(item) for item in publication.portfolio.assessments])
        ),
        "portfolio_calculations": json.loads(
            json.dumps([asdict(item) for item in publication.portfolio.audiences])
        ),
        "failures": failures,
    }
    audit.update(_attention_audit_data(publication.attention))
    quality = verify_publication_quality(audit, portfolio)
    audit["quality_checks"] = {
        "traced_page_ids": list(quality.traced_page_ids),
        "total_size_basis_points": quality.total_size_basis_points,
        "scored_component_ids": list(quality.scored_component_ids),
        "excluded_component_ids": list(quality.excluded_component_ids),
    }
    _validate("manifest.schema.json", manifest)
    _validate("portfolio.schema.json", portfolio)
    _validate("audit.schema.json", audit)

    artifacts = {
        "manifest.json": _json_text(manifest),
        "portfolio.json": _portfolio_json_text(portfolio),
        "audit.json": _json_text(audit),
        "report.html": _report(publication, failures),
        "wikimedia/canonical_articles.json": _json_text(
            audit["canonical_articles"]
        ),
        "clustering/candidate_clusters.json": _json_text(
            audit["candidate_clustering"]
        ),
        "clustering/refinement.json": _json_text(audit["cluster_refinement"]),
    }
    if publication.classification.decisions:
        artifacts["classification/article_judgments.json"] = _json_text(
            audit["article_classifications"]
        )
    for raw_artifact in publication.attention.raw_artifacts:
        relative_name = _raw_artifact_path(
            raw_artifact.operation, raw_artifact.subject
        )
        if relative_name in artifacts:
            raise ValueError(f"duplicate run artifact path: {relative_name}")
        artifacts[relative_name] = _json_text(raw_artifact.payload)
    return artifacts


def _portfolio_record(audience: PortfolioAudience) -> dict[str, object]:
    return {
        "source_component_id": audience.source_component_id,
        "page_ids": list(audience.page_ids),
        "name": audience.name,
        "description": audience.description,
        "estimated_size_index": audience.estimated_size_index,
        "potential_buying_power": audience.potential_buying_power,
        "brand_categories": list(audience.brand_categories),
        "buying_power_rationale": audience.buying_power_rationale,
        "buying_power_scores": asdict(audience.buying_power_scores),
    }


def _portfolio_json_text(portfolio: dict[str, object]) -> str:
    return re.sub(
        r'("estimated_size_index": )([0-9]+(?:\.[0-9]+)?)',
        lambda match: f"{match.group(1)}{float(match.group(2)):.2f}",
        _json_text(portfolio),
    )


def _classification_audit_record(decision: ArticleClassification) -> dict[str, object]:
    return {
        "page_id": decision.page_id,
        "canonical_title": decision.canonical_title,
        "prompt": decision.prompt,
        "accepted": decision.accepted,
        "decision_reason": decision.decision_reason,
        "judgment": asdict(decision.judgment) if decision.judgment else None,
        "attempts": [asdict(attempt) for attempt in decision.attempts],
    }


def _attention_audit_data(attention: WikimediaAttentionResult) -> dict[str, object]:
    return {
        "raw_candidate_titles": list(attention.raw_candidate_titles),
        "canonical_articles": [
            {
                "page_id": article.page_id,
                "canonical_title": article.canonical_title,
                "extract": article.extract,
                "categories": list(article.categories),
                "previous_window_views": article.previous_window_views,
                "current_window_views": article.current_window_views,
                "aliases": [
                    {
                        "raw_title": alias.raw_title,
                        "previous_window_views": alias.previous_window_views,
                        "current_window_views": alias.current_window_views,
                        "daily_views": [
                            {"date": item.date.isoformat(), "views": item.views}
                            for item in alias.daily_views
                        ],
                    }
                    for alias in article.aliases
                ],
            }
            for article in attention.canonical_articles
        ],
    }


def _accepted_signal_records(publication: PublicationInput) -> list[dict[str, object]]:
    accepted_page_ids = {
        decision.page_id for decision in publication.classification.accepted
    }
    return [
        _qualified_signal_record(decision)
        for decision in publication.qualification.qualified
        if decision.article.page_id in accepted_page_ids
    ]


def _run_failure_records(publication: PublicationInput) -> list[dict[str, object]]:
    return [
        *[asdict(failure) for failure in publication.attention.failures],
        *_classification_failure_records(publication.classification),
        *_refinement_failure_records(publication.refinement),
        *_portfolio_failure_records(publication.portfolio),
    ]


def _classification_failure_records(
    classification: ArticleClassificationResult,
) -> list[dict[str, object]]:
    return [
        {
            "operation": "article_classification",
            "subject": f"page:{decision.page_id}",
            "attempts": len(decision.attempts),
            "reason": decision.attempts[-1].error or "structured generation failed",
        }
        for decision in classification.decisions
        if decision.decision_reason == "exhausted_attempts"
    ]


def _failure_summary(failure: dict[str, object]) -> str:
    return f"{failure['operation']}: {failure['subject']} — {failure['reason']}"


def _degradation_summary(
    failures: list[dict[str, object]],
) -> DegradationSummary:
    degraded = bool(failures)
    return DegradationSummary(
        "degraded" if degraded else "success",
        degraded,
        len(failures),
        tuple(_failure_summary(item) for item in failures),
    )


def _degradation_record(summary: DegradationSummary) -> dict[str, object]:
    return {
        "status": summary.status,
        "degraded": summary.degraded,
        "failure_count": summary.failure_count,
        "failure_reasons": list(summary.failure_reasons),
    }


def _refinement_failure_records(
    refinement: ClusterRefinementResult,
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for decision in refinement.decisions:
        if decision.outcome == "exhausted_attempts":
            failures.append(
                {
                    "operation": "cluster_refinement",
                    "subject": f"component:{decision.component_id}",
                    "attempts": len(decision.attempts),
                    "reason": decision.attempts[-1].error or "structured generation failed",
                }
            )
        for assessment in decision.safety_assessments:
            if assessment.decision_reason == "exhausted_attempts":
                failures.append(
                    {
                        "operation": "cluster_safety",
                        "subject": (
                            f"component:{decision.component_id}:"
                            f"audience:{assessment.audience_name}"
                        ),
                        "attempts": len(assessment.attempts),
                        "reason": (
                            assessment.attempts[-1].error
                            or "structured generation failed"
                        ),
                    }
                )
    return failures


def _portfolio_failure_records(portfolio: PortfolioResult) -> list[dict[str, object]]:
    return [
        {
            "operation": "portfolio_assessment",
            "subject": f"component:{assessment.source_component_id}",
            "attempts": len(assessment.attempts),
            "reason": assessment.attempts[-1].error or "structured generation failed",
        }
        for assessment in portfolio.assessments
        if assessment.attempts and not assessment.attempts[-1].validation_valid
    ]


def _raw_artifact_path(operation: str, subject: str) -> str:
    if operation not in {"discovery", "pageviews", "metadata"} or not subject:
        raise ValueError("invalid raw evidence identity")
    filename = quote(subject, safe="") + ".json"
    return str(PurePosixPath("wikimedia") / operation / filename)


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


def _report(
    publication: PublicationInput, failures: list[dict[str, object]]
) -> str:
    degraded_notice = (
        '<section class="notice"><h2>This run is degraded</h2><p>'
        f"{len(failures)} item-level failure(s) occurred; unaffected items completed.</p><ul>"
        + "".join(f"<li>{escape(_failure_summary(item))}</li>" for item in failures)
        + "</ul></section>"
        if failures else ""
    )
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
    titles_by_page_id = {
        article.page_id: article.canonical_title
        for article in publication.attention.canonical_articles
    }
    candidate_clusters = "".join(
        "<li>" + ", ".join(
            f"<strong>{escape(titles_by_page_id[page_id])}</strong>"
            for page_id in component.page_ids
        ) + "</li>"
        for component in publication.clustering.components
        if component.is_candidate_cluster
    ) or "<li>No multi-signal candidate clusters formed.</li>"
    accepted_audiences = "".join(
        "<article class=\"audience\">"
        f"<h2>{escape(audience.name)}</h2>"
        f"<p>{escape(audience.description)}</p>"
        f"<p><strong>Estimated Size Index:</strong> {audience.estimated_size_index:.2f}</p>"
        f"<p><strong>Potential Buying Power:</strong> {escape(audience.potential_buying_power.title())}</p>"
        f"<p><strong>Component scores:</strong> Purchase intent {audience.buying_power_scores.purchase_intent}/3; "
        f"transaction value {audience.buying_power_scores.transaction_value}/3; category breadth "
        f"{audience.buying_power_scores.category_breadth}/3; brand safety {audience.buying_power_scores.brand_safety}/3.</p>"
        f"<p><strong>Relevant brand categories:</strong> {escape(', '.join(audience.brand_categories))}</p>"
        f"<p>{escape(audience.buying_power_rationale)}</p>"
        "</article>"
        for audience in publication.portfolio.audiences
    ) or "<p>No emerging audiences qualified for this run.</p>"
    singleton_signals = "".join(
        f"<li><strong>{escape(titles_by_page_id[component.page_ids[0]])}</strong></li>"
        for component in publication.clustering.components
        if not component.is_candidate_cluster
    ) or "<li>No singleton signals remained.</li>"
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
  {degraded_notice}
  <p class="notice">Qualified signals are not yet accepted audiences; candidate clusters are not accepted audiences until they pass semantic refinement and a separate cluster-level safety veto.</p>
  <h2>Candidate clusters</h2>
  <ul>{candidate_clusters}</ul>
  <h2>Accepted refined audiences</h2>
  {accepted_audiences}
  <h2>Standalone singleton signals</h2>
  <ul>{singleton_signals}</ul>
  <h2>Qualified attention signals</h2>
  <ul>{qualified_items}</ul>
  <h2>Rejected deterministic noise</h2>
  <ul>{noise_items}</ul>
  <h2>Rejected classifications</h2>
  <ul>{classification_rejections}</ul>
</main></body>
</html>
"""
