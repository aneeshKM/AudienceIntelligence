from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import jsonschema

from audience_trend_miner.publication import PublicationInput, publish_run
from audience_trend_miner.classification import (
    ArticleClassificationResult,
    classify_articles,
)
from audience_trend_miner.clustering import (
    CandidateClusteringResult,
    FrozenEmbeddingAdapter,
    form_candidate_clusters,
)
from audience_trend_miner.refinement import ClusterRefinementResult, refine_candidate_clusters
from tests.test_article_classification import ScriptedGenerator, judgment
from audience_trend_miner.trends import TrendQualificationResult, qualify_trends
from audience_trend_miner.wikimedia import (
    AcquisitionFailure,
    AliasTraffic,
    AnalysisWindows,
    CanonicalArticle,
    DailyView,
    RawArtifact,
    WikimediaAttentionResult,
)


def publication_input(
    output_root: Path,
    *,
    attention: WikimediaAttentionResult = WikimediaAttentionResult((), (), ()),
    qualification: TrendQualificationResult = TrendQualificationResult((), (), ()),
    classification: ArticleClassificationResult = ArticleClassificationResult((), (), ()),
    clustering: CandidateClusteringResult = CandidateClusteringResult(
        "sentence-transformers/all-mpnet-base-v2", 0.62, (), (), (), ()
    ),
    refinement: ClusterRefinementResult = ClusterRefinementResult((), (), ()),
) -> PublicationInput:
    return PublicationInput(
        output_root=output_root,
        started_at=datetime(2026, 7, 16, 17, 30, 45, 123456, tzinfo=timezone.utc),
        as_of_argument=date(2026, 7, 16),
        as_of=date(2026, 7, 16),
        windows=AnalysisWindows(
            previous_start=date(2026, 7, 1),
            previous_end=date(2026, 7, 7),
            current_start=date(2026, 7, 8),
            current_end=date(2026, 7, 14),
        ),
        attention=attention,
        qualification=qualification,
        classification=classification,
        clustering=clustering,
        refinement=refinement,
        configuration={
            "model": "fixture/model",
            "classification_mode": "fixture",
            "wikimedia_mode": "fixture",
            "database_host": "localhost",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "similarity_threshold": "0.62",
            "embedding_mode": "fixture",
        },
        run_id=None,
    )


def refinement_audience(name: str, *page_ids: int) -> dict[str, object]:
    return {
        "name": name,
        "page_ids": list(page_ids),
        "rationale": "Distinct articles describe one targetable audience.",
    }


def refinement_decision(
    action: str,
    audiences: list[dict[str, object]],
    rejected_page_ids: list[int],
    *,
    alternative_matches: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "action": action,
        "audiences": audiences,
        "rejected_page_ids": rejected_page_ids,
        "alternative_matches": alternative_matches,
        "rationale": "Fixture refinement decision.",
    }


class RunPublicationTest(unittest.TestCase):
    def test_publishes_complete_empty_run_from_finished_domain_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "runs"
            published = publish_run(publication_input(output_root))

            self.assertEqual(published.name, "20260716T173045123456Z")
            self.assertEqual(
                {path.name for path in published.iterdir()},
                {
                    "manifest.json",
                    "portfolio.json",
                    "audit.json",
                    "report.html",
                    "wikimedia",
                    "clustering",
                    ".complete",
                },
            )
            manifest = json.loads((published / "manifest.json").read_text())
            portfolio = json.loads((published / "portfolio.json").read_text())
            audit = json.loads((published / "audit.json").read_text())
            schema_root = Path(__file__).parents[1] / "audience_trend_miner" / "schemas"
            jsonschema.validate(
                portfolio,
                json.loads((schema_root / "portfolio.schema.json").read_text()),
            )
            jsonschema.validate(
                audit,
                json.loads((schema_root / "audit.schema.json").read_text()),
            )

        self.assertEqual(manifest["current_window"]["start"], "2026-07-08")
        self.assertEqual(portfolio["audiences"], [])
        self.assertEqual(audit["qualified_signals"], [])

    def test_stable_run_id_publishes_at_most_one_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_data = publication_input(Path(temporary_directory) / "runs")
            input_data = replace(input_data, run_id="stable-run")

            first = publish_run(input_data)
            second = publish_run(input_data)

            self.assertEqual(first, second)
            self.assertEqual(first.name, "stable-run")
            self.assertEqual(len(list(first.parent.iterdir())), 1)

            with self.assertRaisesRegex(ValueError, "different run facts"):
                publish_run(replace(input_data, as_of=date(2026, 7, 17)))

            (first / ".complete").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                publish_run(input_data)

    def test_publishes_degraded_qualified_signals_with_alias_lineage_and_evidence(self) -> None:
        daily_views = tuple(
            DailyView(
                date=date(2026, 7, 1 + offset),
                views=10_000 if offset < 7 else 20_000,
            )
            for offset in range(14)
        )
        alias = AliasTraffic("Signal_Alias", 70_000, 140_000, daily_views)
        article = CanonicalArticle(
            page_id=42,
            canonical_title="Commercial Signal",
            extract="A useful lead.",
            categories=("Examples",),
            previous_window_views=70_000,
            current_window_views=140_000,
            aliases=(alias,),
        )
        noise_alias = AliasTraffic("Main_Page", 70_000, 140_000, daily_views)
        noise = CanonicalArticle(
            page_id=1,
            canonical_title="Main Page",
            extract="Navigation.",
            categories=(),
            previous_window_views=70_000,
            current_window_views=140_000,
            aliases=(noise_alias,),
        )
        attention = WikimediaAttentionResult(
            raw_candidate_titles=("Broken_Alias", "Main_Page", "Signal_Alias"),
            canonical_articles=(noise, article),
            raw_artifacts=(
                RawArtifact("metadata", "Signal_Alias", {"page_id": 42}),
            ),
            failures=(
                AcquisitionFailure("metadata", "Broken_Alias", 3, "unavailable"),
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            published = publish_run(
                publication_input(
                    Path(temporary_directory) / "runs",
                    attention=attention,
                    qualification=qualify_trends((noise, article)),
                    classification=classify_articles(
                        (article,),
                        ScriptedGenerator(judgment()),
                        sleep=lambda _: None,
                    ),
                )
            )
            audit = json.loads((published / "audit.json").read_text())
            report = (published / "report.html").read_text()

            self.assertTrue(
                (published / "wikimedia" / "metadata" / "Signal_Alias.json").is_file()
            )

        self.assertTrue(audit["degraded"])
        self.assertEqual(audit["qualified_signals"][0]["alias_titles"], ["Signal_Alias"])
        self.assertEqual(
            next(
                decision["reasons"]
                for decision in audit["decisions"]
                if decision["canonical_title"] == "Commercial Signal"
            ),
            ["classification_accepted"],
        )
        self.assertIn("Qualified attention signals", report)
        self.assertIn("Rejected deterministic noise", report)
        self.assertIn("Main Page", report)
        self.assertIn("not yet accepted audiences", report)

    def test_duplicate_evidence_identity_leaves_no_run_output(self) -> None:
        attention = WikimediaAttentionResult(
            raw_candidate_titles=(),
            canonical_articles=(),
            raw_artifacts=(
                RawArtifact("metadata", "Signal", {}),
                RawArtifact("metadata", "Signal", {}),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "runs"

            with self.assertRaises(ValueError):
                publish_run(publication_input(output_root, attention=attention))

            self.assertFalse(output_root.exists())

    def test_retains_only_accepted_articles_and_publishes_complete_classification_evidence(self) -> None:
        accepted_article = _qualified_article(42, "Home espresso")
        rejected_article = _qualified_article(43, "Election result")
        qualification = qualify_trends((accepted_article, rejected_article))
        classification = classify_articles(
            tuple(item.article for item in qualification.qualified),
            ScriptedGenerator(judgment(), judgment("routine_politics")),
            sleep=lambda _: None,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            published = publish_run(
                publication_input(
                    Path(temporary_directory) / "runs",
                    attention=WikimediaAttentionResult(
                        (), (accepted_article, rejected_article), ()
                    ),
                    qualification=qualification,
                    classification=classification,
                )
            )
            audit = json.loads((published / "audit.json").read_text())
            evidence = json.loads(
                (published / "classification" / "article_judgments.json").read_text()
            )
            report = (published / "report.html").read_text()

        self.assertEqual(
            [signal["canonical_title"] for signal in audit["qualified_signals"]],
            ["Home espresso"],
        )
        self.assertEqual(audit["article_classifications"], evidence)
        rejected = next(item for item in evidence if not item["accepted"])
        self.assertEqual(rejected["decision_reason"], "routine_politics")
        self.assertIn("Election result", rejected["prompt"])
        self.assertTrue(rejected["attempts"][0]["validation_valid"])
        self.assertEqual(
            rejected["attempts"][0]["raw_output"]["rejection_class"],
            "routine_politics",
        )
        decisions = {item["canonical_title"]: item for item in audit["decisions"]}
        self.assertEqual(decisions["Home espresso"]["outcome"], "classified_signal")
        self.assertEqual(
            decisions["Election result"]["outcome"], "classification_rejected"
        )
        qualified_section = report.split("<h2>Rejected classifications</h2>", 1)[0]
        self.assertIn("Home espresso", qualified_section)
        self.assertNotIn("Election result", qualified_section)

    def test_publishes_candidate_components_as_auditable_non_audience_results(self) -> None:
        first = _qualified_article(42, "Running shoes")
        second = _qualified_article(43, "Marathon training")
        singleton = _qualified_article(44, "Home espresso")
        qualification = qualify_trends((first, second, singleton))
        classification = classify_articles(
            (first, second, singleton),
            ScriptedGenerator(judgment(), judgment(), judgment()),
            sleep=lambda _: None,
        )
        clustering = form_candidate_clusters(
            (first, second, singleton),
            FrozenEmbeddingAdapter(
                ((1.0, 0.0), (0.8, 0.6), (-1.0, 0.0))
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            published = publish_run(
                publication_input(
                    Path(temporary_directory) / "runs",
                    attention=WikimediaAttentionResult((), (first, second, singleton), ()),
                    qualification=qualification,
                    classification=classification,
                    clustering=clustering,
                )
            )
            audit = json.loads((published / "audit.json").read_text())
            evidence = json.loads(
                (published / "clustering" / "candidate_clusters.json").read_text()
            )
            portfolio = json.loads((published / "portfolio.json").read_text())
            report = (published / "report.html").read_text()

        self.assertEqual(audit["candidate_clustering"], evidence)
        self.assertEqual(evidence["model"], "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(evidence["threshold"], 0.62)
        self.assertEqual(evidence["components"][0]["page_ids"], [42, 43])
        self.assertTrue(evidence["components"][0]["is_candidate_cluster"])
        self.assertFalse(evidence["components"][1]["is_candidate_cluster"])
        self.assertEqual(portfolio["audiences"], [])
        self.assertIn("Candidate clusters", report)
        self.assertIn("not accepted audiences", report)

    def test_publishes_refined_membership_alternatives_singletons_and_safety_evidence(self) -> None:
        first = _qualified_article(42, "Running shoes")
        second = _qualified_article(43, "Marathon training")
        alternative = _qualified_article(44, "Sports event")
        singleton = _qualified_article(45, "Home espresso")
        articles = (first, second, alternative, singleton)
        qualification = qualify_trends(articles)
        classification = classify_articles(
            articles,
            ScriptedGenerator(*(judgment() for _ in articles)),
            sleep=lambda _: None,
        )
        clustering = form_candidate_clusters(
            articles,
            FrozenEmbeddingAdapter(
                ((1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (-1.0, 0.0))
            ),
        )
        refined = refine_candidate_clusters(
            clustering.components,
            articles,
            ScriptedGenerator(
                refinement_decision(
                    "split",
                    [refinement_audience("Endurance Runners", 42, 43)],
                    [44],
                    alternative_matches=[{
                        "page_id": 44,
                        "audience_name": "Endurance Runners",
                        "rationale": "Adjacent signal retained for review only.",
                    }],
                ),
                {
                    "materially_centered_on_tragedy": False,
                    "materially_centered_on_violent_crime": False,
                    "rationale": "Fixture cluster-level safety assessment.",
                },
            ),
            sleep=lambda _: None,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            published = publish_run(
                publication_input(
                    Path(temporary_directory) / "runs",
                    attention=WikimediaAttentionResult((), articles, ()),
                    qualification=qualification,
                    classification=classification,
                    clustering=clustering,
                    refinement=refined,
                )
            )
            audit = json.loads((published / "audit.json").read_text())
            evidence = json.loads(
                (published / "clustering" / "refinement.json").read_text()
            )

        self.assertEqual(audit["cluster_refinement"], evidence)
        self.assertEqual(evidence["accepted"][0]["page_ids"], [42, 43])
        self.assertEqual(evidence["decisions"][0]["rejected_page_ids"], [44])
        self.assertEqual(evidence["decisions"][0]["alternative_matches"][0]["page_id"], 44)
        self.assertEqual(evidence["rejected_standalone_page_ids"], [45])
        self.assertTrue(evidence["accepted"][0]["safety"]["safe"])

    def test_refinement_exhaustion_degrades_run_and_enters_failure_ledger(self) -> None:
        articles = (
            _qualified_article(42, "Running shoes"),
            _qualified_article(43, "Marathon training"),
        )
        qualification = qualify_trends(articles)
        classification = classify_articles(
            articles,
            ScriptedGenerator(judgment(), judgment()),
            sleep=lambda _: None,
        )
        clustering = form_candidate_clusters(
            articles, FrozenEmbeddingAdapter(((1.0, 0.0), (1.0, 0.0)))
        )
        refined = refine_candidate_clusters(
            clustering.components,
            articles,
            ScriptedGenerator({"invalid": True}, {"invalid": True}, {"invalid": True}),
            sleep=lambda _: None,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            published = publish_run(
                publication_input(
                    Path(temporary_directory) / "runs",
                    attention=WikimediaAttentionResult((), articles, ()),
                    qualification=qualification,
                    classification=classification,
                    clustering=clustering,
                    refinement=refined,
                )
            )
            audit = json.loads((published / "audit.json").read_text())

        self.assertTrue(audit["degraded"])
        self.assertEqual(audit["failures"][0]["operation"], "cluster_refinement")
        self.assertEqual(audit["failures"][0]["attempts"], 3)
        self.assertIn("ValidationError", audit["failures"][0]["reason"])

    def test_schema_validation_failure_creates_no_output_root(self) -> None:
        invalid_attention = WikimediaAttentionResult(
            raw_candidate_titles=(),
            canonical_articles=(),
            raw_artifacts=(),
            failures=(
                AcquisitionFailure("metadata", "Broken_Alias", 4, "unavailable"),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "runs"

            with self.assertRaises(jsonschema.ValidationError):
                publish_run(
                    publication_input(output_root, attention=invalid_attention)
                )

            self.assertFalse(output_root.exists())


def _qualified_article(page_id: int, title: str) -> CanonicalArticle:
    daily_views = tuple(
        DailyView(date=date(2026, 7, 1 + offset), views=10_000)
        for offset in range(14)
    )
    return CanonicalArticle(
        page_id=page_id,
        canonical_title=title,
        extract="Fixture lead.",
        categories=("Consumer topics",),
        previous_window_views=70_000,
        current_window_views=140_000,
        aliases=(AliasTraffic(title, 70_000, 140_000, daily_views),),
    )


if __name__ == "__main__":
    unittest.main()
