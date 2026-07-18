from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.v2.trend_portfolio.test_traffic import _artifacts
from audience_trend_miner.v2.trend_portfolio.narratives import (
    narrative_validation_errors,
)


# Publish qualifying upstream.
def _publish_qualifying_upstream(root: Path, run_id: str) -> tuple[Path, Path]:
    evidence_path, adjudication_path = _artifacts(root, run_id=run_id)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    observations = {
        1: {"previous": 10_000, "current": 30_000},
        2: {"previous": 10_000, "current": 30_000},
        3: {"previous": 160_000, "current": 20_000},
        4: {"previous": 160_000, "current": 20_000},
        5: {"previous": 10_000, "current": 10_000},
        6: {"previous": 10_000, "current": 10_000},
    }
    for page in evidence["payload"]["canonical_pages"]:
        values = observations[page["page_id"]]
        page["observations"] = [
            {
                "date": day["date"],
                "views_ceil": values[day["window"]],
            }
            for day in evidence["payload"]["nominal_days"]
            if day["status"] == "successful"
        ]
    for cutoff in evidence["payload"]["daily_cutoffs"]:
        cutoff["views_ceil"] = 5_000
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return evidence_path, adjudication_path


# Publish nonqualifying upstream.
def _publish_nonqualifying_upstream(root: Path, run_id: str) -> tuple[Path, Path]:
    evidence_path, adjudication_path = _artifacts(root, run_id=run_id)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    successful_dates = {
        day["date"]
        for day in evidence["payload"]["nominal_days"]
        if day["status"] == "successful"
    }
    for page in evidence["payload"]["canonical_pages"]:
        page["observations"] = [
            {"date": day, "views_ceil": 100}
            for day in sorted(successful_dates)
        ]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return evidence_path, adjudication_path


# Run the Trend Portfolio stage with fixture-backed narratives.
def _run_stage(
    output_root: Path,
    fixture_path: Path,
    *,
    run_id: str = "narrative-run",
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "audience_trend_miner",
            "v2-trend-portfolio",
            "--run-id",
            run_id,
            "--output-dir",
            str(output_root),
            "--fixture",
            str(fixture_path),
            "--progress-format",
            "json",
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


# Return bounded narrative.
def _bounded_narrative(
    name: str,
    direction: str,
    *brand_categories: str,
) -> dict[str, object]:
    category_text = ", ".join(brand_categories)
    direction_word = "rose" if direction == "robust_growth" else "declined"
    rating = "medium"
    return {
        "name": name,
        "summary": (
            f"Attention to {name} topics {direction_word} in the supplied comparison."
        ),
        "commercial_interpretation": (
            f"{category_text} brands may find the supplied topic group commercially relevant."
        ),
        "brand_categories": list(brand_categories),
        "buying_power_rating": rating,
        "buying_power_rationale": (
            f"The {rating} rating is a qualitative assessment based on the supplied "
            f"topics' relevance to {category_text}."
        ),
    }


# Write valid fixture.
def _write_valid_fixture(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "model": "fixture/narrative-model",
                "clusters": [
                    {
                        "cluster_id": cluster_id,
                        "responses": [
                            _bounded_narrative(name, direction, category)
                        ],
                    }
                    for cluster_id, name, direction, category in (
                        (
                            "final-audience-cluster-0002",
                            "Shrinking",
                            "robust_shrinking",
                            "HVAC",
                        ),
                        (
                            "final-audience-cluster-0001",
                            "Growing",
                            "robust_growth",
                            "Air quality",
                        ),
                    )
                ],
            }
        ),
        encoding="utf-8",
    )


# Group tests for trend portfolio stage behavior.
class TrendPortfolioStageTest(unittest.TestCase):
    # Verify: source name identity words are not treated as invented claims.
    def test_source_name_identity_words_are_not_treated_as_invented_claims(self) -> None:
        evidence = {
            "source_cluster_name": "Godzilla Kaiju Fans",
            "direction": "sudden_growth",
            "suddenly_trending": True,
        }
        narrative = _bounded_narrative(
            "Godzilla Kaiju Fans", "sudden_growth", "Entertainment"
        )
        narrative["summary"] = (
            "Attention to Godzilla Kaiju Fans topics was suddenly trending "
            "in the supplied comparison."
        )

        self.assertEqual(narrative_validation_errors(narrative, evidence), ())

    # Verify: rejects equivalent prohibited claims and invented traffic.
    def test_rejects_equivalent_prohibited_claims_and_invented_traffic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_qualifying_upstream(root, "narrative-run")
            fixture_path = root / "claims.json"
            _write_valid_fixture(fixture_path)
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            valid_first = fixture["clusters"][0]["responses"][0]
            valid_second = fixture["clusters"][1]["responses"][0]
            fixture["clusters"][0]["responses"] = [
                {
                    **valid_first,
                    "summary": (
                        "Prosperous purchasers aim to acquire equipment; demand "
                        "should expand, spurred by prices."
                    ),
                },
                valid_first,
            ]
            fixture["clusters"][1]["responses"] = [
                {
                    **valid_second,
                    "summary": "Pageviews surged to record levels.",
                },
                valid_second,
            ]
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

            completed = _run_stage(root, fixture_path)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(
                (root / "narrative-run" / "trend-portfolio.json").read_text()
            )["payload"]
            first_errors = payload["narrative_evidence"][0]["attempts"][0]["errors"]
            for claim in ("causation", "reader identity", "income", "intent", "prediction"):
                self.assertTrue(any(claim in error for error in first_errors), claim)
            second_errors = payload["narrative_evidence"][1]["attempts"][0]["errors"]
            self.assertTrue(any("traffic" in error for error in second_errors))

    # Verify: resume rejects tampered checkpoint facts and evidence.
    def test_resume_rejects_tampered_checkpoint_facts_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_qualifying_upstream(root, "narrative-run")
            fixture_path = root / "checkpoint.json"
            _write_valid_fixture(fixture_path)
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            valid_first = fixture["clusters"][0]["responses"][0]
            fixture["clusters"][0]["responses"] = [
                {**valid_first, "summary": "Prices caused the change."},
                valid_first,
            ]
            invalid = {
                **fixture["clusters"][1]["responses"][0],
                "summary": "This audience will buy equipment.",
            }
            fixture["clusters"][1]["responses"] = [invalid, invalid, invalid]
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            failed = _run_stage(root, fixture_path)
            self.assertNotEqual(failed.returncode, 0)
            checkpoint_path = (
                root / "narrative-run" / ".trend-portfolio.checkpoint.json"
            )
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["completed"][0]["evidence"]["attempts"][0]["output"] = valid_first
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            _write_valid_fixture(fixture_path)

            resumed = _run_stage(root, fixture_path)

            self.assertNotEqual(resumed.returncode, 0)
            self.assertIn("checkpoint deterministic facts", resumed.stderr)

    # Verify: no qualifying cluster publishes empty portfolio without calls.
    def test_no_qualifying_cluster_publishes_empty_portfolio_without_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_nonqualifying_upstream(root, "empty-run")
            fixture_path = root / "empty.json"
            fixture_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "model": "fixture/narrative-model",
                        "clusters": [],
                    }
                ),
                encoding="utf-8",
            )

            completed = _run_stage(
                root, fixture_path, run_id="empty-run"
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(
                (root / "empty-run" / "trend-portfolio.json").read_text()
            )["payload"]
            self.assertEqual(payload["counts"], {"qualified": 0, "narrated": 0})
            self.assertEqual(payload["audience_portfolio"], [])
            self.assertEqual(payload["narrative_evidence"], [])

    # Verify: exhausted validation is atomic auditable and resumable.
    def test_exhausted_validation_is_atomic_auditable_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_qualifying_upstream(root, "narrative-run")
            fixture_path = root / "exhausted.json"
            invalid = {
                "name": "Cooling systems",
                "summary": "These readers will buy cooling systems.",
                "commercial_interpretation": "Relevant to home systems.",
                "brand_categories": ["HVAC"],
                "buying_power_rating": "medium",
                "buying_power_rationale": "The topics concern durable systems.",
            }
            valid = _bounded_narrative(
                "Growing", "robust_growth", "Air quality"
            )
            base = root / "base.json"
            _write_valid_fixture(base)
            fixture = json.loads(base.read_text(encoding="utf-8"))
            fixture["clusters"][1]["responses"] = [invalid, invalid, invalid]
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

            failed = _run_stage(root, fixture_path)

            run_directory = root / "narrative-run"
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((run_directory / "trend-portfolio.json").exists())
            checkpoint = json.loads(
                (run_directory / ".trend-portfolio.checkpoint.json").read_text()
            )
            self.assertEqual(len(checkpoint["completed"]), 1)
            failure_path = run_directory / ".trend-portfolio.failure.json"
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual(failure["cluster_id"], "final-audience-cluster-0001")
            self.assertEqual(len(failure["attempts"]), 3)
            self.assertTrue(
                all(attempt["validation_status"] == "invalid" for attempt in failure["attempts"])
            )

            fixture["clusters"][1]["responses"] = [valid]
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            resumed = _run_stage(root, fixture_path)

            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertFalse(failure_path.exists())
            operations = [
                json.loads(line)["operation"] for line in resumed.stdout.splitlines()
            ]
            self.assertEqual(
                operations,
                ["attachment", "qualification", "ranking", "narrative", "publish"],
            )

    # Verify: resume rejects schema valid changes to deterministic facts.
    def test_resume_rejects_schema_valid_changes_to_deterministic_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_qualifying_upstream(root, "narrative-run")
            fixture_path = root / "narratives.json"
            _write_valid_fixture(fixture_path)
            first = _run_stage(root, fixture_path)
            self.assertEqual(first.returncode, 0, first.stderr)
            artifact_path = root / "narrative-run" / "trend-portfolio.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["payload"]["audience_portfolio"][0]["direction"] = "robust_growth"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

            resumed = _run_stage(root, fixture_path)

            self.assertNotEqual(resumed.returncode, 0)
            self.assertIn("deterministic facts", resumed.stderr)

    # Verify: retries invalid copy and publishes code owned facts with audit.
    def test_retries_invalid_copy_and_publishes_code_owned_facts_with_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_qualifying_upstream(root, "narrative-run")
            fixture_path = root / "narratives.json"
            fixture_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "model": "fixture/narrative-model",
                        "clusters": [
                            {
                                "cluster_id": "final-audience-cluster-0002",
                                "responses": [
                                    {
                                        **_bounded_narrative(
                                            "Shrinking",
                                            "robust_shrinking",
                                            "HVAC",
                                        ),
                                        "direction": "robust_growth",
                                    },
                                    {
                                        **_bounded_narrative(
                                            "Shrinking",
                                            "robust_shrinking",
                                            "HVAC",
                                        ),
                                        "summary": "Higher prices caused readers to research cooling.",
                                    },
                                    _bounded_narrative(
                                        "Shrinking",
                                        "robust_shrinking",
                                        "HVAC",
                                        "Home improvement",
                                    ),
                                ],
                            },
                            {
                                "cluster_id": "final-audience-cluster-0001",
                                "responses": [
                                    _bounded_narrative(
                                        "Growing",
                                        "robust_growth",
                                        "Air quality",
                                    )
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            completed = _run_stage(root, fixture_path)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            artifact = json.loads(
                (root / "narrative-run" / "trend-portfolio.json").read_text()
            )
            payload = artifact["payload"]
            self.assertEqual(artifact["status"], "complete")
            self.assertEqual(
                payload["run_facts"],
                {
                    "as_of_date": "2026-07-17",
                    "nominal_windows": {
                        "previous": {"start": "2026-07-02", "end": "2026-07-08"},
                        "current": {"start": "2026-07-09", "end": "2026-07-15"},
                    },
                },
            )
            self.assertEqual(payload["counts"], {"qualified": 2, "narrated": 2})
            self.assertEqual(
                [item["cluster_id"] for item in payload["audience_portfolio"]],
                ["final-audience-cluster-0002", "final-audience-cluster-0001"],
            )
            first = payload["audience_portfolio"][0]
            self.assertEqual(first["direction"], "robust_shrinking")
            self.assertGreater(first["impact_score"], 0)
            self.assertEqual(
                set(first["narrative"]),
                {
                    "name",
                    "summary",
                    "commercial_interpretation",
                    "brand_categories",
                    "buying_power_rating",
                    "buying_power_rationale",
                },
            )
            attempts = payload["narrative_evidence"][0]["attempts"]
            model_input = payload["narrative_evidence"][0]["model_input"]
            self.assertNotIn("deterministic_facts", model_input)
            self.assertEqual(
                [member["canonical_title"] for member in model_input["members"]],
                ["Page 3", "Page 4"],
            )
            self.assertEqual(
                [attempt["validation_status"] for attempt in attempts],
                ["invalid", "invalid", "valid"],
            )
            self.assertIn("additional properties", attempts[0]["errors"][0])
            self.assertTrue(
                any("causation" in error for error in attempts[1]["errors"])
            )
            operations = [
                json.loads(line)["operation"] for line in completed.stdout.splitlines()
            ]
            self.assertEqual(operations[:3], ["attachment", "qualification", "ranking"])
            self.assertEqual(operations[-1], "publish")


if __name__ == "__main__":
    unittest.main()
