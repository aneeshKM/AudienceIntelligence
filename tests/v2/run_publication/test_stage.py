from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema

from audience_trend_miner.v2.shared import canonical_json_fingerprint
from tests.v2.trend_portfolio.test_stage import (
    _publish_qualifying_upstream,
    _run_stage as run_trend_stage,
    _write_valid_fixture,
)
from tests.v2.trend_portfolio.test_traffic import _artifacts


def _semantic_evidence_fingerprint(evidence: dict[str, object]) -> str:
    payload = evidence["payload"]
    assert isinstance(payload, dict)
    pages = payload["canonical_pages"]
    assert isinstance(pages, list)
    records = [
        {
            "page_id": page["page_id"],
            "canonical_title": page["canonical_title"],
            "lead": page["lead"],
            "categories": sorted(set(page["categories"])),
        }
        for page in pages
    ]
    records.sort(key=lambda page: (page["page_id"], page["canonical_title"]))
    return canonical_json_fingerprint(records)


def _publish_completed_upstream(
    root: Path,
    run_id: str = "publication-run",
    *,
    non_empty: bool = True,
) -> None:
    evidence_path, adjudication_path = (
        _publish_qualifying_upstream(root, run_id)
        if non_empty
        else _artifacts(root, run_id=run_id)
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    clusters = adjudication["payload"]["final_audience_clusters"]
    formation = {
        "schema_version": "2.0",
        "run_id": run_id,
        "stage": "semantic-audience-formation",
        "status": "complete",
        "payload": {
            "configuration": {
                "category_rule_set_version": "1.0",
                "embedding_model": "fixture/embedding-model",
                "content_weight": 0.7,
                "category_weight": 0.3,
                "similarity_threshold": 0.76,
                "subdivision_policy": {
                    "max_input_tokens": 16384,
                    "fixed_prompt_tokens": 2048,
                    "stricter_threshold_step": 0.02,
                    "method": "stricter-boundary",
                    "token_estimation": "utf8-bytes-upper-bound",
                },
                "review_cap": 10,
                "wikimedia_evidence_fingerprint": _semantic_evidence_fingerprint(
                    evidence
                ),
            },
            "counts": {
                "eligible_clusters": len(clusters),
                "selected_clusters": len(clusters),
                "omitted_clusters": 0,
                "discarded_singleton_components": 0,
                "subdivided_components": 0,
                "subdivisions_created": 0,
                "singleton_subdivisions": 0,
            },
            "preliminary_clusters": [
                {
                    "cohesion": 0.8,
                    "subdivision": None,
                    "members": cluster["members"],
                }
                for cluster in clusters
            ],
            "completion": {"status": "complete"},
        },
    }
    formation_path = root / run_id / "semantic-audience-formation.json"
    formation_path.write_text(json.dumps(formation), encoding="utf-8")
    adjudication["payload"]["configuration"][
        "semantic_audience_formation_fingerprint"
    ] = canonical_json_fingerprint(formation)
    adjudication_path.write_text(json.dumps(adjudication), encoding="utf-8")

    fixture_path = root / "narratives.json"
    if non_empty:
        _write_valid_fixture(fixture_path)
    else:
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
    completed = run_trend_stage(root, fixture_path, run_id=run_id)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)


def _run_publication(
    root: Path,
    *,
    run_id: str = "publication-run",
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "audience_trend_miner",
            "v2-run-publication",
            "--run-id",
            run_id,
            "--output-dir",
            str(root),
            "--progress-format",
            "json",
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


class RunPublicationStageTest(unittest.TestCase):
    def test_publishes_exact_schema_valid_artifact_set_with_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_completed_upstream(root)

            completed = _run_publication(root)

            publication = root / "publication-run" / "publication"
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                {path.name for path in publication.iterdir()},
                {"portfolio.json", "audit.json", "manifest.json"},
            )
            portfolio = json.loads(
                (publication / "portfolio.json").read_text(encoding="utf-8")
            )
            audit = json.loads(
                (publication / "audit.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (publication / "manifest.json").read_text(encoding="utf-8")
            )
            schema_directory = (
                Path(__file__).parents[3]
                / "audience_trend_miner"
                / "v2"
                / "run_publication"
                / "schemas"
            )
            for name, product in {
                "portfolio": portfolio,
                "audit": audit,
                "manifest": manifest,
            }.items():
                schema = json.loads(
                    (schema_directory / f"{name}.schema.json").read_text(
                        encoding="utf-8"
                    )
                )
                jsonschema.validate(product, schema)
            self.assertEqual(portfolio["run_id"], "publication-run")
            self.assertEqual(len(portfolio["audience_portfolio"]), 2)
            self.assertEqual(audit["run_id"], "publication-run")
            self.assertEqual(
                set(manifest["published_artifacts"]),
                {"portfolio.json", "audit.json"},
            )
            self.assertTrue(
                all(
                    record["sha256"].startswith("sha256:")
                    for record in manifest["published_artifacts"].values()
                )
            )
            for artifact_name, record in manifest["published_artifacts"].items():
                content = (publication / artifact_name).read_bytes()
                self.assertEqual(
                    record,
                    {
                        "schema_version": "1.0",
                        "sha256": f"sha256:{hashlib.sha256(content).hexdigest()}",
                        "bytes": len(content),
                    },
                )

    def test_publishes_a_valid_empty_audience_portfolio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_completed_upstream(root, non_empty=False)

            completed = _run_publication(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            portfolio = json.loads(
                (
                    root
                    / "publication-run"
                    / "publication"
                    / "portfolio.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(portfolio["audience_portfolio"], [])
            self.assertEqual(
                portfolio["completion"], {"status": "complete", "empty": True}
            )

    def test_rejects_absent_incomplete_incompatible_and_mismatched_inputs(self) -> None:
        scenarios = (
            ("absent", None, "artifact is absent"),
            ("incomplete", {"status": "writing"}, "artifact is incomplete"),
            ("incompatible", {"schema_version": "1.0"}, "schema-invalid"),
            ("mismatched", {"run_id": "another-run"}, "different run facts"),
        )
        for name, changes, expected_error in scenarios:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _publish_completed_upstream(root)
                run_directory = root / "publication-run"
                artifact_path = run_directory / "semantic-audience-formation.json"
                if changes is None:
                    artifact_path.unlink()
                    before = None
                else:
                    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                    artifact.update(changes)
                    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
                    before = artifact_path.read_bytes()

                completed = _run_publication(root)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_error, completed.stderr)
                self.assertFalse((run_directory / "publication").exists())
                if before is not None:
                    self.assertEqual(artifact_path.read_bytes(), before)

    def test_rejects_schema_valid_but_incompatible_evidence_without_recalculation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_completed_upstream(root)
            run_directory = root / "publication-run"
            evidence_path = run_directory / "wikimedia-evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["payload"]["canonical_pages"][0]["lead"] = "Changed evidence."
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            upstream_before = {
                path.name: path.read_bytes()
                for path in run_directory.glob("*.json")
            }

            completed = _run_publication(root)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("upstream artifacts are incompatible", completed.stderr)
            self.assertFalse((run_directory / "publication").exists())
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in run_directory.glob("*.json")
                },
                upstream_before,
            )

    def test_rejects_prohibited_secret_and_hidden_reasoning_fields(self) -> None:
        for field in ("api_key", "chain_of_thought"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _publish_completed_upstream(root)
                run_directory = root / "publication-run"
                trend_path = run_directory / "trend-portfolio.json"
                trend = json.loads(trend_path.read_text(encoding="utf-8"))
                attempts = trend["payload"]["narrative_evidence"][0]["attempts"]
                valid_attempt = {**attempts[-1], "attempt": 2}
                attempts[:] = [
                    {
                        **attempts[-1],
                        "validation_status": "invalid",
                        "output": {field: "must-not-publish"},
                        "errors": ["rejected unsafe output"],
                    },
                    valid_attempt,
                ]
                trend_path.write_text(json.dumps(trend), encoding="utf-8")

                completed = _run_publication(root)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("prohibited field", completed.stderr)
                self.assertFalse((run_directory / "publication").exists())

    def test_does_not_publish_unstructured_model_attempt_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_completed_upstream(root)
            run_directory = root / "publication-run"
            trend_path = run_directory / "trend-portfolio.json"
            trend = json.loads(trend_path.read_text(encoding="utf-8"))
            marker = "private hidden reasoning marker"
            attempts = trend["payload"]["narrative_evidence"][0]["attempts"]
            valid_attempt = {**attempts[-1], "attempt": 2}
            attempts[:] = [
                {
                    **attempts[-1],
                    "validation_status": "invalid",
                    "output": marker,
                    "errors": [f"'{marker}' is not of type 'object'"],
                },
                valid_attempt,
            ]
            trend_path.write_text(json.dumps(trend), encoding="utf-8")

            completed = _run_publication(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            audit_text = (
                run_directory / "publication" / "audit.json"
            ).read_text(encoding="utf-8")
            self.assertNotIn(marker, audit_text)

    def test_write_and_interruption_failures_never_expose_partial_publication(self) -> None:
        failures = (
            ("write-1", ("--fail-after-artifact", "1")),
            ("write-2", ("--fail-after-artifact", "2")),
            ("write-3", ("--fail-after-artifact", "3")),
            ("interruption", ("--interrupt-before-completion",)),
        )
        for name, extra in failures:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _publish_completed_upstream(root)
                run_directory = root / "publication-run"

                completed = _run_publication(root, extra=extra)

                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse((run_directory / "publication").exists())
                self.assertEqual(list(run_directory.glob(".publication.*")), [])

    def test_collision_is_preserved_and_a_valid_publication_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_completed_upstream(root)
            publication = root / "publication-run" / "publication"
            publication.mkdir()
            sentinel = publication / "owned.txt"
            sentinel.write_text("do not replace", encoding="utf-8")

            collided = _run_publication(root)

            self.assertNotEqual(collided.returncode, 0)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not replace")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_completed_upstream(root)
            first = _run_publication(root)
            publication = root / "publication-run" / "publication"
            before = {path.name: path.read_bytes() for path in publication.iterdir()}

            resumed = _run_publication(root)

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(
                {path.name: path.read_bytes() for path in publication.iterdir()}, before
            )
            events = [json.loads(line) for line in resumed.stdout.splitlines()]
            self.assertEqual([event["operation"] for event in events], ["resume"])

    def test_resume_rejects_stale_publication_for_changed_compatible_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_completed_upstream(root)
            first = _run_publication(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            run_directory = root / "publication-run"
            publication = run_directory / "publication"
            before = {path.name: path.read_bytes() for path in publication.iterdir()}
            trend_path = run_directory / "trend-portfolio.json"
            trend = json.loads(trend_path.read_text(encoding="utf-8"))
            trend["payload"]["configuration"]["model"] = "changed/model"
            for narrative in trend["payload"]["narrative_evidence"]:
                narrative["model"] = "changed/model"
            trend_path.write_text(json.dumps(trend), encoding="utf-8")

            resumed = _run_publication(root)

            self.assertNotEqual(resumed.returncode, 0)
            self.assertIn("collides with requested run", resumed.stderr)
            self.assertEqual(
                {path.name: path.read_bytes() for path in publication.iterdir()}, before
            )

    def test_resume_rejects_hash_valid_but_internally_changed_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_completed_upstream(root)
            first = _run_publication(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            publication = root / "publication-run" / "publication"
            portfolio_path = publication / "portfolio.json"
            portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
            portfolio["audience_portfolio"][0]["narrative"]["name"] = "Tampered"
            portfolio_path.write_text(
                json.dumps(portfolio, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            content = portfolio_path.read_bytes()
            manifest_path = publication / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["published_artifacts"]["portfolio.json"].update(
                {
                    "sha256": f"sha256:{hashlib.sha256(content).hexdigest()}",
                    "bytes": len(content),
                }
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            resumed = _run_publication(root)

            self.assertNotEqual(resumed.returncode, 0)
            self.assertIn("collides with requested run", resumed.stderr)

    def test_resume_rejects_schema_valid_but_false_manifest_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _publish_completed_upstream(root)
            first = _run_publication(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            manifest_path = (
                root / "publication-run" / "publication" / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["configuration_provenance"]["trend-portfolio"][
                "model"
            ] = "false/provenance"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            resumed = _run_publication(root)

            self.assertNotEqual(resumed.returncode, 0)
            self.assertIn("collides with requested run", resumed.stderr)

    def test_rejects_duplicate_or_mismatched_internal_provenance(self) -> None:
        scenarios = (
            "adjudication-ids",
            "duplicate-traffic",
            "bogus-final-source",
            "duplicate-final-id",
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                _publish_completed_upstream(root)
                run_directory = root / "publication-run"
                trend_path = run_directory / "trend-portfolio.json"
                trend = json.loads(trend_path.read_text(encoding="utf-8"))
                if scenario in {
                    "adjudication-ids",
                    "bogus-final-source",
                    "duplicate-final-id",
                }:
                    adjudication_path = run_directory / "cluster-adjudication.json"
                    adjudication = json.loads(
                        adjudication_path.read_text(encoding="utf-8")
                    )
                    if scenario == "adjudication-ids":
                        adjudication["payload"]["adjudications"][1][
                            "preliminary_cluster_id"
                        ] = "preliminary-cluster-0001"
                    elif scenario == "bogus-final-source":
                        bogus = "preliminary-cluster-9999"
                        adjudication["payload"]["final_audience_clusters"][0][
                            "source_preliminary_cluster_id"
                        ] = bogus
                        for collection in (
                            trend["payload"]["audience_portfolio"],
                            trend["payload"]["audit_cluster_traffic"],
                        ):
                            collection[0]["source_preliminary_cluster_id"] = bogus
                    else:
                        adjudication["payload"]["final_audience_clusters"][1][
                            "cluster_id"
                        ] = adjudication["payload"]["final_audience_clusters"][0][
                            "cluster_id"
                        ]
                        trend["payload"]["counts"] = {"qualified": 0, "narrated": 0}
                        trend["payload"]["audience_portfolio"] = []
                        trend["payload"]["narrative_evidence"] = []
                        trend["payload"]["audit_cluster_traffic"].pop(1)
                    adjudication_path.write_text(
                        json.dumps(adjudication), encoding="utf-8"
                    )
                    trend["payload"]["configuration"][
                        "cluster_adjudication_fingerprint"
                    ] = canonical_json_fingerprint(adjudication)
                else:
                    traffic = trend["payload"]["audit_cluster_traffic"]
                    traffic.append(traffic[0])
                trend_path.write_text(json.dumps(trend), encoding="utf-8")

                completed = _run_publication(root)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("inconsistent", completed.stderr)
                self.assertFalse((run_directory / "publication").exists())

    def test_rejects_cross_stage_value_contradictions(self) -> None:
        scenarios = (
            "formation-member",
            "final-member",
            "traffic-members",
            "traffic-values",
            "percentage-change",
            "coverage",
            "impact-score",
            "ranking",
            "final-narrative",
            "unsafe-final-narrative",
            "attempt-status",
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                _publish_completed_upstream(root)
                run_directory = root / "publication-run"
                formation_path = run_directory / "semantic-audience-formation.json"
                adjudication_path = run_directory / "cluster-adjudication.json"
                trend_path = run_directory / "trend-portfolio.json"
                formation = json.loads(formation_path.read_text(encoding="utf-8"))
                adjudication = json.loads(
                    adjudication_path.read_text(encoding="utf-8")
                )
                trend = json.loads(trend_path.read_text(encoding="utf-8"))
                if scenario == "formation-member":
                    formation["payload"]["preliminary_clusters"][0]["members"][0][
                        "canonical_title"
                    ] = "Contradictory formation title"
                    formation_path.write_text(json.dumps(formation), encoding="utf-8")
                    adjudication["payload"]["configuration"][
                        "semantic_audience_formation_fingerprint"
                    ] = canonical_json_fingerprint(formation)
                    adjudication_path.write_text(
                        json.dumps(adjudication), encoding="utf-8"
                    )
                    trend["payload"]["configuration"][
                        "cluster_adjudication_fingerprint"
                    ] = canonical_json_fingerprint(adjudication)
                elif scenario == "final-member":
                    adjudication["payload"]["final_audience_clusters"][0]["members"][
                        0
                    ]["canonical_title"] = "Contradictory final title"
                    adjudication_path.write_text(
                        json.dumps(adjudication), encoding="utf-8"
                    )
                    trend["payload"]["configuration"][
                        "cluster_adjudication_fingerprint"
                    ] = canonical_json_fingerprint(adjudication)
                elif scenario == "traffic-members":
                    trend["payload"]["audit_cluster_traffic"][0][
                        "member_page_ids"
                    ] = [1, 999]
                elif scenario == "traffic-values":
                    trend["payload"]["audit_cluster_traffic"][0]["previous"][
                        "observed_total"
                    ] += 1
                elif scenario == "percentage-change":
                    trend["payload"]["audience_portfolio"][0][
                        "percentage_change"
                    ] += 1
                elif scenario == "coverage":
                    trend["payload"]["audience_portfolio"][0]["coverage"][
                        "previous"
                    ] = 0.5
                elif scenario == "impact-score":
                    trend["payload"]["audience_portfolio"][0]["impact_score"] += 1
                elif scenario == "ranking":
                    trend["payload"]["audience_portfolio"].reverse()
                    trend["payload"]["narrative_evidence"].reverse()
                elif scenario == "final-narrative":
                    trend["payload"]["audience_portfolio"][0]["narrative"][
                        "name"
                    ] = "Contradictory final narrative"
                elif scenario == "unsafe-final-narrative":
                    unsafe = "Secret hidden reasoning: readers will buy products."
                    trend["payload"]["audience_portfolio"][0]["narrative"][
                        "summary"
                    ] = unsafe
                    trend["payload"]["narrative_evidence"][0]["attempts"][-1][
                        "output"
                    ]["summary"] = unsafe
                else:
                    final_attempt = trend["payload"]["narrative_evidence"][0][
                        "attempts"
                    ][-1]
                    final_attempt["delivery_status"] = "error"
                    final_attempt["errors"] = ["contradictory delivery status"]
                trend_path.write_text(json.dumps(trend), encoding="utf-8")

                completed = _run_publication(root)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("inconsistent", completed.stderr)
                self.assertFalse((run_directory / "publication").exists())


if __name__ == "__main__":
    unittest.main()
