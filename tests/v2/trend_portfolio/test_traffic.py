from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audience_trend_miner.v2.shared import V2ContractError
from audience_trend_miner.v2.trend_portfolio import attach_cluster_traffic


PREVIOUS_DAYS = [f"2026-07-{day:02d}" for day in range(2, 6)]
CURRENT_DAYS = [f"2026-07-{day:02d}" for day in range(9, 16)]


def _member(page_id: int) -> dict[str, object]:
    return {
        "page_id": page_id,
        "canonical_title": f"Page {page_id}",
        "lead": f"Page {page_id} lead.",
        "selected_categories": ["Consumer topics"],
    }


def _artifacts(root: Path, *, run_id: str = "trend-run") -> tuple[Path, Path]:
    run_directory = root / run_id
    run_directory.mkdir(parents=True)
    observations: dict[int, list[dict[str, object]]] = {
        1: [{"date": PREVIOUS_DAYS[0], "views_ceil": 100}],
        2: [{"date": PREVIOUS_DAYS[0], "views_ceil": 100}],
        3: [
            *({"date": day, "views_ceil": 200} for day in PREVIOUS_DAYS),
            *({"date": day, "views_ceil": 100} for day in CURRENT_DAYS),
        ],
        4: [
            *({"date": day, "views_ceil": 200} for day in PREVIOUS_DAYS),
            *({"date": day, "views_ceil": 100} for day in CURRENT_DAYS),
        ],
        5: [
            *({"date": day, "views_ceil": 100} for day in PREVIOUS_DAYS),
            *({"date": day, "views_ceil": 199} for day in CURRENT_DAYS),
        ],
        6: [
            *({"date": day, "views_ceil": 100} for day in PREVIOUS_DAYS),
            *({"date": day, "views_ceil": 199} for day in CURRENT_DAYS),
        ],
    }
    for page_id in (1, 2):
        observations[page_id].extend(
            {"date": day, "views_ceil": 200} for day in CURRENT_DAYS
        )

    evidence = {
        "schema_version": "2.0",
        "run_id": run_id,
        "stage": "wikimedia-evidence",
        "status": "complete",
        "payload": {
            "as_of_date": "2026-07-17",
            "nominal_windows": {
                "previous": {"start": PREVIOUS_DAYS[0], "end": "2026-07-08"},
                "current": {"start": CURRENT_DAYS[0], "end": CURRENT_DAYS[-1]},
            },
            "nominal_days": [
                *(
                    {"date": day, "window": "previous", "status": "successful"}
                    for day in PREVIOUS_DAYS
                ),
                *(
                    {
                        "date": day,
                        "window": "previous",
                        "status": "unavailable",
                    }
                    for day in ("2026-07-06", "2026-07-07", "2026-07-08")
                ),
                *(
                    {"date": day, "window": "current", "status": "successful"}
                    for day in CURRENT_DAYS
                ),
            ],
            "coverage": {"previous": 4, "current": 7},
            "candidate_universe": [f"Page_{page_id}" for page_id in range(1, 7)],
            "canonical_pages": [
                {
                    "page_id": page_id,
                    "canonical_title": f"Page {page_id}",
                    "lead": f"Page {page_id} lead.",
                    "categories": ["Consumer topics"],
                    "aliases": [f"Page_{page_id}"],
                    "observations": observations[page_id],
                }
                for page_id in range(1, 7)
            ],
            "daily_cutoffs": [
                {"date": day, "views_ceil": 50}
                for day in [*PREVIOUS_DAYS, *CURRENT_DAYS]
            ],
            "provenance": {
                "source": "fixture:top-per-country/US",
                "country": "US",
                "project": "en.wikipedia",
                "traffic_measure": "views_ceil",
                "category_visibility": "non-hidden",
            },
            "exclusions": {
                "non_en_wikipedia_records": 0,
                "unavailable_days": [
                    "2026-07-06",
                    "2026-07-07",
                    "2026-07-08",
                ],
                "metadata_pages_unavailable": 0,
                "main_page": 0,
                "internal_namespaces": {},
            },
            "completion": {
                "status": "complete",
                "minimum_successful_days_per_window": 4,
            },
        },
    }
    adjudication = {
        "schema_version": "2.0",
        "run_id": run_id,
        "stage": "cluster-adjudication",
        "status": "complete",
        "payload": {
            "configuration": {
                "model": "fixture/model",
                "framework": {"name": "langgraph", "version": "1.0"},
                "integration": {"name": "fixture", "version": "1.0"},
                "semantic_audience_formation_fingerprint": "sha256:" + "0" * 64,
            },
            "counts": {
                "preliminary_clusters": 3,
                "final_audience_clusters": 3,
                "accepted_pages": 6,
                "rejected_pages": 0,
            },
            "final_audience_clusters": [
                {
                    "cluster_id": f"final-audience-cluster-{index:04d}",
                    "source_preliminary_cluster_id": f"preliminary-cluster-{index:04d}",
                    "name": name,
                    "rationale": f"{name} rationale.",
                    "members": [_member(first), _member(first + 1)],
                }
                for index, (name, first) in enumerate(
                    (("Growing", 1), ("Shrinking", 3), ("Touching", 5)),
                    start=1,
                )
            ],
            "rejected_members": [],
            "adjudications": [
                {
                    "preliminary_cluster_id": f"preliminary-cluster-{index:04d}",
                    "steps": [
                        {
                            "role": "proposer",
                            "status": "completed",
                            "validation_status": "valid",
                            "attempts": [
                                {
                                    "attempt": 1,
                                    "delivery_status": "delivered",
                                    "error": None,
                                }
                            ],
                        }
                    ],
                    "validation": {"status": "valid", "errors": []},
                }
                for index in range(1, 4)
            ],
            "completion": {"status": "complete"},
        },
    }
    evidence_path = run_directory / "wikimedia-evidence.json"
    adjudication_path = run_directory / "cluster-adjudication.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    adjudication_path.write_text(json.dumps(adjudication), encoding="utf-8")
    return evidence_path, adjudication_path


class ClusterTrafficTest(unittest.TestCase):
    def test_attaches_censored_traffic_and_classifies_non_overlapping_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evidence_path, adjudication_path = _artifacts(Path(temporary_directory))

            trends = attach_cluster_traffic(
                run_id="trend-run",
                wikimedia_evidence_path=evidence_path,
                cluster_adjudication_path=adjudication_path,
            )

            self.assertEqual(
                [trend.direction for trend in trends],
                ["robust_growth", "robust_shrinking", "uncertain_direction"],
            )
            growing = trends[0]
            self.assertEqual(growing.previous.observed_total, 200)
            self.assertEqual(growing.previous.observed_page_days, 2)
            self.assertEqual(growing.previous.successful_days, 4)
            self.assertEqual(growing.previous.seven_day_equivalent, 350)
            self.assertEqual(growing.previous.minimum, 3.5)
            self.assertEqual(growing.previous.maximum, 875)
            self.assertEqual(growing.current.minimum, 1414)
            self.assertEqual(growing.current.maximum, 2800)
            self.assertEqual(trends[2].previous.maximum, trends[2].current.minimum)

    def test_rejects_duplicate_terminal_membership_before_traffic_is_attached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evidence_path, adjudication_path = _artifacts(Path(temporary_directory))
            adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
            adjudication["payload"]["final_audience_clusters"][1]["members"][0] = (
                adjudication["payload"]["final_audience_clusters"][0]["members"][0]
            )
            adjudication_path.write_text(json.dumps(adjudication), encoding="utf-8")

            with self.assertRaisesRegex(V2ContractError, "at most one Final Audience Cluster"):
                attach_cluster_traffic(
                    run_id="trend-run",
                    wikimedia_evidence_path=evidence_path,
                    cluster_adjudication_path=adjudication_path,
                )

    def test_rejects_artifacts_from_different_runs_and_missing_cutoff_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence_path, adjudication_path = _artifacts(root)
            adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
            adjudication["run_id"] = "another-run"
            adjudication_path.write_text(json.dumps(adjudication), encoding="utf-8")

            with self.assertRaisesRegex(V2ContractError, "different run facts"):
                attach_cluster_traffic(
                    run_id="trend-run",
                    wikimedia_evidence_path=evidence_path,
                    cluster_adjudication_path=adjudication_path,
                )

            _, adjudication_path = _artifacts(root, run_id="cutoff-run")
            evidence_path = root / "cutoff-run" / "wikimedia-evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["payload"]["daily_cutoffs"][1]["views_ceil"] = None
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

            with self.assertRaisesRegex(V2ContractError, "cutoff"):
                attach_cluster_traffic(
                    run_id="cutoff-run",
                    wikimedia_evidence_path=evidence_path,
                    cluster_adjudication_path=adjudication_path,
                )


if __name__ == "__main__":
    unittest.main()
