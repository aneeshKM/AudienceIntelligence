from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


FIXTURES = Path(__file__).with_name("fixtures")


def page(page_id: int, title: str) -> dict[str, object]:
    return {
        "page_id": page_id,
        "canonical_title": title,
        "lead": f"{title} lead.",
        "selected_categories": ["Consumer topics"],
    }


def publish_formation(output_dir: Path, run_id: str, clusters: list[list[dict[str, object]]]) -> Path:
    run_directory = output_dir / run_id
    run_directory.mkdir(parents=True)
    artifact = {
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
                "wikimedia_evidence_fingerprint": "sha256:" + "0" * 64,
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
                {"cohesion": 0.8, "subdivision": None, "members": members}
                for members in clusters
            ],
            "completion": {"status": "complete"},
        },
    }
    path = run_directory / "semantic-audience-formation.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def run_stage(
    output_dir: Path,
    run_id: str = "adjudication-run",
    *extra: str,
    fixture: str = "stage_decisions.json",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "audience_trend_miner",
            "v2-cluster-adjudication",
            "--run-id",
            run_id,
            "--output-dir",
            str(output_dir),
            "--fixture",
            str(FIXTURES / fixture),
            "--progress-format",
            "json",
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


class ClusterAdjudicationStageTest(unittest.TestCase):
    def test_atomic_failure_resumes_completed_clusters_without_repeating_them(self) -> None:
        clusters = [
            [page(101, "Air purifier"), page(102, "HEPA")],
            [
                page(201, "Air purifier"),
                page(202, "HEPA"),
                page(203, "Air conditioning"),
                page(204, "Heat pump"),
                page(205, "Minister for Energy"),
            ],
            [page(301, "Crime event"), page(302, "Crime suspect")],
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            run_directory = output_dir / "adjudication-run"
            publish_formation(output_dir, "adjudication-run", clusters)

            interrupted = run_stage(output_dir, "adjudication-run", "--interrupt-before-completion")

            self.assertNotEqual(interrupted.returncode, 0)
            self.assertIn("interrupted before artifact completion", interrupted.stderr)
            self.assertFalse((run_directory / "cluster-adjudication.json").exists())
            self.assertTrue((run_directory / ".cluster-adjudication.checkpoint.json").exists())

            resumed = run_stage(
                output_dir,
                "adjudication-run",
                fixture="resume_guard_decisions.json",
            )

            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            payload = json.loads(
                (run_directory / "cluster-adjudication.json").read_text()
            )["payload"]
            self.assertEqual(payload["counts"]["accepted_pages"], 6)
            events = [json.loads(line) for line in resumed.stdout.splitlines()]
            self.assertEqual(
                [event["operation"] for event in events],
                ["resume-cluster", "resume-cluster", "resume-cluster", "publish"],
            )

            completed_resume = run_stage(
                output_dir,
                "adjudication-run",
                fixture="resume_guard_decisions.json",
            )
            self.assertEqual(completed_resume.returncode, 0, completed_resume.stderr)
            self.assertEqual(
                [json.loads(line)["operation"] for line in completed_resume.stdout.splitlines()],
                ["resume"],
            )

    def test_stage_retries_delivery_three_times_and_fails_exhausted_step_closed(self) -> None:
        clusters = [
            [page(401, "Air purifier"), page(402, "Air filter")],
            [page(501, "Air conditioner"), page(502, "Heat pump")],
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            publish_formation(output_dir, "retry-run", clusters)

            completed = run_stage(
                output_dir,
                "retry-run",
                fixture="retry_decisions.json",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(
                (output_dir / "retry-run" / "cluster-adjudication.json").read_text()
            )["payload"]
            first_proposal = payload["adjudications"][0]["steps"][0]
            self.assertEqual(first_proposal["status"], "completed")
            self.assertEqual(
                [attempt["delivery_status"] for attempt in first_proposal["attempts"]],
                ["error", "error", "delivered"],
            )
            exhausted_critic = payload["adjudications"][1]["steps"][1]
            self.assertEqual(exhausted_critic["status"], "exhausted")
            self.assertEqual(exhausted_critic["validation_status"], "not_run")
            self.assertEqual(len(exhausted_critic["attempts"]), 3)
            self.assertEqual(
                payload["adjudications"][1]["validation"],
                {"status": "invalid", "errors": ["exhausted_delivery:critic"]},
            )
            self.assertEqual(
                [member["page_id"] for member in payload["rejected_members"]],
                [501, 502],
            )
            self.assertNotIn("reviser", [
                step["role"] for step in payload["adjudications"][1]["steps"]
            ])

    def test_stage_adjudicates_every_selected_cluster_to_exclusive_terminal_states(self) -> None:
        clusters = [
            [page(101, "Air purifier"), page(102, "HEPA")],
            [
                page(201, "Air purifier"),
                page(202, "HEPA"),
                page(203, "Air conditioning"),
                page(204, "Heat pump"),
                page(205, "Minister for Energy"),
            ],
            [page(301, "Crime event"), page(302, "Crime suspect")],
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            publish_formation(output_dir, "adjudication-run", clusters)

            completed = run_stage(output_dir)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            artifact = json.loads(
                (output_dir / "adjudication-run" / "cluster-adjudication.json").read_text()
            )
            payload = artifact["payload"]
            self.assertEqual(artifact["status"], "complete")
            self.assertEqual(payload["counts"], {
                "preliminary_clusters": 3,
                "final_audience_clusters": 3,
                "accepted_pages": 6,
                "rejected_pages": 3,
            })
            terminal_ids = [
                member["page_id"]
                for cluster in payload["final_audience_clusters"]
                for member in cluster["members"]
            ] + [member["page_id"] for member in payload["rejected_members"]]
            self.assertCountEqual(terminal_ids, [member["page_id"] for cluster in clusters for member in cluster])
            self.assertEqual(len(terminal_ids), len(set(terminal_ids)))
            self.assertEqual(
                [member["page_id"] for member in payload["rejected_members"]],
                [205, 301, 302],
            )
            self.assertEqual(
                [step["role"] for item in payload["adjudications"] for step in item["steps"]],
                ["proposer", "critic", "proposer", "critic", "reviser", "proposer", "critic"],
            )
            events = [json.loads(line) for line in completed.stdout.splitlines()]
            self.assertEqual(events[-1]["operation"], "publish")
            self.assertTrue(any(event["operation"] == "critic" for event in events))
            self.assertTrue(any(event["operation"] == "reviser" for event in events))


if __name__ == "__main__":
    unittest.main()
