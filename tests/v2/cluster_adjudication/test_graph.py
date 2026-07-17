from __future__ import annotations

from pathlib import Path
import unittest

from audience_trend_miner.v2.cluster_adjudication import (
    FrozenProposalAdapter,
    execute_cluster_adjudication,
)
from audience_trend_miner.v2.shared import V2ContractError


FIXTURES = Path(__file__).with_name("fixtures")


class ClusterAdjudicationGraphTest(unittest.TestCase):
    def test_keeps_a_coherent_component_using_only_allowed_page_evidence(self) -> None:
        cluster = {
            "cohesion": 0.91,
            "subdivision": None,
            "members": [
                {
                    "page_id": 101,
                    "canonical_title": "Air purifier",
                    "lead": "An air purifier removes contaminants from indoor air.",
                    "selected_categories": ["Air filters", "Home appliances"],
                    "traffic": 500000,
                    "window": "current",
                },
                {
                    "page_id": 102,
                    "canonical_title": "HEPA",
                    "lead": "HEPA is an efficiency standard for air filters.",
                    "selected_categories": ["Air filters"],
                    "traffic": 450000,
                },
            ],
        }
        adapter = FrozenProposalAdapter.from_file(FIXTURES / "keep_component.json")

        result = execute_cluster_adjudication(cluster, adapter)

        self.assertEqual(
            result.record(),
            {
                "accepted_groups": [
                    {
                        "name": "Home Air Purification",
                        "rationale": (
                            "Shared consumer interest in residential "
                            "air-cleaning products."
                        ),
                        "members": [
                            {
                                key: member[key]
                                for key in (
                                    "page_id",
                                    "canonical_title",
                                    "lead",
                                    "selected_categories",
                                )
                            }
                            for member in cluster["members"]
                        ],
                    }
                ],
                "rejected_members": [],
                "validation": {"status": "valid", "errors": []},
            },
        )
        self.assertEqual(
            adapter.model_inputs,
            [
                [
                    {
                        key: member[key]
                        for key in (
                            "page_id",
                            "canonical_title",
                            "lead",
                            "selected_categories",
                        )
                    }
                    for member in cluster["members"]
                ]
            ],
        )

    def test_splits_internally_and_rejects_a_member_exactly_once(self) -> None:
        members = [
            page(101, "Air purifier"),
            page(102, "HEPA"),
            page(103, "Air conditioning"),
            page(104, "Heat pump"),
            page(105, "Minister for Energy"),
        ]

        result = execute_cluster_adjudication(
            {"cohesion": 0.52, "subdivision": None, "members": members},
            FrozenProposalAdapter.from_file(FIXTURES / "split_and_reject.json"),
        ).record()

        self.assertEqual(
            [
                [member["page_id"] for member in group["members"]]
                for group in result["accepted_groups"]
            ],
            [[101, 102], [103, 104]],
        )
        self.assertEqual(
            result["rejected_members"],
            [
                {
                    "page_id": 105,
                    "canonical_title": "Minister for Energy",
                    "reason": "routine_politics",
                }
            ],
        )
        terminal_ids = [
            member["page_id"]
            for group in result["accepted_groups"]
            for member in group["members"]
        ] + [member["page_id"] for member in result["rejected_members"]]
        self.assertCountEqual(terminal_ids, [101, 102, 103, 104, 105])

    def test_invalid_membership_fails_closed_to_component_rejection(self) -> None:
        members = [page(101, "Air purifier"), page(102, "HEPA"), page(103, "Fan")]
        scenarios = (
            (
                {
                    "groups": [group("Air cleaning", [101, 999])],
                    "rejected": [rejection(102), rejection(103)],
                },
                "unknown_page_id:999",
            ),
            (
                {
                    "groups": [group("Air cleaning", [101, 101])],
                    "rejected": [rejection(102), rejection(103)],
                },
                "duplicate_page_id:101",
            ),
            (
                {
                    "groups": [group("Air cleaning", [101, 102])],
                    "rejected": [],
                },
                "omitted_page_id:103",
            ),
            (
                {
                    "groups": [group("Air cleaning", [101, 102])],
                    "rejected": [rejection(101), rejection(103)],
                },
                "multiply_assigned_page_id:101",
            ),
            (
                {
                    "groups": [group("Air cleaning", [101])],
                    "rejected": [rejection(102), rejection(103)],
                },
                "accepted_group_has_fewer_than_two_distinct_pages:Air cleaning",
            ),
        )

        for proposal, expected_error in scenarios:
            with self.subTest(expected_error=expected_error):
                result = execute_cluster_adjudication(
                    {"members": members}, FrozenProposalAdapter(proposal)
                ).record()

                self.assertEqual(result["accepted_groups"], [])
                self.assertEqual(result["validation"]["status"], "invalid")
                self.assertIn(expected_error, result["validation"]["errors"])
                self.assertEqual(
                    result["rejected_members"],
                    [
                        {
                            "page_id": member["page_id"],
                            "canonical_title": member["canonical_title"],
                            "reason": "invalid_adjudication",
                        }
                        for member in members
                    ],
                )

    def test_rejects_a_whole_component_with_minimal_interpretable_evidence(self) -> None:
        members = [page(201, "Crime event"), page(202, "Crime suspect")]

        result = execute_cluster_adjudication(
            {"members": members},
            FrozenProposalAdapter.from_file(FIXTURES / "reject_component.json"),
        ).record()

        self.assertEqual(result["accepted_groups"], [])
        self.assertEqual(
            result["rejected_members"],
            [
                {
                    "page_id": 201,
                    "canonical_title": "Crime event",
                    "reason": "violent_crime",
                },
                {
                    "page_id": 202,
                    "canonical_title": "Crime suspect",
                    "reason": "violent_crime",
                },
            ],
        )
        self.assertEqual(result["validation"], {"status": "valid", "errors": []})

    def test_malformed_model_output_fails_closed(self) -> None:
        members = [page(101, "Air purifier"), page(102, "HEPA")]

        result = execute_cluster_adjudication(
            {"members": members}, FrozenProposalAdapter({"groups": "not-a-list"})
        ).record()

        self.assertEqual(result["accepted_groups"], [])
        self.assertEqual(
            result["validation"],
            {"status": "invalid", "errors": ["invalid_proposal_shape"]},
        )

    def test_rejects_invalid_preliminary_cluster_evidence_before_model_input(self) -> None:
        scenarios = (
            [page(101, "Duplicate"), page(101, "Duplicate")],
            [{**page(101, ""), "canonical_title": ""}],
            [{**page(101, "Long lead"), "lead": "x" * 601}],
            [
                {
                    **page(101, "Too many categories"),
                    "selected_categories": ["a", "b", "c", "d", "e", "f"],
                }
            ],
            [
                {
                    **page(101, "Duplicate categories"),
                    "selected_categories": ["same", "same"],
                }
            ],
        )

        for members in scenarios:
            with self.subTest(members=members):
                adapter = FrozenProposalAdapter({"groups": [], "rejected": []})
                with self.assertRaises(V2ContractError):
                    execute_cluster_adjudication({"members": members}, adapter)
                self.assertEqual(adapter.model_inputs, [])


def page(page_id: int, title: str) -> dict[str, object]:
    return {
        "page_id": page_id,
        "canonical_title": title,
        "lead": f"{title} lead.",
        "selected_categories": ["Household technology"],
    }


def group(name: str, page_ids: list[int]) -> dict[str, object]:
    return {"name": name, "page_ids": page_ids, "rationale": "Fixture rationale."}


def rejection(page_id: int) -> dict[str, object]:
    return {"page_id": page_id, "reason": "semantic_mismatch"}


if __name__ == "__main__":
    unittest.main()
