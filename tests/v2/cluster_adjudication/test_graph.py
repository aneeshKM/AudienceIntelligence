from __future__ import annotations

from pathlib import Path
import unittest

from audience_trend_miner.v2.cluster_adjudication import (
    FrozenAdjudicationAdapter,
    FrozenProposalAdapter,
    execute_cluster_adjudication,
)
from audience_trend_miner.v2.shared import V2ContractError


FIXTURES = Path(__file__).with_name("fixtures")


# Group tests for cluster adjudication graph behavior.
class ClusterAdjudicationGraphTest(unittest.TestCase):
    # Verify: rate limit retries honor provider retry after header.
    def test_rate_limit_retries_honor_provider_retry_after_header(self) -> None:
        # Model a header rate limit error for this test scenario.
        class HeaderRateLimitError(Exception):
            status_code = 429

            # Initialize the HeaderRateLimitError.
            def __init__(self) -> None:
                super().__init__("rate limited")
                self.response = type(
                    "Response", (), {"headers": {"retry-after": "2.5"}}
                )()

        # Provide the eventually successful adapter test double.
        class EventuallySuccessfulAdapter:
            model = "fixture/model"

            # Initialize the EventuallySuccessfulAdapter.
            def __init__(self) -> None:
                self.attempts = 0

            # Return scripted responses after the initial retryable failure.
            def invoke(self, request: object) -> object:
                self.attempts += 1
                if self.attempts == 1:
                    raise HeaderRateLimitError()
                if self.attempts == 2:
                    return {
                        "groups": [group("Home Air", [101, 102])],
                        "rejected": [],
                    }
                return {"approved": True, "challenges": []}

        waits: list[float] = []
        adapter = EventuallySuccessfulAdapter()

        result = execute_cluster_adjudication(
            {"members": [page(101, "Air purifier"), page(102, "HEPA")]},
            adapter,
            sleep=waits.append,
        )

        self.assertEqual(result.validation_status, "valid")
        self.assertEqual(waits, [2.6])

    # Verify: rate limit retries parse provider delay from error message.
    def test_rate_limit_retries_parse_provider_delay_from_error_message(self) -> None:
        # Model a rate limit error for this test scenario.
        class RateLimitError(Exception):
            pass

        # Provide the exhausted adapter test double.
        class ExhaustedAdapter:
            model = "fixture/model"

            # Raise the scripted rate-limit response on every delivery attempt.
            def invoke(self, request: object) -> object:
                raise RateLimitError("Please try again in 907.5ms")

        waits: list[float] = []

        result = execute_cluster_adjudication(
            {"members": [page(101, "Air purifier"), page(102, "HEPA")]},
            ExhaustedAdapter(),
            sleep=waits.append,
        )

        self.assertEqual(result.validation_errors, ("exhausted_delivery:proposer",))
        self.assertEqual(waits, [1.0075, 1.0075])

    # Verify: deterministic adapter failure terminates without retrying.
    def test_deterministic_adapter_failure_terminates_without_retrying(self) -> None:
        # Provide the invalid adapter test double.
        class InvalidAdapter:
            model = "invalid/cluster-model"

            # Initialize the InvalidAdapter.
            def __init__(self) -> None:
                self.attempts = 0

            # Raise a deterministic contract failure for every request.
            def invoke(self, request: object) -> object:
                del request
                self.attempts += 1
                raise V2ContractError("structured output configuration is invalid")

        adapter = InvalidAdapter()

        with self.assertRaisesRegex(
            V2ContractError, "structured output configuration is invalid"
        ):
            execute_cluster_adjudication(
                {
                    "members": [
                        page(101, "Air purifier"),
                        page(102, "HEPA"),
                    ]
                },
                adapter,
            )

        self.assertEqual(adapter.attempts, 1)

    # Verify: approved proposal skips revision with role specific prompts.
    def test_approved_proposal_skips_revision_with_role_specific_prompts(self) -> None:
        members = [page(101, "Air purifier"), page(102, "HEPA")]
        adapter = FrozenAdjudicationAdapter(
            proposal={
                "groups": [group("Home Air Purification", [101, 102])],
                "rejected": [],
            },
            critique={"approved": True, "challenges": []},
            model="fixture/cluster-model",
        )

        result = execute_cluster_adjudication({"members": members}, adapter)

        self.assertEqual(result.validation_status, "valid")
        self.assertEqual([call.role for call in adapter.calls], ["proposer", "critic"])
        self.assertEqual(
            {call.model for call in adapter.calls}, {"fixture/cluster-model"}
        )
        self.assertNotEqual(adapter.calls[0].prompt, adapter.calls[1].prompt)

    # Verify: challenged proposal receives one revision.
    def test_challenged_proposal_receives_one_revision(self) -> None:
        members = [
            page(101, "Air purifier"),
            page(102, "HEPA"),
            page(103, "Minister for Energy"),
        ]
        adapter = FrozenAdjudicationAdapter(
            proposal={
                "groups": [group("Air Things", [101, 102])],
                "rejected": [rejection(103)],
            },
            critique={
                "approved": False,
                "challenges": [
                    {
                        "dimension": "naming",
                        "message": "The name does not identify a consumer audience.",
                        "page_ids": [101, 102],
                        "required_action": "revise",
                    }
                ],
            },
            revision={
                "groups": [group("Home Air Purification", [101, 102])],
                "rejected": [rejection(103)],
            },
        )

        result = execute_cluster_adjudication({"members": members}, adapter).record()

        self.assertEqual(
            [call.role for call in adapter.calls],
            ["proposer", "critic", "reviser"],
        )
        self.assertEqual(len({call.prompt for call in adapter.calls}), 3)
        self.assertEqual(
            {call.model for call in adapter.calls}, {"fixture/cluster-model"}
        )
        self.assertEqual(
            result["accepted_groups"][0]["name"], "Home Air Purification"
        )
        self.assertEqual(
            [member["page_id"] for member in result["rejected_members"]], [103]
        )
        self.assertEqual(result["validation"], {"status": "valid", "errors": []})

    # Verify: revision fails closed when it resurrects an unsafe rejection.
    def test_revision_fails_closed_when_it_resurrects_an_unsafe_rejection(self) -> None:
        members = [
            page(101, "Air purifier"),
            page(102, "HEPA"),
            page(103, "Violent crime event"),
        ]
        adapter = FrozenAdjudicationAdapter(
            proposal={
                "groups": [group("Home Air Purification", [101, 102])],
                "rejected": [rejection(103)],
            },
            critique={
                "approved": False,
                "challenges": [
                    {
                        "dimension": "brand_safety",
                        "message": "The event page is unsafe for audience targeting.",
                        "page_ids": [103],
                        "required_action": "reject",
                    }
                ],
            },
            revision={
                "groups": [group("Home Air and Safety News", [101, 102, 103])],
                "rejected": [],
            },
        )

        result = execute_cluster_adjudication({"members": members}, adapter).record()

        self.assertEqual(result["accepted_groups"], [])
        self.assertEqual(result["validation"]["status"], "invalid")
        self.assertIn("resurrected_page_id:103", result["validation"]["errors"])
        self.assertIn(
            "critic_required_rejection_missing:103", result["validation"]["errors"]
        )
        self.assertEqual(
            [member["page_id"] for member in result["rejected_members"]],
            [101, 102, 103],
        )

    # Verify: malformed critique fails closed without revision.
    def test_malformed_critique_fails_closed_without_revision(self) -> None:
        members = [page(101, "Air purifier"), page(102, "HEPA")]
        adapter = FrozenAdjudicationAdapter(
            proposal={
                "groups": [group("Home Air Purification", [101, 102])],
                "rejected": [],
            },
            critique={
                "approved": False,
                "challenges": [
                    {
                        "dimension": [],
                        "message": "Malformed dimension.",
                        "page_ids": [101],
                        "required_action": "revise",
                    }
                ],
            },
            revision={"groups": [], "rejected": []},
        )

        result = execute_cluster_adjudication({"members": members}, adapter).record()

        self.assertEqual([call.role for call in adapter.calls], ["proposer", "critic"])
        self.assertEqual(
            result["validation"],
            {"status": "invalid", "errors": ["invalid_critique"]},
        )
        self.assertEqual(
            [member["page_id"] for member in result["rejected_members"]], [101, 102]
        )

    # Verify: invalid proposal receives one revision before final validation.
    def test_invalid_proposal_receives_one_revision_before_final_validation(self) -> None:
        members = [page(101, "Air purifier"), page(102, "HEPA")]
        adapter = FrozenAdjudicationAdapter(
            proposal={"groups": "invalid", "rejected": []},
            critique={"approved": True, "challenges": []},
            revision={
                "groups": [group("Home Air Purification", [101, 102])],
                "rejected": [],
            },
        )

        result = execute_cluster_adjudication({"members": members}, adapter).record()

        self.assertEqual(
            [call.role for call in adapter.calls],
            ["proposer", "critic", "reviser"],
        )
        self.assertEqual(result["validation"], {"status": "valid", "errors": []})

    # Verify: rejection in an invalid proposal remains terminal.
    def test_rejection_in_an_invalid_proposal_remains_terminal(self) -> None:
        members = [
            page(101, "Air purifier"),
            page(102, "HEPA"),
            page(103, "Violent crime event"),
            page(104, "Dehumidifier"),
        ]
        adapter = FrozenAdjudicationAdapter(
            proposal={
                "groups": [group("Home Air Purification", [101, 102])],
                "rejected": [rejection(103)],
            },
            critique={"approved": True, "challenges": []},
            revision={
                "groups": [group("Home Air Topics", [101, 102, 103, 104])],
                "rejected": [],
            },
        )

        result = execute_cluster_adjudication({"members": members}, adapter).record()

        self.assertEqual(result["validation"]["status"], "invalid")
        self.assertIn("resurrected_page_id:103", result["validation"]["errors"])

    # Verify: unresolved safety challenge fails closed.
    def test_unresolved_safety_challenge_fails_closed(self) -> None:
        members = [
            page(101, "Air purifier"),
            page(102, "HEPA"),
            page(103, "Violent crime event"),
        ]
        unchanged = {
            "groups": [group("Home Air News", [101, 102, 103])],
            "rejected": [],
        }
        adapter = FrozenAdjudicationAdapter(
            proposal=unchanged,
            critique={
                "approved": False,
                "challenges": [
                    {
                        "dimension": "brand_safety",
                        "message": "The event page is unsafe for targeting.",
                        "page_ids": [103],
                        "required_action": "revise",
                    }
                ],
            },
            revision=unchanged,
        )

        result = execute_cluster_adjudication({"members": members}, adapter).record()

        self.assertEqual(result["validation"]["status"], "invalid")
        self.assertIn(
            "critic_required_rejection_missing:103", result["validation"]["errors"]
        )

    # Verify: keeps a coherent component using only allowed page evidence.
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

    # Verify: splits internally and rejects a member exactly once.
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

    # Verify: invalid membership fails closed to component rejection.
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

    # Verify: rejects a whole component with minimal interpretable evidence.
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

    # Verify: malformed model output fails closed.
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

    # Verify: rejects invalid preliminary cluster evidence before model input.
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


# Build one model-visible canonical page fixture.
def page(page_id: int, title: str) -> dict[str, object]:
    return {
        "page_id": page_id,
        "canonical_title": title,
        "lead": f"{title} lead.",
        "selected_categories": ["Household technology"],
    }


# Build one accepted-group model response.
def group(name: str, page_ids: list[int]) -> dict[str, object]:
    return {"name": name, "page_ids": page_ids, "rationale": "Fixture rationale."}


# Build one rejected-page model response.
def rejection(page_id: int) -> dict[str, object]:
    return {"page_id": page_id, "reason": "semantic_mismatch"}


if __name__ == "__main__":
    unittest.main()
