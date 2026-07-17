from __future__ import annotations

import unittest

from audience_trend_miner.mediawiki_source_contract import (
    Observation,
    RestSource,
    summarize_action_response,
    summarize_country_response,
    summarize_rest_observation,
    views_ceil_bounds,
)


class MediaWikiSourceContractExperimentTest(unittest.TestCase):
    def test_action_summary_preserves_redirect_missing_and_metadata_evidence(self) -> None:
        summary = summarize_action_response(
            {
                "continue": {"clcontinue": "42|Category", "continue": "||extracts"},
                "query": {
                    "redirects": [{"from": "Alias", "to": "Canonical"}],
                    "pages": [
                        {
                            "pageid": 42,
                            "title": "Canonical",
                            "extract": "A useful lead.",
                            "categories": [{"title": "Category:Examples"}],
                        },
                        {"title": "Definitely missing", "missing": True},
                    ],
                },
            }
        )

        self.assertEqual(summary["redirects"], [{"from": "Alias", "to": "Canonical"}])
        self.assertEqual(summary["missing_titles"], ["Definitely missing"])
        self.assertEqual(
            summary["pages"],
            [
                {
                    "page_id": 42,
                    "canonical_title": "Canonical",
                    "lead_characters": 14,
                    "visible_category_count": 1,
                }
            ],
        )
        self.assertTrue(summary["has_continuation"])

    def test_country_summary_exposes_cutoff_and_ceiling_rounding_bounds(self) -> None:
        summary = summarize_country_response(
            {
                "items": [
                    {
                        "articles": [
                            {"article": "A", "project": "en.wikipedia", "views_ceil": 4_800},
                            {"article": "B", "project": "de.wikipedia", "views_ceil": 4_700},
                        ]
                    }
                ]
            }
        )

        self.assertEqual(summary["published_record_count"], 2)
        self.assertEqual(summary["daily_cutoff_views_ceil"], 4_700)
        self.assertTrue(summary["all_views_ceil_multiples_of_100"])
        self.assertEqual(views_ceil_bounds(4_700), (4_601, 4_700))

    def test_rest_summary_associates_identity_lead_and_redirect_hops_with_title(self) -> None:
        summary = summarize_rest_observation(
            RestSource.SUMMARY,
            "USA",
            Observation(
                status=200,
                seconds=0.1,
                payload={"pageid": 3434750, "title": "United States", "extract": "Lead."},
                retry_after=None,
                rate_limit=None,
                wire_requests=2,
                final_url="https://en.wikipedia.org/api/rest_v1/page/summary/United_States",
            ),
        )

        self.assertEqual(summary["requested_title"], "USA")
        self.assertEqual(summary["page_id"], 3434750)
        self.assertEqual(summary["canonical_title"], "United States")
        self.assertEqual(summary["lead"], {"format": "plain_text_summary", "characters": 5})
        self.assertEqual(summary["redirect_hops"], 1)
        self.assertFalse(summary["visible_categories_available"])


if __name__ == "__main__":
    unittest.main()
