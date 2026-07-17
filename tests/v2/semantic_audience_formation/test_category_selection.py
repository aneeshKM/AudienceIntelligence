from __future__ import annotations

import math
import unittest

from audience_trend_miner.v2.semantic_audience_formation import select_categories


class SelectedCategoryTest(unittest.TestCase):
    def test_filters_audited_noise_and_ranks_top_five_by_whole_universe_idf(self) -> None:
        pages = [
            {
                "page_id": 3,
                "canonical_title": "Sparse",
                "lead": "Sparse evidence.",
                "categories": ["Shared", "Sparse only"],
            },
            {
                "page_id": 1,
                "canonical_title": "Rich",
                "lead": "Rich evidence.",
                "categories": [
                    "Shared",
                    "Rare Z",
                    "Rare A",
                    "Fourth",
                    "Fifth",
                    "Sixth",
                    "Rare A",
                    "Living people",
                    "1999 births",
                    "Articles with short description",
                    "Wikipedia:Featured topics",
                ],
            },
            {
                "page_id": 2,
                "canonical_title": "Empty",
                "lead": "No useful category evidence.",
                "categories": ["All articles lacking sources", "CS1 errors"],
            },
        ]

        selection = select_categories(pages)

        self.assertEqual(selection.rule_set["version"], "1.0")
        self.assertEqual(
            selection.rule_set["hidden_categories"],
            "excluded by Wikimedia Evidence provenance",
        )
        self.assertIn(r"^Living people$", selection.rule_set["noise_patterns"])
        self.assertEqual(
            [(page.page_id, page.selected_categories) for page in selection.pages],
            [
                (1, ("Fifth", "Fourth", "Rare A", "Rare Z", "Sixth")),
                (2, ()),
                (3, ("Sparse only", "Shared")),
            ],
        )
        self.assertEqual(selection.category_document_frequency["Shared"], 2)
        self.assertAlmostEqual(selection.category_idf["Shared"], math.log(3 / 2))
        self.assertAlmostEqual(selection.category_idf["Rare A"], math.log(3))


if __name__ == "__main__":
    unittest.main()
