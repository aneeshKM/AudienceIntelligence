from __future__ import annotations

import unittest
from datetime import date

from audience_trend_miner.wikimedia import HttpWikimediaAdapter


class RecordingJsonTransport:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get_json(self, url: str) -> object:
        self.urls.append(url)
        if "/top/" in url:
            return {"items": [{"articles": [{"article": "Raw_Title"}]}]}
        if "/per-article/" in url:
            return {"items": [{"timestamp": "2026070100", "views": 12}]}
        return {
            "query": {
                "pages": [
                    {
                        "pageid": 123,
                        "title": "Canonical Title",
                        "extract": "Lead text.",
                        "categories": [{"title": "Category:Examples"}],
                    }
                ]
            }
        }


class HttpWikimediaAdapterTest(unittest.TestCase):
    def test_translates_wikimedia_operations_and_response_envelopes(self) -> None:
        transport = RecordingJsonTransport()
        adapter = HttpWikimediaAdapter(transport=transport)

        discovery = adapter.daily_top_pages(date(2026, 7, 8))
        pageviews = adapter.article_pageviews(
            "Raw/Title", date(2026, 7, 1), date(2026, 7, 14)
        )
        metadata = adapter.article_metadata("Raw/Title")

        self.assertEqual(discovery.titles, ("Raw_Title",))
        self.assertEqual(pageviews.daily_views[0].date, date(2026, 7, 1))
        self.assertEqual(
            (metadata.page_id, metadata.canonical_title, metadata.categories),
            (123, "Canonical Title", ("Examples",)),
        )
        self.assertEqual(
            transport.urls,
            [
                "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/2026/07/08",
                "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Raw%2FTitle/daily/2026070100/2026071400",
                "https://en.wikipedia.org/w/api.php?action=query&format=json&formatversion=2&redirects=1&prop=extracts%7Ccategories&exintro=1&explaintext=1&titles=Raw%2FTitle",
            ],
        )


if __name__ == "__main__":
    unittest.main()
