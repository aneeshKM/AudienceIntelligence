from __future__ import annotations

import unittest
from datetime import date
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from audience_trend_miner.wikimedia import (
    HttpWikimediaAdapter,
    UrllibJsonTransport,
    WikimediaTransientError,
)


class RecordingJsonTransport:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get_json(self, url: str) -> object:
        self.urls.append(url)
        if "/top-per-country/" in url:
            return {
                "items": [
                    {
                        "articles": [
                            {
                                "project": "en.wikipedia",
                                "article": "Raw_Title",
                                "views_ceil": 12,
                            },
                            {
                                "project": "de.wikipedia",
                                "article": "Anderer_Titel",
                                "views_ceil": 7,
                            },
                        ]
                    }
                ]
            }
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
    def test_metadata_batch_follows_continuation_and_preserves_alias_lineage(self) -> None:
        class ContinuingTransport:
            def __init__(self):
                self.urls = []

            def get_json(self, url):
                self.urls.append(url)
                if len(self.urls) == 1:
                    return {
                        "continue": {"clcontinue": "42|Later", "continue": "||"},
                        "query": {
                            "redirects": [
                                {"from": "Alias_A", "to": "Intermediate"},
                                {"from": "Intermediate", "to": "Canonical A"},
                            ],
                            "pages": [{"pageid": 42, "title": "Canonical A", "extract": "x" * 650, "categories": [{"title": "Category:First"}]}],
                        },
                    }
                return {"query": {"pages": [{"pageid": 42, "title": "Canonical A", "categories": [{"title": "Category:Later"}]}]}}

        transport = ContinuingTransport()
        result = HttpWikimediaAdapter(transport=transport).metadata_batch(("Alias_A", "Missing"))

        self.assertEqual(len(transport.urls), 2)
        first = parse_qs(urlparse(transport.urls[0]).query)
        self.assertEqual(first["titles"], ["Alias_A|Missing"])
        self.assertEqual(first["clshow"], ["!hidden"])
        self.assertEqual(first["cllimit"], ["max"])
        self.assertEqual(first["maxlag"], ["5"])
        second = parse_qs(urlparse(transport.urls[1]).query)
        self.assertEqual(second["clcontinue"], ["42|Later"])
        self.assertEqual(result.pages[0].extract, "x" * 600)
        self.assertEqual(result.pages[0].categories, ("First", "Later"))
        self.assertEqual(result.aliases, {"Alias_A": 42})

    def test_country_top_pages_uses_us_all_access_contract(self) -> None:
        transport = RecordingJsonTransport()
        adapter = HttpWikimediaAdapter(transport=transport)

        response = adapter.daily_country_top_pages(date(2026, 7, 8))

        self.assertEqual(
            transport.urls,
            [
                "https://wikimedia.org/api/rest_v1/metrics/pageviews/"
                "top-per-country/US/all-access/2026/07/08"
            ],
        )
        self.assertEqual(
            [
                (record.project, record.article, record.views_ceil)
                for record in response.records
            ],
            [("en.wikipedia", "Raw_Title", 12), ("de.wikipedia", "Anderer_Titel", 7)],
        )

    def test_transport_exposes_retry_after_for_throttling(self) -> None:
        error = __import__("urllib.error").error.HTTPError(
            "https://wikimedia.example/data",
            429,
            "throttled",
            {"Retry-After": "9"},
            None,
        )
        with patch("audience_trend_miner.wikimedia.urlopen", side_effect=error):
            with self.assertRaises(WikimediaTransientError) as raised:
                UrllibJsonTransport(request_interval_seconds=0).get_json(
                    "https://wikimedia.example/data"
                )

        self.assertEqual(raised.exception.retry_after_seconds, 9.0)

    def test_transport_paces_consecutive_requests(self) -> None:
        ssl_context = MagicMock()

        with (
            patch("audience_trend_miner.wikimedia.urlopen") as urlopen,
            patch(
                "audience_trend_miner.wikimedia.time.monotonic",
                side_effect=(10.0, 10.0, 10.0, 10.2),
            ),
            patch("audience_trend_miner.wikimedia.time.sleep") as sleep,
        ):
            urlopen.side_effect = (
                BytesIO(b'{"ok": true}'),
                BytesIO(b'{"ok": true}'),
            )
            transport = UrllibJsonTransport(
                ssl_context=ssl_context, request_interval_seconds=0.2
            )
            transport.get_json("https://wikimedia.example/first")
            transport.get_json("https://wikimedia.example/second")

        sleep.assert_called_once_with(0.1999999999999993)

    def test_transport_prefers_macos_system_ca_bundle(self) -> None:
        ssl_context = MagicMock()

        with (
            patch("audience_trend_miner.wikimedia.MACOS_CA_BUNDLE") as ca_bundle,
            patch(
                "audience_trend_miner.wikimedia.ssl.create_default_context",
                return_value=ssl_context,
            ) as create_default_context,
        ):
            ca_bundle.is_file.return_value = True
            ca_bundle.__str__.return_value = "/etc/ssl/cert.pem"
            transport = UrllibJsonTransport()

        create_default_context.assert_called_once_with(cafile="/etc/ssl/cert.pem")
        self.assertIs(transport._ssl_context, ssl_context)

    def test_transport_supplies_a_trusted_ssl_context(self) -> None:
        response = BytesIO(b'{"ok": true}')
        ssl_context = MagicMock()

        with patch("audience_trend_miner.wikimedia.urlopen") as urlopen:
            urlopen.return_value = response
            result = UrllibJsonTransport(ssl_context=ssl_context).get_json(
                "https://wikimedia.example/data"
            )

        self.assertEqual(result, {"ok": True})
        self.assertIs(urlopen.call_args.kwargs["context"], ssl_context)

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
