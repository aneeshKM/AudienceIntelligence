from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
import json
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from audience_trend_miner.wikimedia import USER_AGENT, trusted_ssl_context


ACTION_API = "https://en.wikipedia.org/w/api.php"
SUMMARY_REST_API = "https://en.wikipedia.org/api/rest_v1/page/summary"
CORE_REST_API = "https://en.wikipedia.org/w/rest.php/v1/page"
COUNTRY_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top-per-country"


class RestSource(Enum):
    SUMMARY = "summary_rest"
    CORE_WITH_HTML = "core_rest_with_html"


@dataclass(frozen=True)
class Observation:
    status: int
    seconds: float
    payload: object | None
    retry_after: str | None
    rate_limit: str | None
    wire_requests: int
    final_url: str


def views_ceil_bounds(value: int) -> tuple[int, int]:
    """Return the inclusive integer bounds implied by ceiling-to-100 publication."""
    if value < 0 or value % 100:
        raise ValueError("views_ceil must be a non-negative multiple of 100")
    return (max(0, value - 99), value)


def summarize_action_response(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or not isinstance(raw.get("query"), dict):
        raise ValueError("invalid Action API response")
    query = raw["query"]
    pages = query.get("pages", [])
    if not isinstance(pages, list):
        raise ValueError("invalid Action API pages")
    present = []
    missing = []
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("invalid Action API page")
        if page.get("missing") is True:
            missing.append(str(page["title"]))
            continue
        categories = page.get("categories", [])
        present.append(
            {
                "page_id": int(page["pageid"]),
                "canonical_title": str(page["title"]),
                "lead_characters": len(str(page.get("extract", ""))),
                "visible_category_count": len(categories),
            }
        )
    redirects = [
        {"from": str(item["from"]), "to": str(item["to"])}
        for item in query.get("redirects", [])
    ]
    return {
        "redirects": redirects,
        "missing_titles": sorted(missing),
        "pages": sorted(present, key=lambda page: int(page["page_id"])),
        "has_continuation": "continue" in raw,
    }


def summarize_country_response(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError("invalid top-per-country response")
    try:
        articles = raw["items"][0]["articles"]
        values = [int(article["views_ceil"]) for article in articles]
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid top-per-country response") from error
    cutoff = min(values) if values else None
    return {
        "published_record_count": len(values),
        "daily_cutoff_views_ceil": cutoff,
        "all_views_ceil_multiples_of_100": bool(values)
        and all(value % 100 == 0 for value in values),
        "en_wikipedia_record_count": sum(
            article.get("project") == "en.wikipedia" for article in articles
        ),
    }


def summarize_rest_observation(
    source: RestSource, requested_title: str, observation: Observation
) -> dict[str, object]:
    payload = observation.payload if isinstance(observation.payload, dict) else {}
    is_summary = source is RestSource.SUMMARY
    page_id = payload.get("pageid" if is_summary else "id")
    content = payload.get("extract" if is_summary else "html", "")
    return {
        "requested_title": requested_title,
        "status": observation.status,
        "missing": observation.status == 404,
        "page_id": int(page_id) if page_id is not None else None,
        "canonical_title": payload.get("title"),
        "lead": {
            "format": "plain_text_summary" if is_summary else "rendered_full_html",
            "characters": len(str(content)),
        },
        "visible_categories_available": False,
        "redirect_hops": observation.wire_requests - 1,
        "final_url": observation.final_url,
    }


class CountingRedirectHandler(HTTPRedirectHandler):
    def __init__(self) -> None:
        self.redirect_hops = 0

    def redirect_request(self, *args, **kwargs):
        self.redirect_hops += 1
        return super().redirect_request(*args, **kwargs)


class LiveClient:
    def __init__(self) -> None:
        self._ssl_context = trusted_ssl_context()
        self._redirect_handler = CountingRedirectHandler()
        self._opener = build_opener(
            HTTPSHandler(context=self._ssl_context), self._redirect_handler
        )

    def get(self, url: str) -> Observation:
        started = time.monotonic()
        request = Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        self._redirect_handler.redirect_hops = 0
        try:
            with self._opener.open(request, timeout=30) as response:
                payload = json.load(response)
                return Observation(
                    response.status,
                    time.monotonic() - started,
                    payload,
                    response.headers.get("Retry-After"),
                    response.headers.get("X-RateLimit-Remaining"),
                    self._redirect_handler.redirect_hops + 1,
                    response.geturl(),
                )
        except HTTPError as error:
            payload = None
            try:
                payload = json.load(error)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            return Observation(
                error.code,
                time.monotonic() - started,
                payload,
                error.headers.get("Retry-After"),
                error.headers.get("X-RateLimit-Remaining"),
                self._redirect_handler.redirect_hops + 1,
                error.geturl(),
            )


def _metrics(observations: list[Observation]) -> dict[str, object]:
    return {
        "request_count": sum(item.wire_requests for item in observations),
        "latency_seconds": round(sum(item.seconds for item in observations), 4),
        "statuses": [item.status for item in observations],
        "retry_after_seen": any(item.retry_after is not None for item in observations),
        "rate_limit_remaining": next(
            (
                item.rate_limit
                for item in reversed(observations)
                if item.rate_limit is not None
            ),
            None,
        ),
    }


def run_live_experiment(
    titles: tuple[str, ...], day: date, *, category_limit: str = "max"
) -> dict[str, object]:
    client = LiveClient()
    action_parameters: dict[str, object] = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "maxlag": 5,
        "redirects": 1,
        "converttitles": 1,
        "prop": "extracts|categories",
        "exintro": 1,
        "explaintext": 1,
        "clshow": "!hidden",
        "cllimit": category_limit,
        "titles": "|".join(titles),
    }
    action_observations: list[Observation] = []
    action_summaries: list[dict[str, object]] = []
    while True:
        observation = client.get(f"{ACTION_API}?{urlencode(action_parameters)}")
        action_observations.append(observation)
        if observation.status != 200 or observation.payload is None:
            break
        payload = observation.payload
        action_summaries.append(summarize_action_response(payload))
        if not isinstance(payload, dict):
            break
        continuation = payload.get("continue")
        if not isinstance(continuation, dict):
            break
        action_parameters.update(continuation)

    alternatives: dict[str, object] = {}
    for source, base, suffix in (
        (RestSource.SUMMARY, SUMMARY_REST_API, ""),
        (RestSource.CORE_WITH_HTML, CORE_REST_API, "/with_html"),
    ):
        observations = [
            client.get(f"{base}/{quote(title, safe='')}{suffix}")
            for title in titles
        ]
        alternatives[source.value] = {
            **_metrics(observations),
            "batching": {"supported": False, "title_limit": 1},
            "pages": [
                summarize_rest_observation(source, title, observation)
                for title, observation in zip(titles, observations)
            ],
            "response_keys": sorted(
                {
                    key
                    for item in observations
                    if isinstance(item.payload, dict)
                    for key in item.payload
                }
            ),
        }

    country = client.get(f"{COUNTRY_API}/US/all-access/{day:%Y/%m/%d}")
    country_summary = (
        summarize_country_response(country.payload)
        if country.status == 200 and country.payload is not None
        else None
    )
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"titles": list(titles), "country_day": day.isoformat()},
        "action_query": {
            **_metrics(action_observations),
            "batching": {"supported": True, "title_limit": 50},
            "category_limit": category_limit,
            "responses": action_summaries,
        },
        "alternatives": alternatives,
        "country_top_pages": {**_metrics([country]), "summary": country_summary},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare bounded MediaWiki metadata evidence without saving raw responses."
    )
    parser.add_argument(
        "--titles",
        nargs="+",
        default=["USA", "Barack Obama", "AudienceTrendMinerDefinitelyMissing"],
    )
    parser.add_argument(
        "--country-day",
        type=date.fromisoformat,
        default=date(2025, 7, 15),
    )
    parser.add_argument(
        "--category-limit",
        default="max",
        help="Action API cllimit; use 10 to force a continuation comparison.",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_live_experiment(
                tuple(arguments.titles),
                arguments.country_day,
                category_limit=arguments.category_limit,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
