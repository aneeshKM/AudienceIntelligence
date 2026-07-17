from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import ssl
from threading import Lock
import time
from typing import Mapping, Protocol
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import certifi


DEFAULT_WIKIMEDIA_REST_BASE_URL = "https://wikimedia.org/api/rest_v1"
DEFAULT_WIKIPEDIA_ACTION_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "AudienceTrendMiner/0.1 (https://github.com/aneeshKM/AudienceIntelligence)"
MACOS_CA_BUNDLE = Path("/etc/ssl/cert.pem")
DEFAULT_REQUEST_INTERVAL_SECONDS = 0.5


def trusted_ssl_context() -> ssl.SSLContext:
    ca_bundle = MACOS_CA_BUNDLE if MACOS_CA_BUNDLE.is_file() else Path(certifi.where())
    return ssl.create_default_context(cafile=str(ca_bundle))


@dataclass(frozen=True)
class AnalysisWindows:
    previous_start: date
    previous_end: date
    current_start: date
    current_end: date


@dataclass(frozen=True)
class DailyView:
    date: date
    views: int


@dataclass(frozen=True)
class AliasTraffic:
    raw_title: str
    previous_window_views: int
    current_window_views: int
    daily_views: tuple[DailyView, ...]


@dataclass(frozen=True)
class CanonicalArticle:
    page_id: int
    canonical_title: str
    extract: str
    categories: tuple[str, ...]
    previous_window_views: int
    current_window_views: int
    aliases: tuple[AliasTraffic, ...]


@dataclass(frozen=True)
class RawArtifact:
    operation: str
    subject: str
    payload: object


@dataclass(frozen=True)
class AcquisitionFailure:
    operation: str
    subject: str
    attempts: int
    reason: str


class WikimediaTransientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retry_immediately: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.retry_immediately = retry_immediately
        self.retry_after_seconds = retry_after_seconds
        self.attempts = 1
        super().__init__(message)


class WikimediaPermanentError(RuntimeError):
    def __init__(self, message: str) -> None:
        self.attempts = 1
        super().__init__(message)


class IncompleteCandidateUniverseError(RuntimeError):
    def __init__(self, failure: AcquisitionFailure) -> None:
        self.failure = failure
        super().__init__(
            f"Candidate Universe discovery failed for {failure.subject} "
            f"after {failure.attempts} attempts: {failure.reason}"
        )


@dataclass(frozen=True)
class WikimediaAttentionResult:
    raw_candidate_titles: tuple[str, ...]
    canonical_articles: tuple[CanonicalArticle, ...]
    raw_artifacts: tuple[RawArtifact, ...]
    failures: tuple[AcquisitionFailure, ...] = ()

    @property
    def degraded(self) -> bool:
        return bool(self.failures)

@dataclass(frozen=True)
class DiscoveryResponse:
    titles: tuple[str, ...]
    raw: object


@dataclass(frozen=True)
class CountryPageviewRecord:
    project: str
    article: str
    views_ceil: int


@dataclass(frozen=True)
class CountryTopPagesResponse:
    records: tuple[CountryPageviewRecord, ...]
    raw: object


@dataclass(frozen=True)
class PageviewsResponse:
    daily_views: tuple[DailyView, ...]
    raw: object


@dataclass(frozen=True)
class MetadataResponse:
    page_id: int
    canonical_title: str
    extract: str
    categories: tuple[str, ...]
    raw: object


@dataclass(frozen=True)
class MetadataBatchResponse:
    pages: tuple[MetadataResponse, ...]
    aliases: Mapping[str, int]
    unavailable_titles: tuple[str, ...]


@dataclass(frozen=True)
class AliasEvidence:
    traffic: AliasTraffic
    metadata: MetadataResponse
    artifacts: tuple[RawArtifact, ...]


@dataclass(frozen=True)
class AliasEvidenceFailure:
    failure: AcquisitionFailure
    artifacts: tuple[RawArtifact, ...]


class WikimediaAdapter(Protocol):
    def daily_top_pages(self, day: date) -> DiscoveryResponse: ...

    def article_pageviews(
        self, raw_title: str, start: date, end: date
    ) -> PageviewsResponse: ...

    def article_metadata(self, raw_title: str) -> MetadataResponse: ...


class JsonTransport(Protocol):
    def get_json(self, url: str) -> object: ...


class UrllibJsonTransport:
    def __init__(
        self,
        *,
        ssl_context: ssl.SSLContext | None = None,
        request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
    ) -> None:
        self._ssl_context = ssl_context or trusted_ssl_context()
        self._request_interval_seconds = request_interval_seconds
        self._request_lock = Lock()
        self._next_request_at = 0.0

    def _wait_for_request_slot(self) -> None:
        with self._request_lock:
            now = time.monotonic()
            delay = self._next_request_at - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_request_at = now + self._request_interval_seconds

    def get_json(self, url: str) -> object:
        try:
            self._wait_for_request_slot()
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(
                request, timeout=30, context=self._ssl_context
            ) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 429 or error.code >= 500:
                retry_after = error.headers.get("Retry-After") if error.headers else None
                try:
                    retry_after_seconds = float(retry_after) if retry_after else None
                except ValueError:
                    try:
                        retry_at = parsedate_to_datetime(retry_after)
                        retry_after_seconds = max(
                            0.0,
                            (retry_at - datetime.now(timezone.utc)).total_seconds(),
                        )
                    except (TypeError, ValueError):
                        retry_after_seconds = None
                raise WikimediaTransientError(
                    str(error), retry_after_seconds=retry_after_seconds
                ) from error
            raise WikimediaPermanentError(str(error)) from error
        except Exception as error:
            raise WikimediaTransientError(str(error)) from error


class HttpWikimediaAdapter:
    def __init__(
        self,
        *,
        transport: JsonTransport | None = None,
        rest_base_url: str = DEFAULT_WIKIMEDIA_REST_BASE_URL,
        action_api_url: str = DEFAULT_WIKIPEDIA_ACTION_API_URL,
    ) -> None:
        self._transport = transport or UrllibJsonTransport()
        self._rest_base_url = rest_base_url.rstrip("/")
        self._action_api_url = action_api_url

    def daily_top_pages(self, day: date) -> DiscoveryResponse:
        url = (
            f"{self._rest_base_url}/metrics/pageviews/top/"
            f"en.wikipedia/all-access/{day:%Y/%m/%d}"
        )
        raw = self._transport.get_json(url)
        try:
            titles = tuple(
                article["article"]
                for item in raw["items"]
                for article in item["articles"]
            )
        except (KeyError, TypeError) as error:
            raise WikimediaTransientError("invalid daily top-pages response") from error
        return DiscoveryResponse(titles=titles, raw=raw)

    def daily_country_top_pages(self, day: date) -> CountryTopPagesResponse:
        url = (
            f"{self._rest_base_url}/metrics/pageviews/top-per-country/"
            f"US/all-access/{day:%Y/%m/%d}"
        )
        raw = self._transport.get_json(url)
        try:
            records = tuple(
                CountryPageviewRecord(
                    project=str(article["project"]),
                    article=str(article["article"]),
                    views_ceil=int(article["views_ceil"]),
                )
                for item in raw["items"]
                for article in item["articles"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WikimediaTransientError(
                "invalid country top-pages response"
            ) from error
        return CountryTopPagesResponse(records=records, raw=raw)

    def article_pageviews(
        self, raw_title: str, start: date, end: date
    ) -> PageviewsResponse:
        encoded_title = quote(raw_title, safe="")
        url = (
            f"{self._rest_base_url}/metrics/pageviews/per-article/"
            "en.wikipedia/all-access/user/"
            f"{encoded_title}/daily/{start:%Y%m%d}00/{end:%Y%m%d}00"
        )
        raw = self._transport.get_json(url)
        try:
            daily_views = tuple(
                DailyView(
                    date=date.fromisoformat(
                        f"{item['timestamp'][0:4]}-{item['timestamp'][4:6]}-{item['timestamp'][6:8]}"
                    ),
                    views=int(item["views"]),
                )
                for item in raw["items"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WikimediaTransientError("invalid Pageviews response") from error
        return PageviewsResponse(daily_views=daily_views, raw=raw)

    def article_metadata(self, raw_title: str) -> MetadataResponse:
        url = f"{self._action_api_url}?{urlencode({
            'action': 'query',
            'format': 'json',
            'formatversion': 2,
            'redirects': 1,
            'prop': 'extracts|categories',
            'exintro': 1,
            'explaintext': 1,
            'titles': raw_title,
        })}"
        raw = self._transport.get_json(url)
        try:
            page = raw["query"]["pages"][0]
            return MetadataResponse(
                page_id=int(page["pageid"]),
                canonical_title=str(page["title"]),
                extract=str(page.get("extract", "")),
                categories=tuple(
                    str(category["title"]).removeprefix("Category:")
                    for category in page.get("categories", [])
                ),
                raw=raw,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WikimediaTransientError("invalid metadata response") from error

    def metadata_batch(self, titles: tuple[str, ...]) -> MetadataBatchResponse:
        if not titles or len(titles) > 50:
            raise ValueError("metadata batches require between 1 and 50 titles")
        parameters: dict[str, object] = {
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
            "cllimit": "max",
            "titles": "|".join(titles),
        }
        pages: dict[int, dict[str, object]] = {}
        redirects: dict[str, str] = {}
        normalized: dict[str, str] = {}
        missing: set[str] = set()
        while True:
            raw = self._transport.get_json(
                f"{self._action_api_url}?{urlencode(parameters)}"
            )
            try:
                query = raw["query"]
                for item in query.get("normalized", []):
                    normalized[str(item["from"])] = str(item["to"])
                for item in query.get("redirects", []):
                    redirects[str(item["from"])] = str(item["to"])
                for item in query["pages"]:
                    if item.get("missing") is True:
                        missing.add(str(item["title"]))
                        continue
                    page_id = int(item["pageid"])
                    accumulated = pages.setdefault(
                        page_id,
                        {
                            "title": str(item["title"]),
                            "extract": "",
                            "categories": set(),
                        },
                    )
                    if "extract" in item:
                        accumulated["extract"] = str(item["extract"])[:600]
                    accumulated["categories"].update(
                        str(category["title"]).removeprefix("Category:")
                        for category in item.get("categories", [])
                    )
                continuation = raw.get("continue")
            except (KeyError, TypeError, ValueError) as error:
                raise WikimediaTransientError("invalid metadata batch response") from error
            if not isinstance(continuation, dict):
                break
            parameters.update(continuation)

        page_ids_by_title = {
            str(page["title"]): page_id for page_id, page in pages.items()
        }
        aliases: dict[str, int] = {}
        for requested in titles:
            resolved = normalized.get(requested, requested)
            resolved = redirects.get(resolved, resolved)
            if resolved in page_ids_by_title:
                aliases[requested] = page_ids_by_title[resolved]
        return MetadataBatchResponse(
            pages=tuple(
                MetadataResponse(
                    page_id,
                    str(page["title"]),
                    str(page["extract"]),
                    tuple(sorted(page["categories"])),
                    {},
                )
                for page_id, page in sorted(pages.items())
            ),
            aliases=aliases,
            unavailable_titles=tuple(
                sorted(title for title in titles if title not in aliases or title in missing)
            ),
        )


class FixtureWikimediaAdapter:
    def __init__(
        self,
        *,
        discovery: Mapping[str, list[str]],
        pageviews: Mapping[str, list[dict[str, object]]],
        metadata: Mapping[str, dict[str, object]],
        transient_failures: Mapping[str, int] | None = None,
    ) -> None:
        self._discovery = discovery
        self._pageviews = pageviews
        self._metadata = metadata
        self._transient_failures = dict(transient_failures or {})
        self._failure_lock = Lock()

    @classmethod
    def from_file(cls, path: Path) -> FixtureWikimediaAdapter:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            discovery=payload["discovery"],
            pageviews=payload["pageviews"],
            metadata=payload["metadata"],
            transient_failures=payload.get("transient_failures"),
        )

    def daily_top_pages(self, day: date) -> DiscoveryResponse:
        self._fail_if_scripted(f"discovery:{day.isoformat()}")
        titles = tuple(self._discovery[day.isoformat()])
        return DiscoveryResponse(titles=titles, raw={"titles": list(titles)})

    def article_pageviews(
        self, raw_title: str, start: date, end: date
    ) -> PageviewsResponse:
        self._fail_if_scripted(f"pageviews:{raw_title}")
        raw = self._pageviews[raw_title]
        return PageviewsResponse(
            daily_views=tuple(
                DailyView(date.fromisoformat(str(item["date"])), int(item["views"]))
                for item in raw
            ),
            raw={"daily_views": raw},
        )

    def article_metadata(self, raw_title: str) -> MetadataResponse:
        self._fail_if_scripted(f"metadata:{raw_title}")
        raw = self._metadata[raw_title]
        return MetadataResponse(
            page_id=int(raw["page_id"]),
            canonical_title=str(raw["canonical_title"]),
            extract=str(raw["extract"]),
            categories=tuple(str(category) for category in raw["categories"]),
            raw=raw,
        )

    def _fail_if_scripted(self, key: str) -> None:
        with self._failure_lock:
            remaining = self._transient_failures.get(key, 0)
            if remaining:
                self._transient_failures[key] = remaining - 1
                raise WikimediaTransientError(
                    f"scripted transient failure: {key}",
                    retry_immediately=True,
                )
