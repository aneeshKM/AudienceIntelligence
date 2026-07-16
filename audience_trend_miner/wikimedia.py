from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path
import random
from threading import Lock
import time
from typing import Mapping, Protocol
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DEFAULT_WIKIMEDIA_REST_BASE_URL = "https://wikimedia.org/api/rest_v1"
DEFAULT_WIKIPEDIA_ACTION_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "AudienceTrendMiner/0.1 (https://github.com/aneeshKM/AudienceIntelligence)"
MAX_ATTEMPTS = 3


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
    name: str
    payload: object


@dataclass(frozen=True)
class AcquisitionFailure:
    operation: str
    subject: str
    attempts: int
    reason: str


class WikimediaTransientError(RuntimeError):
    def __init__(self, message: str, *, retry_immediately: bool = False) -> None:
        self.retry_immediately = retry_immediately
        self.attempts = 1
        super().__init__(message)


class WikimediaPermanentError(RuntimeError):
    def __init__(self, message: str) -> None:
        self.attempts = 1
        super().__init__(message)


@dataclass(frozen=True)
class AcquisitionSettings:
    max_concurrent_articles: int = 8
    base_backoff_seconds: float = 1.0


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

    def audit_data(self) -> dict[str, object]:
        return {
            "raw_candidate_titles": list(self.raw_candidate_titles),
            "canonical_articles": [
                {
                    "page_id": article.page_id,
                    "canonical_title": article.canonical_title,
                    "extract": article.extract,
                    "categories": list(article.categories),
                    "previous_window_views": article.previous_window_views,
                    "current_window_views": article.current_window_views,
                    "aliases": [
                        {
                            "raw_title": alias.raw_title,
                            "previous_window_views": alias.previous_window_views,
                            "current_window_views": alias.current_window_views,
                            "daily_views": [
                                {"date": item.date.isoformat(), "views": item.views}
                                for item in alias.daily_views
                            ],
                        }
                        for alias in article.aliases
                    ],
                }
                for article in self.canonical_articles
            ],
            "failures": [
                {
                    "operation": failure.operation,
                    "subject": failure.subject,
                    "attempts": failure.attempts,
                    "reason": failure.reason,
                }
                for failure in self.failures
            ],
            "degraded": self.degraded,
        }


@dataclass(frozen=True)
class DiscoveryResponse:
    titles: tuple[str, ...]
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
class _AliasOutcome:
    alias: AliasTraffic | None = None
    pageviews: PageviewsResponse | None = None
    metadata: MetadataResponse | None = None
    failure: AcquisitionFailure | None = None


class WikimediaAdapter(Protocol):
    def daily_top_pages(self, day: date) -> DiscoveryResponse: ...

    def article_pageviews(
        self, raw_title: str, start: date, end: date
    ) -> PageviewsResponse: ...

    def article_metadata(self, raw_title: str) -> MetadataResponse: ...


class JsonTransport(Protocol):
    def get_json(self, url: str) -> object: ...


class UrllibJsonTransport:
    def get_json(self, url: str) -> object:
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 429 or error.code >= 500:
                raise WikimediaTransientError(str(error)) from error
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


def acquire_wikimedia_attention(
    windows: AnalysisWindows,
    adapter: WikimediaAdapter,
    settings: AcquisitionSettings = AcquisitionSettings(),
) -> WikimediaAttentionResult:
    titles: set[str] = set()
    artifacts: list[RawArtifact] = []
    day = windows.current_start
    while day <= windows.current_end:
        try:
            response = _attempt(
                lambda: adapter.daily_top_pages(day), settings
            )
        except (WikimediaTransientError, WikimediaPermanentError) as error:
            raise IncompleteCandidateUniverseError(
                AcquisitionFailure(
                    operation="discovery",
                    subject=day.isoformat(),
                    attempts=error.attempts,
                    reason=str(error),
                )
            ) from error
        titles.update(response.titles)
        artifacts.append(RawArtifact(f"discovery/{day.isoformat()}.json", response.raw))
        day += timedelta(days=1)

    raw_titles = tuple(sorted(titles))
    with ThreadPoolExecutor(max_workers=settings.max_concurrent_articles) as executor:
        acquired = list(
            executor.map(
                lambda title: _acquire_alias(title, windows, adapter, settings),
                raw_titles,
            )
        )

    failures: list[AcquisitionFailure] = []
    grouped: dict[int, list[tuple[AliasTraffic, MetadataResponse]]] = {}
    for outcome in acquired:
        if outcome.pageviews is not None:
            title = outcome.alias.raw_title if outcome.alias else outcome.failure.subject
            encoded_name = title.replace("/", "%2F")
            artifacts.append(
                RawArtifact(f"pageviews/{encoded_name}.json", outcome.pageviews.raw)
            )
        if outcome.metadata is not None:
            title = outcome.alias.raw_title if outcome.alias else outcome.failure.subject
            encoded_name = title.replace("/", "%2F")
            artifacts.append(
                RawArtifact(f"metadata/{encoded_name}.json", outcome.metadata.raw)
            )
        if outcome.failure is not None:
            failures.append(outcome.failure)
            continue
        grouped.setdefault(outcome.metadata.page_id, []).append(
            (outcome.alias, outcome.metadata)
        )

    canonical_articles: list[CanonicalArticle] = []
    for page_id in sorted(grouped):
        aliases_and_metadata = grouped[page_id]
        canonical_titles = {
            metadata.canonical_title for _, metadata in aliases_and_metadata
        }
        if len(canonical_titles) != 1:
            failures.append(
                AcquisitionFailure(
                    operation="canonicalization",
                    subject=str(page_id),
                    attempts=1,
                    reason=(
                        "aliases returned conflicting canonical titles: "
                        + ", ".join(sorted(canonical_titles))
                    ),
                )
            )
            continue
        metadata = aliases_and_metadata[0][1]
        aliases = tuple(item[0] for item in aliases_and_metadata)
        canonical_articles.append(
            CanonicalArticle(
                page_id=page_id,
                canonical_title=metadata.canonical_title,
                extract=metadata.extract,
                categories=metadata.categories,
                previous_window_views=sum(
                    alias.previous_window_views for alias in aliases
                ),
                current_window_views=sum(alias.current_window_views for alias in aliases),
                aliases=aliases,
            )
        )

    return WikimediaAttentionResult(
        raw_candidate_titles=raw_titles,
        canonical_articles=tuple(canonical_articles),
        raw_artifacts=tuple(artifacts),
        failures=tuple(failures),
    )


def _attempt(operation, settings: AcquisitionSettings):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return operation()
        except WikimediaPermanentError as error:
            error.attempts = attempt
            raise
        except WikimediaTransientError as error:
            error.attempts = attempt
            if attempt == MAX_ATTEMPTS:
                raise
            if not error.retry_immediately:
                delay = settings.base_backoff_seconds * (2 ** (attempt - 1))
                time.sleep(delay + random.uniform(0, delay))


def _acquire_alias(
    raw_title: str,
    windows: AnalysisWindows,
    adapter: WikimediaAdapter,
    settings: AcquisitionSettings,
) -> _AliasOutcome:
    expected_dates = tuple(
        windows.previous_start + timedelta(days=offset)
        for offset in range((windows.current_end - windows.previous_start).days + 1)
    )

    def fetch_complete_pageviews() -> PageviewsResponse:
        response = adapter.article_pageviews(
            raw_title, windows.previous_start, windows.current_end
        )
        observed_dates = tuple(item.date for item in response.daily_views)
        if observed_dates != expected_dates:
            raise WikimediaTransientError(
                "Pageviews response must contain complete dated observations "
                f"from {windows.previous_start} through {windows.current_end}"
            )
        return response

    try:
        pageviews = _attempt(fetch_complete_pageviews, settings)
    except (WikimediaTransientError, WikimediaPermanentError) as error:
        return _AliasOutcome(
            failure=AcquisitionFailure(
                operation="pageviews",
                subject=raw_title,
                attempts=error.attempts,
                reason=str(error),
            )
        )
    try:
        metadata = _attempt(
            lambda: adapter.article_metadata(raw_title), settings
        )
    except (WikimediaTransientError, WikimediaPermanentError) as error:
        return _AliasOutcome(
            pageviews=pageviews,
            failure=AcquisitionFailure(
                operation="metadata",
                subject=raw_title,
                attempts=error.attempts,
                reason=str(error),
            ),
        )
    previous_views = sum(
        item.views
        for item in pageviews.daily_views
        if windows.previous_start <= item.date <= windows.previous_end
    )
    current_views = sum(
        item.views
        for item in pageviews.daily_views
        if windows.current_start <= item.date <= windows.current_end
    )
    return _AliasOutcome(
        alias=AliasTraffic(
            raw_title=raw_title,
            previous_window_views=previous_views,
            current_window_views=current_views,
            daily_views=pageviews.daily_views,
        ),
        pageviews=pageviews,
        metadata=metadata,
    )
