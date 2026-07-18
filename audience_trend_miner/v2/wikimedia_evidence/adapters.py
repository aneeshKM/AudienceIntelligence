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
from typing import Any, Mapping, Protocol, cast
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi


DEFAULT_WIKIMEDIA_REST_BASE_URL = "https://wikimedia.org/api/rest_v1"
DEFAULT_WIKIPEDIA_ACTION_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "AudienceTrendMiner/0.1 (https://github.com/aneeshKM/AudienceIntelligence)"
MACOS_CA_BUNDLE = Path("/etc/ssl/cert.pem")
DEFAULT_REQUEST_INTERVAL_SECONDS = 0.5


# Build an SSL context from the macOS or certifi CA bundle.
def trusted_ssl_context() -> ssl.SSLContext:
    ca_bundle = MACOS_CA_BUNDLE if MACOS_CA_BUNDLE.is_file() else Path(certifi.where())
    return ssl.create_default_context(cafile=str(ca_bundle))


# Mark a Wikimedia failure as safe to retry after a delay.
class WikimediaTransientError(RuntimeError):
    # Store retry guidance on a recoverable Wikimedia failure.
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


# Mark a Wikimedia failure as deterministic and non-retryable.
class WikimediaPermanentError(RuntimeError):
    # Mark a Wikimedia failure that should not be retried.
    def __init__(self, message: str) -> None:
        self.attempts = 1
        super().__init__(message)


# Represent one ranked page from the country top-pages API.
@dataclass(frozen=True)
class CountryPageviewRecord:
    project: str
    article: str
    views_ceil: int


# Carry one day of ranked country traffic and its cutoff.
@dataclass(frozen=True)
class CountryTopPagesResponse:
    records: tuple[CountryPageviewRecord, ...]
    raw: object


# Carry canonical identity and semantic metadata for one page.
@dataclass(frozen=True)
class MetadataResponse:
    page_id: int
    canonical_title: str
    extract: str
    categories: tuple[str, ...]
    raw: object


# Carry resolved metadata plus aliases and unavailable titles.
@dataclass(frozen=True)
class MetadataBatchResponse:
    pages: tuple[MetadataResponse, ...]
    aliases: Mapping[str, int]
    unavailable_titles: tuple[str, ...]


# Define the rate-limited JSON transport boundary.
class JsonTransport(Protocol):
    # Fetch a URL and decode its JSON response.
    def get_json(self, url: str) -> object: ...


# Provide paced HTTPS JSON requests with classified failures.
class UrllibJsonTransport:
    # Configure HTTPS and request pacing for Wikimedia calls.
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

    # Block until the shared request interval permits another call.
    def _wait_for_request_slot(self) -> None:
        with self._request_lock:
            now = time.monotonic()
            delay = self._next_request_at - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_request_at = now + self._request_interval_seconds

    # Fetch JSON and classify HTTP failures as transient or permanent.
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
                        retry_at = parsedate_to_datetime(cast(str, retry_after))
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


# Translate Wikimedia REST and Action API responses into domain records.
class HttpWikimediaAdapter:
    # Configure the Wikimedia transport and API endpoint URLs.
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

    # Fetch and parse the United States' top Wikipedia pages for one day.
    def daily_country_top_pages(self, day: date) -> CountryTopPagesResponse:
        url = (
            f"{self._rest_base_url}/metrics/pageviews/top-per-country/"
            f"US/all-access/{day:%Y/%m/%d}"
        )
        raw = self._transport.get_json(url)
        try:
            document = cast(dict[str, Any], raw)
            records = tuple(
                CountryPageviewRecord(
                    project=str(article["project"]),
                    article=str(article["article"]),
                    views_ceil=int(article["views_ceil"]),
                )
                for item in document["items"]
                for article in item["articles"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WikimediaTransientError(
                "invalid country top-pages response"
            ) from error
        return CountryTopPagesResponse(records=records, raw=raw)

    # Fetch metadata for up to 50 titles while resolving aliases and pagination.
    def metadata_batch(self, titles: tuple[str, ...]) -> MetadataBatchResponse:
        # Wikimedia's Action API limits title batches, while category pagination may
        # require several requests for that same batch.
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
        pages: dict[int, dict[str, Any]] = {}
        redirects: dict[str, str] = {}
        normalized: dict[str, str] = {}
        missing: set[str] = set()
        # Merge every continuation page into page-ID keyed metadata accumulators.
        while True:
            raw = cast(
                dict[str, Any],
                self._transport.get_json(
                    f"{self._action_api_url}?{urlencode(parameters)}"
                ),
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
            # Absence of a continuation object marks the complete logical response.
            if not isinstance(continuation, dict):
                break
            parameters.update(continuation)

        page_ids_by_title = {
            str(page["title"]): page_id for page_id, page in pages.items()
        }
        # Replay normalization and redirect chains from each requested spelling to a
        # stable page ID, with cycle protection for malformed redirect data.
        aliases: dict[str, int] = {}
        for requested in titles:
            resolved = normalized.get(requested, requested)
            visited: set[str] = set()
            while resolved in redirects and resolved not in visited:
                visited.add(resolved)
                resolved = redirects[resolved]
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
