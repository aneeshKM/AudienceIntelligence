from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Protocol, cast

import jsonschema

from audience_trend_miner.v2.wikimedia_evidence.jobs import (
    CompletedEvidence,
    EvidenceJob,
    EvidenceJobExecution,
    EvidenceJobStore,
    FailedEvidence,
    TerminalEvidence,
    COUNTRY_DAY_OPERATION,
    METADATA_BATCH_OPERATION,
)
from audience_trend_miner.v2.wikimedia_evidence.adapters import (
    CountryTopPagesResponse,
    HttpWikimediaAdapter,
    WikimediaPermanentError,
)
from audience_trend_miner.v2.shared import (
    ARTIFACT_SCHEMA_VERSION,
    BoundedProgress,
    ProgressEvent,
    ProgressSink,
    V2ContractError,
    atomic_write_json,
    canonical_json_fingerprint,
    consume_artifact,
    record_run_configuration,
    validate_artifact,
    validate_identifier,
    validate_schema,
)


STAGE = "wikimedia-evidence"
MINIMUM_SUCCESSFUL_DAYS = 4
SCHEMA_PATH = Path(__file__).with_name("schemas") / "wikimedia-evidence.schema.json"
TECHNICAL_NAMESPACES = frozenset(
    {
        "category",
        "draft",
        "file",
        "help",
        "media",
        "mediawiki",
        "module",
        "portal",
        "special",
        "template",
        "timedtext",
        "user",
        "user talk",
        "wikipedia",
    }
)


def deterministic_exclusion_reason(title: str) -> str | None:
    normalized = title.replace("_", " ").strip()
    if normalized.casefold() == "main page":
        return "main_page"
    namespace, separator, _ = normalized.partition(":")
    if separator and namespace.casefold() in TECHNICAL_NAMESPACES:
        return f"technical_namespace:{namespace.casefold()}"
    return None


class CountryTopPagesAdapter(Protocol):
    def daily_country_top_pages(self, day: date) -> CountryTopPagesResponse: ...


def consume_wikimedia_evidence(path: Path, *, run_id: str) -> dict[str, object]:
    """Load completed, schema-compatible Wikimedia Evidence for a downstream stage."""
    artifact = consume_artifact(path, run_id=run_id, stage=STAGE)
    try:
        validate_schema(SCHEMA_PATH, artifact["payload"])
    except jsonschema.ValidationError as error:
        raise V2ContractError(
            f"Wikimedia Evidence is schema-incompatible: {error.message}"
        ) from error
    return artifact


def acquire_country_days(
    run_id: str,
    days: tuple[date, ...],
    adapter: CountryTopPagesAdapter,
    store: EvidenceJobStore,
    progress_sink: ProgressSink,
    *,
    workers: int = 4,
) -> tuple[TerminalEvidence, ...]:
    """Acquire independently resumable US country-day Analytics evidence."""
    validate_identifier(run_id, "run_id")
    if len(days) != 14 or any(
        later != earlier + timedelta(days=1)
        for earlier, later in zip(days, days[1:])
    ):
        raise V2ContractError("country acquisition requires fourteen consecutive days")
    subjects = tuple(day.isoformat() for day in days)
    try:
        store.ensure_run(
            run_id,
            {
                "country": "US",
                "access": "all-access",
                "days": ",".join(subjects),
            },
        )
    except ValueError as error:
        raise V2ContractError(str(error)) from error
    store.schedule_country_days(run_id, subjects)
    existing = store.terminal_results(run_id, (COUNTRY_DAY_OPERATION,))
    sequence = 0
    sequence_lock = Lock()

    def emit(subject: str, message: str) -> None:
        nonlocal sequence
        with sequence_lock:
            sequence += 1
            current = sequence
            progress_sink(
                ProgressEvent(
                    run_id=run_id,
                    sequence=current,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    module=STAGE,
                    operation="acquire",
                    level="info",
                    message=f"{message} {subject}",
                    progress=BoundedProgress(current, len(subjects)),
                )
            )

    for result in existing:
        emit(result.subject, "resumed")

    def fetch(job: EvidenceJob) -> object:
        response = adapter.daily_country_top_pages(date.fromisoformat(job.subject))
        return {
            "records": [
                {
                    "project": record.project,
                    "article": record.article,
                    "views_ceil": record.views_ceil,
                }
                for record in response.records
                if record.project == "en.wikipedia"
            ],
            "daily_cutoff_views_ceil": min(
                (record.views_ceil for record in response.records), default=None
            ),
            "non_en_wikipedia_records": sum(
                record.project != "en.wikipedia" for record in response.records
            ),
        }

    EvidenceJobExecution(store, honor_retry_after=True).drain(
        run_id,
        (COUNTRY_DAY_OPERATION,),
        fetch,
        workers=workers,
        is_terminal_error=lambda error: isinstance(error, WikimediaPermanentError),
        on_terminal=lambda job: emit(job.subject, "processed"),
    )
    results = store.results_at_barrier(run_id, (COUNTRY_DAY_OPERATION,))
    for window, window_results in (
        ("previous", results[:7]),
        ("current", results[7:]),
    ):
        successful_days = sum(
            not isinstance(result, FailedEvidence) for result in window_results
        )
        if successful_days < MINIMUM_SUCCESSFUL_DAYS:
            raise V2ContractError(
                f"{window} Effective Window has {successful_days} successful days; "
                f"at least {MINIMUM_SUCCESSFUL_DAYS} are required"
            )
    return results


def execute_wikimedia_evidence(
    *,
    run_id: str,
    as_of_date: date,
    output_root: Path,
    adapter: HttpWikimediaAdapter,
    store: EvidenceJobStore,
    progress_sink: ProgressSink,
    workers: int = 4,
) -> Path:
    """Acquire, resolve, validate, and atomically publish production Run Evidence."""
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    record_run_configuration(
        run_directory,
        run_id,
        {"as_of": as_of_date.isoformat(), "wikimedia_mode": "production"},
    )
    artifact_path = run_directory / f"{STAGE}.json"
    if artifact_path.exists():
        return _resume_completed_evidence(
            artifact_path,
            run_id=run_id,
            as_of_date=as_of_date,
            progress_sink=progress_sink,
        )
    previous_start = as_of_date - timedelta(days=15)
    previous_end = as_of_date - timedelta(days=9)
    current_start = as_of_date - timedelta(days=8)
    current_end = as_of_date - timedelta(days=2)
    days = tuple(previous_start + timedelta(days=offset) for offset in range(14))
    country_results = acquire_country_days(
        run_id, days, adapter, store, progress_sink, workers=workers
    )
    candidate_titles = sorted(
        {
            str(record["article"])
            for result in country_results
            if isinstance(result, CompletedEvidence)
            for record in result.evidence["records"]
        }
    )
    batches = tuple(
        json.dumps(candidate_titles[offset : offset + 50], separators=(",", ":"))
        for offset in range(0, len(candidate_titles), 50)
    )
    store.schedule_metadata_batches(run_id, batches)

    def fetch_metadata(job: EvidenceJob) -> object:
        response = adapter.metadata_batch(tuple(json.loads(job.subject)))
        return {
            "pages": [
                {
                    "page_id": page.page_id,
                    "canonical_title": page.canonical_title,
                    "lead": page.extract,
                    "categories": list(page.categories),
                }
                for page in response.pages
            ],
            "aliases": response.aliases,
            "unavailable_titles": list(response.unavailable_titles),
        }

    EvidenceJobExecution(store, honor_retry_after=True).drain(
        run_id,
        (METADATA_BATCH_OPERATION,),
        fetch_metadata,
        workers=workers,
        is_terminal_error=lambda error: isinstance(error, WikimediaPermanentError),
    )
    metadata_results = store.results_at_barrier(run_id, (METADATA_BATCH_OPERATION,))
    total_progress = 15 + len(batches)
    for offset, result in enumerate(metadata_results, start=15):
        progress_sink(
            ProgressEvent(
                run_id=run_id,
                sequence=offset,
                timestamp=datetime.now(timezone.utc).isoformat(),
                module=STAGE,
                operation="resolve",
                level="info",
                message=f"resolved metadata batch {offset - 14}",
                progress=BoundedProgress(offset, total_progress),
            )
        )
    canonical_metadata_by_alias: dict[str, dict[str, object]] = {}
    canonical_metadata_by_page_id: dict[int, dict[str, object]] = {}
    unavailable_titles: set[str] = set()
    for result in metadata_results:
        if isinstance(result, FailedEvidence):
            unavailable_titles.update(json.loads(result.subject))
            continue
        for page in result.evidence["pages"]:
            canonical_metadata_by_page_id[int(page["page_id"])] = page
        unavailable_titles.update(result.evidence["unavailable_titles"])
        for alias, page_id in result.evidence["aliases"].items():
            canonical_metadata_by_alias[alias] = canonical_metadata_by_page_id[
                int(page_id)
            ]
    for title in unavailable_titles:
        canonical_metadata_by_alias[title] = {"unavailable": True}

    observations_by_alias: dict[str, list[dict[str, object]]] = {}
    nominal_days: list[dict[str, object]] = []
    daily_cutoffs: list[dict[str, object]] = []
    unavailable_days: list[str] = []
    coverage = {"previous": 0, "current": 0}
    non_en_wikipedia_records = 0
    for day, result in zip(days, country_results):
        day_text = day.isoformat()
        window = "previous" if day <= previous_end else "current"
        if isinstance(result, FailedEvidence):
            nominal_days.append({"date": day_text, "window": window, "status": "unavailable"})
            unavailable_days.append(day_text)
            continue
        coverage[window] += 1
        nominal_days.append({"date": day_text, "window": window, "status": "successful"})
        daily_cutoffs.append({"date": day_text, "views_ceil": result.evidence["daily_cutoff_views_ceil"]})
        non_en_wikipedia_records += int(result.evidence["non_en_wikipedia_records"])
        for record in result.evidence["records"]:
            observations_by_alias.setdefault(str(record["article"]), []).append(
                {"date": day_text, "views_ceil": int(record["views_ceil"])}
            )
    canonical_pages, metadata_exclusions = _canonicalize(
        set(candidate_titles), observations_by_alias, canonical_metadata_by_alias
    )
    payload = {
        "as_of_date": as_of_date.isoformat(),
        "nominal_windows": {
            "previous": {"start": previous_start.isoformat(), "end": previous_end.isoformat()},
            "current": {"start": current_start.isoformat(), "end": current_end.isoformat()},
        },
        "nominal_days": nominal_days,
        "coverage": coverage,
        "candidate_universe": candidate_titles,
        "canonical_pages": canonical_pages,
        "daily_cutoffs": daily_cutoffs,
        "provenance": {"source": "wikimedia:top-per-country/US+action-query", "country": "US", "project": "en.wikipedia", "traffic_measure": "views_ceil", "category_visibility": "non-hidden"},
        "exclusions": {"non_en_wikipedia_records": non_en_wikipedia_records, "unavailable_days": unavailable_days, **metadata_exclusions},
        "completion": {"status": "complete", "minimum_successful_days_per_window": MINIMUM_SUCCESSFUL_DAYS},
    }
    artifact = {"schema_version": ARTIFACT_SCHEMA_VERSION, "run_id": run_id, "stage": STAGE, "status": "complete", "payload": payload}
    validate_schema(SCHEMA_PATH, payload)
    validate_artifact(artifact, run_id=run_id, stage=STAGE)
    store.reserve_publication_path(run_id, str(artifact_path))
    atomic_write_json(artifact_path, artifact)
    store.mark_publication_complete(run_id, str(artifact_path))
    progress_sink(
        ProgressEvent(
            run_id=run_id,
            sequence=total_progress,
            timestamp=datetime.now(timezone.utc).isoformat(),
            module=STAGE,
            operation="publish",
            level="info",
            message="published complete Wikimedia Evidence",
            progress=BoundedProgress(total_progress, total_progress),
        )
    )
    return artifact_path


def execute_wikimedia_evidence_fixture(
    *,
    run_id: str,
    as_of_date: date,
    output_root: Path,
    fixture_path: Path,
    progress_sink: ProgressSink,
) -> Path:
    validate_identifier(run_id, "run_id")
    fixture = _load_fixture(fixture_path)
    previous_start = as_of_date - timedelta(days=15)
    previous_end = as_of_date - timedelta(days=9)
    current_start = as_of_date - timedelta(days=8)
    current_end = as_of_date - timedelta(days=2)
    nominal_dates = tuple(
        previous_start + timedelta(days=offset) for offset in range(14)
    )

    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    record_run_configuration(
        run_directory,
        run_id,
        {
            "as_of": as_of_date.isoformat(),
            "wikimedia_fixture_fingerprint": canonical_json_fingerprint(fixture),
            "wikimedia_mode": "fixture",
        },
    )
    artifact_path = run_directory / f"{STAGE}.json"
    if artifact_path.exists():
        return _resume_completed_evidence(
            artifact_path,
            run_id=run_id,
            as_of_date=as_of_date,
            progress_sink=progress_sink,
        )

    daily_responses = cast(dict[str, object], fixture["daily_responses"])
    nominal_days: list[dict[str, object]] = []
    daily_cutoffs: list[dict[str, object]] = []
    candidate_titles: set[str] = set()
    observations_by_alias: dict[str, list[dict[str, object]]] = {}
    unavailable_days: list[str] = []
    coverage = {"previous": 0, "current": 0}
    non_en_wikipedia_records = 0

    for sequence, day in enumerate(nominal_dates, start=1):
        day_text = day.isoformat()
        response = daily_responses.get(day_text)
        window = "previous" if day <= previous_end else "current"
        if not isinstance(response, dict) or "error" in response:
            nominal_days.append(
                {"date": day_text, "window": window, "status": "unavailable"}
            )
            unavailable_days.append(day_text)
        else:
            records = response.get("records")
            if not isinstance(records, list):
                raise V2ContractError(f"fixture records are invalid for {day_text}")
            coverage[window] += 1
            nominal_days.append(
                {"date": day_text, "window": window, "status": "successful"}
            )
            en_records = [
                record
                for record in records
                if isinstance(record, dict) and record.get("project") == "en.wikipedia"
            ]
            non_en_wikipedia_records += len(records) - len(en_records)
            cutoff = min(
                (
                    int(record["views_ceil"])
                    for record in records
                    if isinstance(record, dict)
                ),
                default=None,
            )
            daily_cutoffs.append({"date": day_text, "views_ceil": cutoff})
            for record in en_records:
                title = str(record["article"])
                candidate_titles.add(title)
                observations_by_alias.setdefault(title, []).append(
                    {"date": day_text, "views_ceil": int(record["views_ceil"])}
                )
        progress_sink(
            ProgressEvent(
                run_id=run_id,
                sequence=sequence,
                timestamp=f"{day_text}T00:00:00+00:00",
                module=STAGE,
                operation="acquire",
                level="info",
                message=f"processed {day_text}",
                progress=BoundedProgress(sequence, 15),
            )
        )

    for window, successful_days in coverage.items():
        if successful_days < MINIMUM_SUCCESSFUL_DAYS:
            raise V2ContractError(
                f"{window} Effective Window has {successful_days} successful days; "
                f"at least {MINIMUM_SUCCESSFUL_DAYS} are required"
            )

    canonical_pages, metadata_exclusions = _canonicalize(
        candidate_titles, observations_by_alias, fixture["canonical_pages"]
    )
    payload = {
        "as_of_date": as_of_date.isoformat(),
        "nominal_windows": {
            "previous": {
                "start": previous_start.isoformat(),
                "end": previous_end.isoformat(),
            },
            "current": {
                "start": current_start.isoformat(),
                "end": current_end.isoformat(),
            },
        },
        "nominal_days": nominal_days,
        "coverage": coverage,
        "candidate_universe": sorted(candidate_titles),
        "canonical_pages": canonical_pages,
        "daily_cutoffs": daily_cutoffs,
        "provenance": {
            "source": fixture["source"],
            "country": "US",
            "project": "en.wikipedia",
            "traffic_measure": "views_ceil",
            "category_visibility": "non-hidden",
        },
        "exclusions": {
            "non_en_wikipedia_records": non_en_wikipedia_records,
            "unavailable_days": unavailable_days,
            **metadata_exclusions,
        },
        "completion": {
            "status": "complete",
            "minimum_successful_days_per_window": MINIMUM_SUCCESSFUL_DAYS,
        },
    }
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": "complete",
        "payload": payload,
    }
    validate_schema(SCHEMA_PATH, payload)
    validate_artifact(artifact, run_id=run_id, stage=STAGE)
    atomic_write_json(artifact_path, artifact)
    progress_sink(
        ProgressEvent(
            run_id=run_id,
            sequence=15,
            timestamp=f"{as_of_date.isoformat()}T00:00:00+00:00",
            module=STAGE,
            operation="publish",
            level="info",
            message="published complete Wikimedia Evidence",
            progress=BoundedProgress(15, 15),
        )
    )
    return artifact_path


def _resume_completed_evidence(
    artifact_path: Path,
    *,
    run_id: str,
    as_of_date: date,
    progress_sink: ProgressSink,
) -> Path:
    artifact = consume_wikimedia_evidence(artifact_path, run_id=run_id)
    payload = artifact["payload"]
    assert isinstance(payload, dict)
    if payload["as_of_date"] != as_of_date.isoformat():
        raise V2ContractError(
            "completed Wikimedia Evidence conflicts with requested configuration"
        )
    progress_sink(
        ProgressEvent(
            run_id=run_id,
            sequence=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            module=STAGE,
            operation="resume",
            level="info",
            message="resumed compatible completed Wikimedia Evidence",
            progress=BoundedProgress(1, 1),
        )
    )
    return artifact_path


def _canonicalize(
    candidate_titles: set[str],
    observations_by_alias: Mapping[str, list[dict[str, object]]],
    canonical_facts: object,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not isinstance(canonical_facts, dict):
        raise V2ContractError("fixture canonical_pages are invalid")
    grouped: dict[int, dict[str, Any]] = {}
    unavailable = 0
    main_page = 0
    internal_namespaces: dict[str, int] = {}
    for alias in sorted(candidate_titles):
        facts = canonical_facts.get(alias)
        if not isinstance(facts, dict):
            raise V2ContractError(f"fixture has no canonical identity for {alias}")
        if facts.get("unavailable") is True:
            unavailable += 1
            continue
        canonical_title = str(facts["canonical_title"])
        exclusion = deterministic_exclusion_reason(canonical_title)
        if exclusion == "main_page":
            main_page += 1
            continue
        if exclusion and exclusion.startswith("technical_namespace:"):
            namespace = exclusion.removeprefix("technical_namespace:")
            internal_namespaces[namespace] = internal_namespaces.get(namespace, 0) + 1
            continue
        page_id = int(facts["page_id"])
        lead = str(facts.get("lead", ""))[:600]
        categories = sorted({str(item) for item in facts.get("categories", [])})
        page = grouped.setdefault(
            page_id,
            {
                "page_id": page_id,
                "canonical_title": canonical_title,
                "lead": lead,
                "categories": categories,
                "aliases": [],
                "observations": [],
            },
        )
        if page["canonical_title"] != canonical_title:
            raise V2ContractError(f"canonical title conflict for page {page_id}")
        if page["lead"] != lead or page["categories"] != categories:
            raise V2ContractError(f"canonical metadata conflict for page {page_id}")
        page["aliases"].append(alias)
        page["observations"].extend(observations_by_alias.get(alias, []))
    for page in grouped.values():
        observations_by_date: dict[str, int] = {}
        for observation in page["observations"]:
            observation_date = str(observation["date"])
            observations_by_date[observation_date] = (
                observations_by_date.get(observation_date, 0)
                + int(observation["views_ceil"])
            )
        page["observations"] = [
            {
                "date": observation_date,
                "views_ceil": observations_by_date[observation_date],
            }
            for observation_date in sorted(observations_by_date)
        ]
    return (
        [grouped[page_id] for page_id in sorted(grouped)],
        {
            "metadata_pages_unavailable": unavailable,
            "main_page": main_page,
            "internal_namespaces": dict(sorted(internal_namespaces.items())),
        },
    )


def _load_fixture(path: Path) -> dict[str, object]:
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V2ContractError("Wikimedia Evidence fixture is unreadable") from error
    if (
        not isinstance(fixture, dict)
        or fixture.get("schema_version") != "1.0"
        or not isinstance(fixture.get("source"), str)
        or not isinstance(fixture.get("daily_responses"), dict)
    ):
        raise V2ContractError("Wikimedia Evidence fixture has an invalid shape")
    return fixture
