from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import jsonschema

from audience_trend_miner.v2.cluster_adjudication.stage import (
    SCHEMA_PATH as CLUSTER_ADJUDICATION_SCHEMA_PATH,
)
from audience_trend_miner.v2.shared import (
    V2ContractError,
    consume_artifact,
    validate_schema,
)
from audience_trend_miner.v2.wikimedia_evidence.stage import (
    SCHEMA_PATH as WIKIMEDIA_EVIDENCE_SCHEMA_PATH,
)


Direction = Literal[
    "robust_growth",
    "robust_shrinking",
    "sudden_growth",
    "uncertain_direction",
]
Window = Literal["previous", "current"]


@dataclass(frozen=True)
class WindowTraffic:
    """Published traffic and its normalized conservative range for one window."""

    observed_total: int
    observed_page_days: int
    successful_days: int
    conservative_observed_minimum: int
    conservative_observed_maximum: int
    seven_day_equivalent: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ClusterTraffic:
    """Deterministic traffic evidence attached to terminal cluster membership."""

    cluster_id: str
    source_preliminary_cluster_id: str
    name: str
    rationale: str
    member_page_ids: tuple[int, ...]
    previous: WindowTraffic
    current: WindowTraffic
    direction: Direction


@dataclass(frozen=True)
class _TrafficEvidence:
    successful_dates: dict[Window, tuple[str, ...]]
    cutoffs: dict[str, int]
    pages: dict[int, dict[str, int]]


def attach_cluster_traffic(
    *,
    run_id: str,
    wikimedia_evidence_path: Path,
    cluster_adjudication_path: Path,
) -> tuple[ClusterTraffic, ...]:
    """Attach censored Wikimedia traffic after cluster membership is terminal."""
    evidence_artifact = consume_artifact(
        wikimedia_evidence_path,
        run_id=run_id,
        stage="wikimedia-evidence",
    )
    adjudication_artifact = consume_artifact(
        cluster_adjudication_path,
        run_id=run_id,
        stage="cluster-adjudication",
    )
    evidence_payload = _compatible_payload(
        evidence_artifact,
        schema_path=WIKIMEDIA_EVIDENCE_SCHEMA_PATH,
        stage_name="Wikimedia Evidence",
    )
    adjudication_payload = _compatible_payload(
        adjudication_artifact,
        schema_path=CLUSTER_ADJUDICATION_SCHEMA_PATH,
        stage_name="Cluster Adjudication",
    )

    successful_dates = _successful_dates(evidence_payload)
    cutoffs = _daily_cutoffs(evidence_payload, successful_dates)
    pages = _canonical_pages(evidence_payload, successful_dates)
    traffic_evidence = _TrafficEvidence(successful_dates, cutoffs, pages)
    clusters = cast(
        list[dict[str, object]], adjudication_payload["final_audience_clusters"]
    )
    _validate_terminal_membership(clusters, pages)

    attached: list[ClusterTraffic] = []
    for cluster in clusters:
        members = cast(list[dict[str, object]], cluster["members"])
        page_ids = tuple(cast(int, member["page_id"]) for member in members)
        previous = _window_traffic(
            page_ids,
            "previous",
            traffic_evidence,
        )
        current = _window_traffic(
            page_ids,
            "current",
            traffic_evidence,
        )
        attached.append(
            ClusterTraffic(
                cluster_id=cast(str, cluster["cluster_id"]),
                source_preliminary_cluster_id=cast(
                    str, cluster["source_preliminary_cluster_id"]
                ),
                name=cast(str, cluster["name"]),
                rationale=cast(str, cluster["rationale"]),
                member_page_ids=page_ids,
                previous=previous,
                current=current,
                direction=_direction(previous, current),
            )
        )
    return tuple(attached)


def _compatible_payload(
    artifact: dict[str, object],
    *,
    schema_path: Path,
    stage_name: str,
) -> dict[str, object]:
    payload = artifact["payload"]
    try:
        validate_schema(schema_path, payload)
    except jsonschema.ValidationError as error:
        raise V2ContractError(
            f"{stage_name} is schema-incompatible: {error.message}"
        ) from error
    return cast(dict[str, object], payload)


def _successful_dates(
    payload: dict[str, object],
) -> dict[Window, tuple[str, ...]]:
    nominal_days = cast(list[dict[str, object]], payload["nominal_days"])
    dates: dict[Window, list[str]] = {"previous": [], "current": []}
    seen: set[str] = set()
    for day in nominal_days:
        day_text = cast(str, day["date"])
        if day_text in seen:
            raise V2ContractError("Wikimedia Evidence contains duplicate nominal days")
        seen.add(day_text)
        if day["status"] == "successful":
            dates[cast(Window, day["window"])].append(day_text)
    coverage = cast(dict[str, int], payload["coverage"])
    if any(len(dates[window]) != coverage[window] for window in dates):
        raise V2ContractError(
            "Wikimedia Evidence coverage conflicts with successful Analytics days"
        )
    return {window: tuple(values) for window, values in dates.items()}


def _daily_cutoffs(
    payload: dict[str, object],
    successful_dates: dict[Window, tuple[str, ...]],
) -> dict[str, int]:
    expected_dates = {
        day for dates in successful_dates.values() for day in dates
    }
    cutoffs: dict[str, int] = {}
    for item in cast(list[dict[str, object]], payload["daily_cutoffs"]):
        day = cast(str, item["date"])
        if day in cutoffs:
            raise V2ContractError("Wikimedia Evidence contains duplicate daily cutoffs")
        cutoff = item["views_ceil"]
        if cutoff is None:
            raise V2ContractError(
                f"Wikimedia Evidence has no verified cutoff for successful day {day}"
            )
        cutoffs[day] = cast(int, cutoff)
    if set(cutoffs) != expected_dates:
        raise V2ContractError(
            "Wikimedia Evidence cutoff dates conflict with successful Analytics days"
        )
    return cutoffs


def _canonical_pages(
    payload: dict[str, object],
    successful_dates: dict[Window, tuple[str, ...]],
) -> dict[int, dict[str, int]]:
    allowed_dates = {
        day for dates in successful_dates.values() for day in dates
    }
    pages: dict[int, dict[str, int]] = {}
    for page in cast(list[dict[str, object]], payload["canonical_pages"]):
        page_id = cast(int, page["page_id"])
        if page_id in pages:
            raise V2ContractError("Wikimedia Evidence contains duplicate Canonical Pages")
        observations: dict[str, int] = {}
        for item in cast(list[dict[str, object]], page["observations"]):
            day = cast(str, item["date"])
            if day in observations:
                raise V2ContractError(
                    f"Canonical Page {page_id} has duplicate daily observations"
                )
            if day not in allowed_dates:
                raise V2ContractError(
                    f"Canonical Page {page_id} has an observation outside an Effective Window"
                )
            observations[day] = cast(int, item["views_ceil"])
        pages[page_id] = observations
    return pages


def _validate_terminal_membership(
    clusters: list[dict[str, object]],
    pages: dict[int, dict[str, int]],
) -> None:
    accepted_page_ids: list[int] = []
    for cluster in clusters:
        members = cast(list[dict[str, object]], cluster["members"])
        accepted_page_ids.extend(cast(int, member["page_id"]) for member in members)
    if len(accepted_page_ids) != len(set(accepted_page_ids)):
        raise V2ContractError(
            "a Canonical Page may contribute to at most one Final Audience Cluster"
        )
    unknown = sorted(set(accepted_page_ids) - set(pages))
    if unknown:
        raise V2ContractError(
            f"Cluster Adjudication references Canonical Pages absent from Wikimedia Evidence: {unknown}"
        )


def _window_traffic(
    page_ids: tuple[int, ...],
    window: Window,
    evidence: _TrafficEvidence,
) -> WindowTraffic:
    dates = evidence.successful_dates[window]
    observed_total = 0
    observed_page_days = 0
    minimum = 0
    maximum = 0
    for day in dates:
        for page_id in page_ids:
            published = evidence.pages[page_id].get(day)
            if published is None:
                maximum += evidence.cutoffs[day]
                continue
            observed_total += published
            observed_page_days += 1
            minimum += max(0, published - 99)
            maximum += published
    successful_days = len(dates)
    return WindowTraffic(
        observed_total=observed_total,
        observed_page_days=observed_page_days,
        successful_days=successful_days,
        conservative_observed_minimum=minimum,
        conservative_observed_maximum=maximum,
        seven_day_equivalent=observed_total * 7 / successful_days,
        minimum=minimum * 7 / successful_days,
        maximum=maximum * 7 / successful_days,
    )


def _direction(previous: WindowTraffic, current: WindowTraffic) -> Direction:
    if previous.observed_total == 0 and current.observed_total > 0:
        return "sudden_growth"
    if (
        current.conservative_observed_minimum * previous.successful_days
        > previous.conservative_observed_maximum * current.successful_days
    ):
        return "robust_growth"
    if (
        current.conservative_observed_maximum * previous.successful_days
        < previous.conservative_observed_minimum * current.successful_days
    ):
        return "robust_shrinking"
    return "uncertain_direction"
