from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import jsonschema


FROZEN_EVALUATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "articles", "clusters", "top_audience_editor_reviews"],
    "properties": {
        "schema_version": {"const": "1.0"},
        "articles": {"type": "array", "minItems": 1, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["page_id", "commercially_relevant"],
            "properties": {"page_id": {"type": "integer", "minimum": 1},
                           "commercially_relevant": {"type": "boolean"}},
        }},
        "clusters": {"type": "array", "minItems": 1, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["component_id", "name", "has_unrelated_member", "tragedy", "violent_crime"],
            "properties": {"name": {"type": "string", "minLength": 1},
                           "component_id": {"type": "integer", "minimum": 1},
                           "has_unrelated_member": {"type": "boolean"},
                           "tragedy": {"type": "boolean"},
                           "violent_crime": {"type": "boolean"}},
        }},
        "top_audience_editor_reviews": {"type": "array", "minItems": 5,
            "maxItems": 5, "items": {"type": "object", "additionalProperties": False,
                "required": ["rank", "audience_name", "reviewer", "reviewed_at", "coherent", "name_useful", "brand_useful"],
                "properties": {"rank": {"type": "integer", "minimum": 1, "maximum": 5},
                               "audience_name": {"type": "string", "minLength": 1},
                               "reviewer": {"type": "string", "minLength": 1},
                               "reviewed_at": {"type": "string", "format": "date"},
                               "coherent": {"type": "boolean"},
                               "name_useful": {"type": "boolean"},
                               "brand_useful": {"type": "boolean"}}}},
    },
}


@dataclass(frozen=True)
class FrozenEvaluationResult:
    commercial_relevance: float
    approved_top_five: int
    passed: bool


@dataclass(frozen=True)
class PublicationQualityResult:
    traced_page_ids: tuple[int, ...]
    total_size_basis_points: int


def evaluate_frozen_fixture(
    fixture: object,
    audit: dict[str, object],
    portfolio: dict[str, object],
) -> FrozenEvaluationResult:
    """Compare produced V1 decisions with the immutable editor-labelled set."""
    jsonschema.validate(fixture, FROZEN_EVALUATION_SCHEMA)
    assert isinstance(fixture, dict)
    labels = {item["page_id"]: item for item in fixture["articles"]}
    produced_ids = {item["page_id"] for item in audit["article_classifications"]}
    if produced_ids != set(labels):
        raise ValueError("produced article decisions do not match the frozen fixture")
    accepted_articles = [
        labels[item["page_id"]]
        for item in audit["article_classifications"]
        if item["accepted"]
    ]
    if not accepted_articles:
        raise ValueError("evaluation fixture must contain accepted articles")
    relevance = sum(item["commercially_relevant"] for item in accepted_articles) / len(accepted_articles)
    accepted_components = {
        item["source_component_id"]
        for item in audit["cluster_refinement"]["accepted"]
    }
    cluster_labels = {item["component_id"]: item for item in fixture["clusters"]}
    for component_id in accepted_components:
        if component_id not in cluster_labels:
            raise ValueError(f"accepted component {component_id} has no frozen label")
        cluster = cluster_labels[component_id]
        if cluster["has_unrelated_member"]:
            raise ValueError(f"accepted cluster has an unrelated member: {cluster['name']}")
        if cluster["tragedy"] or cluster["violent_crime"]:
            raise ValueError(f"accepted cluster fails the violent-crime/tragedy gate: {cluster['name']}")
    reviews = sorted(fixture["top_audience_editor_reviews"], key=lambda item: item["rank"])
    produced_top_five = [item["name"] for item in portfolio["audiences"][:5]]
    if produced_top_five != [item["audience_name"] for item in reviews]:
        raise ValueError("editor reviews do not match the produced top-five audiences")
    approvals = sum(
        review["coherent"] and review["name_useful"] and review["brand_useful"]
        for review in reviews
    )
    if relevance < 0.8:
        raise ValueError("commercial relevance is below 80%")
    if approvals < 4:
        raise ValueError("fewer than four top-five audiences have complete editor approval")
    return FrozenEvaluationResult(relevance, approvals, True)


def verify_publication_quality(
    audit: dict[str, object], portfolio: dict[str, object]
) -> PublicationQualityResult:
    """Independently verify final membership lineage and Size Index arithmetic."""
    articles = {item["page_id"]: item for item in audit["canonical_articles"]}
    qualified = {item["page_id"] for item in audit["qualified_signals"]}
    classified = {item["page_id"] for item in audit["article_classifications"] if item["accepted"]}
    components = {
        item["component_id"]: set(item["page_ids"])
        for item in audit["candidate_clustering"]["components"]
    }
    refined = {
        item["source_component_id"]: item
        for item in audit["cluster_refinement"]["accepted"]
    }
    calculations = {
        item["source_component_id"]: item for item in audit["portfolio_calculations"]
    }
    audiences = portfolio["audiences"]
    if audiences:
        _verify_all_consumed_dispositions(audit, articles, classified, components)
    all_page_ids: list[int] = []
    traffic: list[int] = []
    for audience in audiences:
        component_id = audience["source_component_id"]
        page_ids = tuple(audience["page_ids"])
        calculation = calculations.get(component_id)
        accepted = refined.get(component_id)
        if calculation is None or accepted is None or not accepted["safety"]["safe"]:
            raise ValueError(f"incomplete refinement-to-portfolio lineage for component {component_id}")
        if set(page_ids) != set(calculation["page_ids"]) or set(page_ids) != set(accepted["page_ids"]):
            raise ValueError(f"portfolio membership disagrees for component {component_id}")
        if not set(page_ids).issubset(components.get(component_id, set())):
            raise ValueError(f"cluster lineage is incomplete for component {component_id}")
        for page_id in page_ids:
            article = articles.get(page_id)
            if article is None or not article["aliases"]:
                raise ValueError(f"alias lineage is incomplete for page {page_id}")
            if page_id not in qualified or page_id not in classified:
                raise ValueError(f"decision lineage is incomplete for page {page_id}")
        all_page_ids.extend(page_ids)
        traffic.append(sum(articles[page_id]["current_window_views"] for page_id in page_ids))
    if len(all_page_ids) != len(set(all_page_ids)):
        raise ValueError("final portfolio membership is not exclusive")
    expected_points = _independent_basis_points(traffic)
    for audience, points in zip(audiences, expected_points, strict=True):
        calculation = calculations[audience["source_component_id"]]
        if calculation["size_basis_points"] != points or calculation["estimated_size_index"] != points / 100:
            raise ValueError(f"Size Index is incorrect for component {audience['source_component_id']}")
        if audience["estimated_size_index"] != points / 100:
            raise ValueError(f"published Size Index is incorrect for component {audience['source_component_id']}")
    return PublicationQualityResult(tuple(sorted(all_page_ids)), sum(expected_points))


def _verify_all_consumed_dispositions(
    audit: dict[str, object],
    articles: dict[int, dict[str, object]],
    classified: set[int],
    components: dict[int, set[int]],
) -> None:
    raw_titles = set(audit["raw_candidate_titles"])
    alias_titles = {
        alias["raw_title"]
        for article in articles.values()
        for alias in article["aliases"]
    }
    failed_titles = {
        failure["subject"]
        for failure in audit["failures"]
        if failure["operation"] in {"pageviews", "metadata", "canonicalization"}
    }
    if raw_titles != alias_titles | failed_titles:
        raise ValueError("raw candidate disposition lineage is incomplete")
    decision_ids = {item["page_id"] for item in audit["decisions"]}
    if decision_ids != set(articles):
        raise ValueError("canonical qualification lineage is incomplete")
    classification_ids = {item["page_id"] for item in audit["article_classifications"]}
    qualified_for_classification = {
        item["page_id"]
        for item in audit["decisions"]
        if item["outcome"] in {"classified_signal", "classification_rejected"}
    }
    if classification_ids != qualified_for_classification:
        raise ValueError("classification filtering lineage is incomplete")
    clustered_ids = set().union(*components.values()) if components else set()
    if clustered_ids != classified:
        raise ValueError("classification-to-clustering lineage is incomplete")
    candidate_components = {
        item["component_id"]
        for item in audit["candidate_clustering"]["components"]
        if item["is_candidate_cluster"]
    }
    refined_components = {
        item["component_id"] for item in audit["cluster_refinement"]["decisions"]
    }
    if candidate_components != refined_components:
        raise ValueError("cluster refinement disposition lineage is incomplete")
    singleton_ids = {
        next(iter(components[item["component_id"]]))
        for item in audit["candidate_clustering"]["components"]
        if not item["is_candidate_cluster"]
    }
    if singleton_ids != set(audit["cluster_refinement"]["rejected_standalone_page_ids"]):
        raise ValueError("singleton disposition lineage is incomplete")


def _independent_basis_points(traffic: list[int]) -> list[int]:
    if not traffic:
        return []
    total = sum(traffic)
    quotas = [Fraction(value * 10_000, total) for value in traffic]
    points = [quota.numerator // quota.denominator for quota in quotas]
    remainder_order = sorted(
        range(len(quotas)), key=lambda index: (-(quotas[index] - points[index]), index)
    )
    for index in remainder_order[:10_000 - sum(points)]:
        points[index] += 1
    return points
