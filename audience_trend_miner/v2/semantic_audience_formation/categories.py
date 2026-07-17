from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Mapping, Sequence


CATEGORY_RULE_SET_VERSION = "1.0"
CATEGORY_NOISE_PATTERNS = (
    r"^\d{4} births$",
    r"^\d{4} deaths$",
    r"^Living people$",
    r"^Possibly living people$",
    r"^Year of birth missing",
    r"^Year of death missing",
    r"^Articles with",
    r"^All articles",
    r"^CS1",
    r"^Webarchive",
    r"^Wikipedia:",
)
_CATEGORY_NOISE = tuple(re.compile(pattern) for pattern in CATEGORY_NOISE_PATTERNS)


@dataclass(frozen=True)
class SelectedCategoryPage:
    page_id: int
    canonical_title: str
    lead: str
    selected_categories: tuple[str, ...]


@dataclass(frozen=True)
class CategorySelection:
    pages: tuple[SelectedCategoryPage, ...]
    rule_set: Mapping[str, object]
    category_document_frequency: Mapping[str, int]
    category_idf: Mapping[str, float]


def select_categories(pages: Sequence[Mapping[str, object]]) -> CategorySelection:
    """Select meaningful category evidence across the full Canonical Page universe."""
    meaningful_by_page: list[tuple[Mapping[str, object], set[str]]] = []
    document_frequency: dict[str, int] = {}
    for page in pages:
        categories = {
            category
            for category in page["categories"]
            if isinstance(category, str) and not _is_noise(category)
        }
        meaningful_by_page.append((page, categories))
        for category in categories:
            document_frequency[category] = document_frequency.get(category, 0) + 1

    total_pages = len(pages)
    category_idf = {
        category: math.log(total_pages / frequency)
        for category, frequency in document_frequency.items()
    }
    selected_pages = tuple(
        SelectedCategoryPage(
            page_id=int(page["page_id"]),
            canonical_title=str(page["canonical_title"]),
            lead=str(page["lead"]),
            selected_categories=tuple(
                sorted(categories, key=lambda category: (-category_idf[category], category))[
                    :5
                ]
            ),
        )
        for page, categories in sorted(
            meaningful_by_page, key=lambda item: int(item[0]["page_id"])
        )
    )
    rule_set: Mapping[str, object] = MappingProxyType(
        {
            "version": CATEGORY_RULE_SET_VERSION,
            "hidden_categories": "excluded by Wikimedia Evidence provenance",
            "noise_patterns": CATEGORY_NOISE_PATTERNS,
        }
    )
    return CategorySelection(
        pages=selected_pages,
        rule_set=rule_set,
        category_document_frequency=MappingProxyType(document_frequency),
        category_idf=MappingProxyType(category_idf),
    )


def _is_noise(category: str) -> bool:
    return any(pattern.search(category) for pattern in _CATEGORY_NOISE)
