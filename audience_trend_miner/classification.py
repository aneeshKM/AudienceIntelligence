from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time
from typing import Callable, Protocol
from urllib import request

import jsonschema

from audience_trend_miner.configuration import DEFAULT_MODEL
from audience_trend_miner.wikimedia import CanonicalArticle


REJECTION_CLASSES = (
    "accepted",
    "tragedy",
    "violent_crime",
    "death_driven",
    "routine_politics",
    "isolated_news",
    "no_consumer_audience",
)
ARTICLE_JUDGMENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "supports_consumer_audience",
        "brand_safe",
        "rejection_class",
        "rationale",
    ],
    "properties": {
        "supports_consumer_audience": {"type": "boolean"},
        "brand_safe": {"type": "boolean"},
        "rejection_class": {"enum": list(REJECTION_CLASSES)},
        "rationale": {"type": "string", "minLength": 1},
    },
    "allOf": [
        {
            "if": {"properties": {"rejection_class": {"const": "accepted"}}},
            "then": {
                "properties": {
                    "supports_consumer_audience": {"const": True},
                    "brand_safe": {"const": True},
                }
            },
            "else": {
                "anyOf": [
                    {"properties": {"supports_consumer_audience": {"const": False}}},
                    {"properties": {"brand_safe": {"const": False}}},
                ]
            },
        }
    ],
}


class StructuredGenerator(Protocol):
    def generate(self, prompt: str, schema: dict[str, object]) -> object: ...


@dataclass(frozen=True)
class ArticleJudgment:
    supports_consumer_audience: bool
    brand_safe: bool
    rejection_class: str
    rationale: str


@dataclass(frozen=True)
class ClassificationAttempt:
    attempt: int
    raw_output: object | None
    validation_valid: bool
    error: str | None


@dataclass(frozen=True)
class ArticleClassification:
    page_id: int
    canonical_title: str
    prompt: str
    accepted: bool
    decision_reason: str
    judgment: ArticleJudgment | None
    attempts: tuple[ClassificationAttempt, ...]

@dataclass(frozen=True)
class ArticleClassificationResult:
    accepted: tuple[ArticleClassification, ...]
    rejected: tuple[ArticleClassification, ...]
    decisions: tuple[ArticleClassification, ...]


def classify_articles(
    articles: tuple[CanonicalArticle, ...],
    generator: StructuredGenerator,
    *,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> ArticleClassificationResult:
    decisions = tuple(
        classify_article(article, generator, sleep=sleep, jitter=jitter)
        for article in articles
    )
    return ArticleClassificationResult(
        accepted=tuple(decision for decision in decisions if decision.accepted),
        rejected=tuple(decision for decision in decisions if not decision.accepted),
        decisions=decisions,
    )


def classify_article(
    article: CanonicalArticle,
    generator: StructuredGenerator,
    *,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> ArticleClassification:
    prompt = _article_prompt(article)
    attempts: list[ClassificationAttempt] = []
    for attempt_number in range(1, 4):
        raw_output: object | None = None
        try:
            raw_output = generator.generate(prompt, ARTICLE_JUDGMENT_SCHEMA)
            parsed_output = (
                json.loads(raw_output) if isinstance(raw_output, str) else raw_output
            )
            jsonschema.validate(parsed_output, ARTICLE_JUDGMENT_SCHEMA)
            assert isinstance(parsed_output, dict)
            judgment = ArticleJudgment(**parsed_output)
            attempts.append(
                ClassificationAttempt(attempt_number, raw_output, True, None)
            )
            accepted = judgment.rejection_class == "accepted"
            return ArticleClassification(
                article.page_id,
                article.canonical_title,
                prompt,
                accepted,
                "accepted" if accepted else judgment.rejection_class,
                judgment,
                tuple(attempts),
            )
        except Exception as error:
            attempts.append(
                ClassificationAttempt(
                    attempt_number,
                    raw_output,
                    False,
                    f"{type(error).__name__}: {error}",
                )
            )
            if attempt_number < 3:
                sleep((2 ** (attempt_number - 1)) + jitter())
    return ArticleClassification(
        article.page_id,
        article.canonical_title,
        prompt,
        False,
        "exhausted_attempts",
        None,
        tuple(attempts),
    )


def _article_prompt(article: CanonicalArticle) -> str:
    return (
        "Decide whether this Wikipedia attention signal supports a commercially "
        "meaningful, brand-safe consumer audience. Reject tragedy, violent crime, "
        "death-driven attention, routine politics, isolated non-commercial news, "
        "and any signal without a defensible consumer audience.\n\n"
        f"Title: {article.canonical_title}\n"
        f"Lead extract: {article.extract}\n"
        f"Categories: {', '.join(article.categories)}"
    )


class GroqStructuredGenerator:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1/chat/completions",
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = (
            model or os.environ.get("AUDIENCE_TREND_MINER_MODEL") or DEFAULT_MODEL
        )
        self.base_url = base_url

    def generate(self, prompt: str, schema: dict[str, object]) -> object:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is required for article classification")
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "article_judgment",
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
        ).encode()
        api_request = request.Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(api_request, timeout=60) as response:
            payload = json.load(response)
        return payload["choices"][0]["message"]["content"]


class FixtureStructuredGenerator:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses

    @classmethod
    def from_file(cls, path: os.PathLike[str]) -> FixtureStructuredGenerator:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(list(payload["responses"]))

    def generate(self, prompt: str, schema: dict[str, object]) -> object:
        if not self.responses:
            raise RuntimeError("classification fixture responses exhausted")
        response = self.responses.pop(0)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(str(response["error"]))
        return response
