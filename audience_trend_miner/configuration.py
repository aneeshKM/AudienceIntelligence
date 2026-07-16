from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit


DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_DATABASE_URL = "postgresql://localhost/audience_intelligence"


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class EffectiveRunConfiguration:
    groq_api_key: str
    model: str
    database_url: str
    classification_fixture: Path | None
    wikimedia_fixture: Path | None
    wikimedia_base_url: str | None

    def safe_provenance(self) -> dict[str, str]:
        return {
            "model": self.model,
            "classification_mode": (
                "fixture" if self.classification_fixture else "live"
            ),
            "wikimedia_mode": (
                "fixture"
                if self.wikimedia_fixture
                else "disabled"
                if self.wikimedia_base_url == ""
                else "live"
            ),
            "database_host": urlsplit(self.database_url).hostname or "local",
        }


def load_run_configuration(
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path = Path(".env"),
) -> EffectiveRunConfiguration:
    environment = os.environ if environ is None else environ
    dotenv = _dotenv_values(dotenv_path)

    def value(name: str, default: str = "") -> str:
        return environment.get(name) or dotenv.get(name) or default

    classification_fixture = value("AUDIENCE_TREND_MINER_CLASSIFICATION_FIXTURE")
    wikimedia_fixture = value("AUDIENCE_TREND_MINER_WIKIMEDIA_FIXTURE")
    if (classification_fixture or wikimedia_fixture) and value(
        "AUDIENCE_TREND_MINER_TEST_MODE"
    ) != "1":
        raise ConfigurationError(
            "fixture adapters require AUDIENCE_TREND_MINER_TEST_MODE=1"
        )
    api_key = value("GROQ_API_KEY")
    if not classification_fixture and not api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is required unless explicit classification fixture mode is selected"
        )
    if "AUDIENCE_TREND_MINER_WIKIMEDIA_BASE_URL" in environment:
        base_url: str | None = environment[
            "AUDIENCE_TREND_MINER_WIKIMEDIA_BASE_URL"
        ]
    elif "AUDIENCE_TREND_MINER_WIKIMEDIA_BASE_URL" in dotenv:
        base_url = dotenv["AUDIENCE_TREND_MINER_WIKIMEDIA_BASE_URL"]
    else:
        base_url = None
    return EffectiveRunConfiguration(
        groq_api_key=api_key,
        model=value("AUDIENCE_TREND_MINER_MODEL", DEFAULT_MODEL),
        database_url=value("DATABASE_URL", DEFAULT_DATABASE_URL),
        classification_fixture=Path(classification_fixture)
        if classification_fixture
        else None,
        wikimedia_fixture=Path(wikimedia_fixture) if wikimedia_fixture else None,
        wikimedia_base_url=base_url,
    )


def _dotenv_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, raw_value = line.partition("=")
        if not separator or not key.strip():
            continue
        parsed = raw_value.strip()
        if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in "\"'":
            parsed = parsed[1:-1]
        values[key.strip()] = parsed
    return values
