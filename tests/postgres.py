from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit


def test_database_url() -> str:
    configured = os.environ.get("TEST_DATABASE_URL")
    if configured:
        return configured

    runtime_url = os.environ.get("DATABASE_URL") or _dotenv_database_url()
    if not runtime_url:
        return "postgresql://localhost:5432/audience_intelligence_test"

    parsed = urlsplit(runtime_url)
    credentials = ""
    if parsed.username:
        credentials = quote(parsed.username, safe="")
        if parsed.password:
            credentials += f":{quote(parsed.password, safe='')}"
        credentials += "@"
    netloc = f"{credentials}{parsed.hostname or 'localhost'}:5432"
    return urlunsplit(
        (
            parsed.scheme or "postgresql",
            netloc,
            "/audience_intelligence_test",
            parsed.query,
            "",
        )
    )


def _dotenv_database_url() -> str | None:
    path = Path(".env")
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        name, separator, value = raw_line.partition("=")
        if separator and name.strip() == "DATABASE_URL":
            return value.strip()
    return None
