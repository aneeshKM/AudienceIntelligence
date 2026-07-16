from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from audience_trend_miner.configuration import ConfigurationError, load_run_configuration


class EffectiveRunConfigurationTest(unittest.TestCase):
    def test_resolves_shell_then_dotenv_then_global_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text(
                "GROQ_API_KEY=file-key\n"
                "AUDIENCE_TREND_MINER_MODEL=file/model\n"
                "DATABASE_URL=postgresql://file/db\n"
            )
            configuration = load_run_configuration(
                environ={"AUDIENCE_TREND_MINER_MODEL": "shell/model"},
                dotenv_path=dotenv,
            )

        self.assertEqual(configuration.groq_api_key, "file-key")
        self.assertEqual(configuration.model, "shell/model")
        self.assertEqual(configuration.database_url, "postgresql://file/db")

    def test_live_run_fails_at_startup_without_llm_credentials(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "GROQ_API_KEY"):
            load_run_configuration(environ={}, dotenv_path=Path("missing.env"))

    def test_fixture_mode_needs_no_live_key_and_exposes_only_safe_facts(self) -> None:
        configuration = load_run_configuration(
            environ={
                "AUDIENCE_TREND_MINER_CLASSIFICATION_FIXTURE": "/secret/local.json",
                "AUDIENCE_TREND_MINER_WIKIMEDIA_FIXTURE": "/secret/wiki.json",
                "DATABASE_URL": "postgresql://user:password@localhost/private",
                "AUDIENCE_TREND_MINER_TEST_MODE": "1",
            },
            dotenv_path=Path("missing.env"),
        )

        self.assertEqual(
            configuration.safe_provenance(),
            {
                "model": "openai/gpt-oss-120b",
                "classification_mode": "fixture",
                "wikimedia_mode": "fixture",
                "database_host": "localhost",
            },
        )

    def test_fixture_mode_is_rejected_outside_test_or_ci_mode(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "TEST_MODE"):
            load_run_configuration(
                environ={
                    "AUDIENCE_TREND_MINER_CLASSIFICATION_FIXTURE": "fixture.json"
                },
                dotenv_path=Path("missing.env"),
            )


if __name__ == "__main__":
    unittest.main()
