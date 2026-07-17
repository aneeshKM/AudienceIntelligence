from __future__ import annotations

import json
import os
import unittest

import jsonschema

from audience_trend_miner.classification import (
    ARTICLE_JUDGMENT_SCHEMA,
    GroqStructuredGenerator,
)
from audience_trend_miner.configuration import load_run_configuration


LIVE_GROQ_TEST_ENABLED = (
    __name__ == "__main__" or os.environ.get("RUN_LIVE_GROQ_TESTS") == "1"
)


@unittest.skipUnless(
    LIVE_GROQ_TEST_ENABLED,
    "set RUN_LIVE_GROQ_TESTS=1 to call the live Groq API",
)
class LiveGroqIntegrationTest(unittest.TestCase):
    def test_configured_model_returns_strict_article_judgment(self) -> None:
        configuration = load_run_configuration()

        raw_output = GroqStructuredGenerator(
            api_key=configuration.groq_api_key,
            model=configuration.model,
        ).generate(
            (
                "Decide whether this Wikipedia article supports a commercially "
                "meaningful, brand-safe consumer audience. Title: Running shoe. "
                "Lead extract: Footwear designed for running and related sports. "
                "Categories: Sports footwear, Running equipment."
            ),
            ARTICLE_JUDGMENT_SCHEMA,
        )

        self.assertIsInstance(raw_output, str)
        output = json.loads(raw_output)
        jsonschema.validate(output, ARTICLE_JUDGMENT_SCHEMA)
        print(f"Groq article judgment: {output}")


if __name__ == "__main__":
    unittest.main()
