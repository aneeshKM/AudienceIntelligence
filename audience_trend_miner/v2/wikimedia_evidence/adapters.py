"""Source adapters used by the V2 Wikimedia Evidence stage.

The HTTP transport remains shared with the runnable V1 pipeline so both
contracts retain identical Wikimedia source semantics during V1 preservation.
"""

from audience_trend_miner.wikimedia import (
    CountryTopPagesResponse,
    HttpWikimediaAdapter,
    WikimediaPermanentError,
)

__all__ = [
    "CountryTopPagesResponse",
    "HttpWikimediaAdapter",
    "WikimediaPermanentError",
]
