"""Public interface for the local V2 run server."""

from audience_trend_miner.v2.ui.server import DEFAULT_HOST, create_app, serve

__all__ = ["DEFAULT_HOST", "create_app", "serve"]
