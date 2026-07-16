from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from audience_trend_miner.run import execute_run


def main() -> int:
    parser = argparse.ArgumentParser(prog="audience-trend-miner")
    parser.add_argument("--as-of", type=date.fromisoformat, required=False)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", required=False)
    arguments = parser.parse_args()
    execute_run(arguments.as_of, arguments.output_dir, run_id=arguments.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
