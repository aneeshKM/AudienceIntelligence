from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "v2-fixture-stage":
        return _fixture_stage_main(sys.argv[2:])
    parser = argparse.ArgumentParser(prog="audience-trend-miner")
    parser.add_argument("--as-of", type=date.fromisoformat, required=False)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", required=False)
    arguments = parser.parse_args()
    from audience_trend_miner.run import execute_run

    execute_run(arguments.as_of, arguments.output_dir, run_id=arguments.run_id)
    return 0


def _fixture_stage_main(arguments: list[str]) -> int:
    from audience_trend_miner.v2_contracts import (
        V2ContractError,
        execute_fixture_stage,
        human_progress_sink,
        json_progress_sink,
    )

    parser = argparse.ArgumentParser(prog="audience-trend-miner v2-fixture-stage")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--progress-format", choices=("human", "json"), default="human")
    parser.add_argument("--consume-existing", action="store_true")
    parser.add_argument("--interrupt-before-completion", action="store_true", help=argparse.SUPPRESS)
    parsed = parser.parse_args(arguments)
    sink = json_progress_sink(sys.stdout) if parsed.progress_format == "json" else human_progress_sink(sys.stdout)
    try:
        execute_fixture_stage(
            run_id=parsed.run_id,
            configuration={"as_of": parsed.as_of.isoformat()},
            output_root=parsed.output_dir,
            fixture_path=parsed.fixture,
            progress_sink=sink,
            consume_existing=parsed.consume_existing,
            interrupt_before_completion=parsed.interrupt_before_completion,
        )
    except V2ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
