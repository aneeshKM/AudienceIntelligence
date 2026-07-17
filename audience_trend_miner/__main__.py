from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
from typing import Callable


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "v2-fixture-stage":
        return _fixture_stage_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "v2-wikimedia-evidence":
        return _wikimedia_evidence_main(sys.argv[2:])
    parser = argparse.ArgumentParser(prog="audience-trend-miner")
    parser.add_argument("--as-of", type=date.fromisoformat, required=False)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", required=False)
    arguments = parser.parse_args()
    from audience_trend_miner.run import execute_run

    execute_run(arguments.as_of, arguments.output_dir, run_id=arguments.run_id)
    return 0


def _wikimedia_evidence_main(arguments: list[str]) -> int:
    from audience_trend_miner.v2_wikimedia_evidence import (
        execute_wikimedia_evidence_fixture,
    )

    parser = argparse.ArgumentParser(
        prog="audience-trend-miner v2-wikimedia-evidence"
    )
    _add_v2_fixture_arguments(parser)
    parsed = parser.parse_args(arguments)
    sink = _v2_progress_sink(parsed.progress_format)
    return _execute_v2(
        lambda: execute_wikimedia_evidence_fixture(
            run_id=parsed.run_id,
            as_of_date=parsed.as_of,
            output_root=parsed.output_dir,
            fixture_path=parsed.fixture,
            progress_sink=sink,
        )
    )


def _fixture_stage_main(arguments: list[str]) -> int:
    from audience_trend_miner.v2_contracts import (
        execute_fixture_stage,
    )

    parser = argparse.ArgumentParser(prog="audience-trend-miner v2-fixture-stage")
    _add_v2_fixture_arguments(parser)
    parser.add_argument("--consume-existing", action="store_true")
    parser.add_argument("--interrupt-before-completion", action="store_true", help=argparse.SUPPRESS)
    parsed = parser.parse_args(arguments)
    sink = _v2_progress_sink(parsed.progress_format)
    return _execute_v2(
        lambda: execute_fixture_stage(
            run_id=parsed.run_id,
            configuration={"as_of": parsed.as_of.isoformat()},
            output_root=parsed.output_dir,
            fixture_path=parsed.fixture,
            progress_sink=sink,
            consume_existing=parsed.consume_existing,
            interrupt_before_completion=parsed.interrupt_before_completion,
        )
    )


def _add_v2_fixture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--progress-format", choices=("human", "json"), default="human"
    )


def _v2_progress_sink(progress_format: str):
    from audience_trend_miner.v2_contracts import (
        human_progress_sink,
        json_progress_sink,
    )

    return (
        json_progress_sink(sys.stdout)
        if progress_format == "json"
        else human_progress_sink(sys.stdout)
    )


def _execute_v2(action: Callable[[], object]) -> int:
    from audience_trend_miner.v2_contracts import V2ContractError

    try:
        action()
    except V2ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
