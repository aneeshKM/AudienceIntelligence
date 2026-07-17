from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import sys
from typing import Callable


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "v2-run":
        return _v2_run_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "v2-fixture-stage":
        return _fixture_stage_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "v2-wikimedia-evidence":
        return _wikimedia_evidence_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "v2-semantic-audience-formation":
        return _semantic_audience_formation_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "v2-cluster-adjudication":
        return _cluster_adjudication_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "v2-trend-portfolio":
        return _trend_portfolio_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "v2-run-publication":
        return _run_publication_main(sys.argv[2:])
    parser = argparse.ArgumentParser(prog="audience-trend-miner")
    parser.add_argument("--as-of", type=date.fromisoformat, required=False)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", required=False)
    arguments = parser.parse_args()
    from audience_trend_miner.run import execute_run

    execute_run(arguments.as_of, arguments.output_dir, run_id=arguments.run_id)
    return 0


def _v2_run_main(arguments: list[str]) -> int:
    from audience_trend_miner.v2.cluster_adjudication import (
        DEFAULT_CLUSTER_MODEL,
        FrozenStageAdapterFactory,
        ProductionStageAdapterFactory,
        execute_cluster_adjudication_stage,
    )
    from audience_trend_miner.v2.run_publication import (
        GlobalRunStages,
        execute_global_run,
        execute_run_publication,
    )
    from audience_trend_miner.v2.semantic_audience_formation import (
        execute_preliminary_clustering,
    )
    from audience_trend_miner.v2.semantic_audience_formation.clustering import (
        DEFAULT_SIMILARITY_THRESHOLD,
    )
    from audience_trend_miner.v2.semantic_audience_formation.embeddings import (
        DEFAULT_EMBEDDING_BATCH_SIZE,
        DEFAULT_EMBEDDING_MODEL,
        FrozenEmbeddingAdapter,
        SentenceTransformerEmbeddingAdapter,
    )
    from audience_trend_miner.v2.semantic_audience_formation.stage import (
        DEFAULT_REVIEW_CAP,
        parse_review_cap,
    )
    from audience_trend_miner.v2.shared import (
        V2ContractError,
        canonical_json_fingerprint,
    )
    from audience_trend_miner.v2.trend_portfolio import (
        DEFAULT_NARRATIVE_MODEL,
        FrozenNarrativeAdapterFactory,
        ProductionNarrativeAdapterFactory,
        execute_trend_portfolio_stage,
    )
    from audience_trend_miner.v2.wikimedia_evidence import (
        execute_wikimedia_evidence,
        execute_wikimedia_evidence_fixture,
    )

    def review_cap_argument(value: str):
        try:
            return parse_review_cap(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    parser = argparse.ArgumentParser(prog="audience-trend-miner v2-run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wikimedia-fixture", type=Path)
    parser.add_argument("--embedding-fixture", type=Path)
    parser.add_argument("--cluster-fixture", type=Path)
    parser.add_argument("--narrative-fixture", type=Path)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=os.environ.get(
            "AUDIENCE_TREND_MINER_SIMILARITY_THRESHOLD",
            str(DEFAULT_SIMILARITY_THRESHOLD),
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get(
            "AUDIENCE_TREND_MINER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=os.environ.get(
            "AUDIENCE_TREND_MINER_EMBEDDING_BATCH_SIZE",
            str(DEFAULT_EMBEDDING_BATCH_SIZE),
        ),
    )
    parser.add_argument(
        "--review-cap",
        type=review_cap_argument,
        default=os.environ.get(
            "AUDIENCE_TREND_MINER_MAX_LLM_CLUSTERS", str(DEFAULT_REVIEW_CAP)
        ),
    )
    parser.add_argument(
        "--cluster-model",
        default=os.environ.get(
            "AUDIENCE_TREND_MINER_CLUSTER_MODEL", DEFAULT_CLUSTER_MODEL
        ),
    )
    parser.add_argument(
        "--narrative-model",
        default=os.environ.get(
            "AUDIENCE_TREND_MINER_NARRATIVE_MODEL", DEFAULT_NARRATIVE_MODEL
        ),
    )
    parser.add_argument(
        "--progress-format", choices=("human", "json"), default="human"
    )
    parsed = parser.parse_args(arguments)
    if parsed.wikimedia_fixture is None and not parsed.database_url:
        parser.error("--database-url or DATABASE_URL is required without --wikimedia-fixture")
    if (
        parsed.cluster_fixture is None or parsed.narrative_fixture is None
    ) and not os.environ.get("GROQ_API_KEY"):
        parser.error(
            "GROQ_API_KEY is required when cluster or narrative fixtures are absent"
        )

    try:
        embedding_adapter = (
            FrozenEmbeddingAdapter.from_file(parsed.embedding_fixture)
            if parsed.embedding_fixture is not None
            else SentenceTransformerEmbeddingAdapter(
                model=parsed.embedding_model,
                batch_size=parsed.embedding_batch_size,
            )
        )
        cluster_factory = (
            FrozenStageAdapterFactory.from_file(parsed.cluster_fixture)
            if parsed.cluster_fixture is not None
            else ProductionStageAdapterFactory(model=parsed.cluster_model)
        )
        narrative_factory = (
            FrozenNarrativeAdapterFactory.from_file(parsed.narrative_fixture)
            if parsed.narrative_fixture is not None
            else ProductionNarrativeAdapterFactory(model=parsed.narrative_model)
        )
    except V2ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    def fixture_fingerprint(path: Path | None) -> str | None:
        if path is None:
            return None
        return canonical_json_fingerprint(
            json.loads(path.read_text(encoding="utf-8"))
        )

    global_configuration = {
        "as_of": parsed.as_of.isoformat(),
        "wikimedia": canonical_json_fingerprint(
            {
                "mode": "fixture" if parsed.wikimedia_fixture else "production",
                "fixture": fixture_fingerprint(parsed.wikimedia_fixture),
                "database": (
                    None
                    if parsed.wikimedia_fixture
                    else canonical_json_fingerprint(parsed.database_url)
                ),
            }
        ),
        "semantic_audience_formation": canonical_json_fingerprint(
            {
                "mode": "fixture" if parsed.embedding_fixture else "production",
                "fixture": fixture_fingerprint(parsed.embedding_fixture),
                "model": embedding_adapter.model,
                "batch_size": (
                    None if parsed.embedding_fixture else parsed.embedding_batch_size
                ),
                "similarity_threshold": parsed.similarity_threshold,
                "review_cap": parsed.review_cap,
            }
        ),
        "cluster_adjudication": canonical_json_fingerprint(
            {
                "mode": "fixture" if parsed.cluster_fixture else "production",
                "fixture": fixture_fingerprint(parsed.cluster_fixture),
                "model": cluster_factory.model,
            }
        ),
        "trend_portfolio": canonical_json_fingerprint(
            {
                "mode": "fixture" if parsed.narrative_fixture else "production",
                "fixture": fixture_fingerprint(parsed.narrative_fixture),
                "model": narrative_factory.model,
            }
        ),
    }

    if parsed.wikimedia_fixture is not None:
        def run_wikimedia(sink):
            return execute_wikimedia_evidence_fixture(
                run_id=parsed.run_id,
                as_of_date=parsed.as_of,
                output_root=parsed.output_dir,
                fixture_path=parsed.wikimedia_fixture,
                progress_sink=sink,
            )
    else:
        from audience_trend_miner.v2.wikimedia_evidence.adapters import (
            HttpWikimediaAdapter,
        )
        from audience_trend_miner.v2.wikimedia_evidence.jobs import EvidenceJobStore

        store = EvidenceJobStore(parsed.database_url)
        store.migrate()

        def run_wikimedia(sink):
            return execute_wikimedia_evidence(
                run_id=parsed.run_id,
                as_of_date=parsed.as_of,
                output_root=parsed.output_dir,
                adapter=HttpWikimediaAdapter(),
                store=store,
                progress_sink=sink,
                workers=parsed.workers,
            )

    stages = GlobalRunStages(
        wikimedia_evidence=run_wikimedia,
        semantic_audience_formation=lambda sink: execute_preliminary_clustering(
            run_id=parsed.run_id,
            output_root=parsed.output_dir,
            embedding_adapter=embedding_adapter,
            threshold=parsed.similarity_threshold,
            review_cap=parsed.review_cap,
            progress_sink=sink,
        ),
        cluster_adjudication=lambda sink: execute_cluster_adjudication_stage(
            run_id=parsed.run_id,
            output_root=parsed.output_dir,
            adapter_factory=cluster_factory,
            progress_sink=sink,
        ),
        trend_portfolio=lambda sink: execute_trend_portfolio_stage(
            run_id=parsed.run_id,
            output_root=parsed.output_dir,
            adapter_factory=narrative_factory,
            progress_sink=sink,
        ),
        run_publication=lambda sink: execute_run_publication(
            run_id=parsed.run_id,
            output_root=parsed.output_dir,
            progress_sink=sink,
        ),
    )
    return _execute_v2(
        lambda: execute_global_run(
            run_id=parsed.run_id,
            run_directory=parsed.output_dir / parsed.run_id,
            configuration=global_configuration,
            progress_sink=_v2_progress_sink(parsed.progress_format),
            stages=stages,
        )
    )


def _wikimedia_evidence_main(arguments: list[str]) -> int:
    from audience_trend_miner.v2.wikimedia_evidence import (
        execute_wikimedia_evidence,
        execute_wikimedia_evidence_fixture,
    )

    parser = argparse.ArgumentParser(
        prog="audience-trend-miner v2-wikimedia-evidence"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-format", choices=("human", "json"), default="human")
    parsed = parser.parse_args(arguments)
    sink = _v2_progress_sink(parsed.progress_format)
    if parsed.fixture is None:
        if not parsed.database_url:
            parser.error("--database-url or DATABASE_URL is required without --fixture")
        from audience_trend_miner.v2.wikimedia_evidence.jobs import EvidenceJobStore
        from audience_trend_miner.v2.wikimedia_evidence.adapters import (
            HttpWikimediaAdapter,
        )

        store = EvidenceJobStore(parsed.database_url)
        store.migrate()
        return _execute_v2(
            lambda: execute_wikimedia_evidence(
                run_id=parsed.run_id,
                as_of_date=parsed.as_of,
                output_root=parsed.output_dir,
                adapter=HttpWikimediaAdapter(),
                store=store,
                progress_sink=sink,
                workers=parsed.workers,
            )
        )
    return _execute_v2(
        lambda: execute_wikimedia_evidence_fixture(
            run_id=parsed.run_id,
            as_of_date=parsed.as_of,
            output_root=parsed.output_dir,
            fixture_path=parsed.fixture,
            progress_sink=sink,
        )
    )


def _semantic_audience_formation_main(arguments: list[str]) -> int:
    from audience_trend_miner.v2.semantic_audience_formation import (
        execute_category_selection,
        execute_preliminary_clustering,
    )
    from audience_trend_miner.v2.semantic_audience_formation.stage import (
        DEFAULT_REVIEW_CAP,
        parse_review_cap,
    )
    from audience_trend_miner.v2.semantic_audience_formation.embeddings import (
        DEFAULT_EMBEDDING_BATCH_SIZE,
        DEFAULT_EMBEDDING_MODEL,
        FrozenEmbeddingAdapter,
        SentenceTransformerEmbeddingAdapter,
    )
    from audience_trend_miner.v2.semantic_audience_formation.clustering import (
        DEFAULT_SIMILARITY_THRESHOLD,
    )

    def review_cap_argument(value: str):
        try:
            return parse_review_cap(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    parser = argparse.ArgumentParser(
        prog="audience-trend-miner v2-semantic-audience-formation"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wikimedia-evidence", type=Path)
    parser.add_argument("--embedding-fixture", type=Path)
    formation_mode = parser.add_mutually_exclusive_group()
    formation_mode.add_argument(
        "--similarity-threshold",
        type=float,
        default=os.environ.get(
            "AUDIENCE_TREND_MINER_SIMILARITY_THRESHOLD",
            str(DEFAULT_SIMILARITY_THRESHOLD),
        ),
        help=(
            "inclusive Combined Similarity boundary; the selected production "
            f"value is {DEFAULT_SIMILARITY_THRESHOLD}"
        ),
    )
    formation_mode.add_argument(
        "--category-selection-only",
        action="store_true",
        help="stop after deterministic Selected Category formation",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get(
            "AUDIENCE_TREND_MINER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=os.environ.get(
            "AUDIENCE_TREND_MINER_EMBEDDING_BATCH_SIZE",
            str(DEFAULT_EMBEDDING_BATCH_SIZE),
        ),
    )
    parser.add_argument(
        "--review-cap",
        type=review_cap_argument,
        default=os.environ.get(
            "AUDIENCE_TREND_MINER_MAX_LLM_CLUSTERS",
            str(DEFAULT_REVIEW_CAP),
        ),
        help="Preliminary Cluster review cap: a positive integer or 'all'",
    )
    parser.add_argument("--progress-format", choices=("human", "json"), default="human")
    parsed = parser.parse_args(arguments)
    sink = _v2_progress_sink(parsed.progress_format)
    if parsed.category_selection_only:
        if parsed.embedding_fixture is not None:
            parser.error(
                "--embedding-fixture cannot be used with --category-selection-only"
            )
        return _execute_v2(
            lambda: execute_category_selection(
                run_id=parsed.run_id,
                output_root=parsed.output_dir,
                wikimedia_evidence_path=parsed.wikimedia_evidence,
                progress_sink=sink,
            )
        )
    if parsed.embedding_fixture is not None:
        return _execute_v2(
            lambda: execute_preliminary_clustering(
                run_id=parsed.run_id,
                output_root=parsed.output_dir,
                wikimedia_evidence_path=parsed.wikimedia_evidence,
                embedding_adapter=FrozenEmbeddingAdapter.from_file(
                    parsed.embedding_fixture
                ),
                threshold=parsed.similarity_threshold,
                review_cap=parsed.review_cap,
                progress_sink=sink,
            )
        )
    return _execute_v2(
        lambda: execute_preliminary_clustering(
            run_id=parsed.run_id,
            output_root=parsed.output_dir,
            wikimedia_evidence_path=parsed.wikimedia_evidence,
            embedding_adapter=SentenceTransformerEmbeddingAdapter(
                model=parsed.embedding_model,
                batch_size=parsed.embedding_batch_size,
            ),
            threshold=parsed.similarity_threshold,
            review_cap=parsed.review_cap,
            progress_sink=sink,
        )
    )


def _fixture_stage_main(arguments: list[str]) -> int:
    from audience_trend_miner.v2.shared import (
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


def _cluster_adjudication_main(arguments: list[str]) -> int:
    from audience_trend_miner.v2.cluster_adjudication import (
        DEFAULT_CLUSTER_MODEL,
        FrozenStageAdapterFactory,
        ProductionStageAdapterFactory,
        execute_cluster_adjudication_stage,
    )

    parser = argparse.ArgumentParser(
        prog="audience-trend-miner v2-cluster-adjudication"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-audience-formation", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "AUDIENCE_TREND_MINER_CLUSTER_MODEL", DEFAULT_CLUSTER_MODEL
        ),
    )
    parser.add_argument(
        "--progress-format", choices=("human", "json"), default="human"
    )
    parser.add_argument(
        "--interrupt-before-completion",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parsed = parser.parse_args(arguments)
    if parsed.fixture is None and not os.environ.get("GROQ_API_KEY"):
        parser.error("GROQ_API_KEY is required without --fixture")
    adapter_factory = (
        FrozenStageAdapterFactory.from_file(parsed.fixture)
        if parsed.fixture is not None
        else ProductionStageAdapterFactory(model=parsed.model)
    )
    return _execute_v2(
        lambda: execute_cluster_adjudication_stage(
            run_id=parsed.run_id,
            output_root=parsed.output_dir,
            semantic_formation_path=parsed.semantic_audience_formation,
            adapter_factory=adapter_factory,
            progress_sink=_v2_progress_sink(parsed.progress_format),
            interrupt_before_completion=parsed.interrupt_before_completion,
        )
    )


def _trend_portfolio_main(arguments: list[str]) -> int:
    from audience_trend_miner.v2.trend_portfolio import (
        DEFAULT_NARRATIVE_MODEL,
        FrozenNarrativeAdapterFactory,
        ProductionNarrativeAdapterFactory,
        execute_trend_portfolio_stage,
    )

    parser = argparse.ArgumentParser(
        prog="audience-trend-miner v2-trend-portfolio"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wikimedia-evidence", type=Path)
    parser.add_argument("--cluster-adjudication", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "AUDIENCE_TREND_MINER_NARRATIVE_MODEL", DEFAULT_NARRATIVE_MODEL
        ),
    )
    parser.add_argument("--progress-format", choices=("human", "json"), default="human")
    parser.add_argument("--interrupt-before-completion", action="store_true", help=argparse.SUPPRESS)
    parsed = parser.parse_args(arguments)
    if parsed.fixture is None and not os.environ.get("GROQ_API_KEY"):
        parser.error("GROQ_API_KEY is required without --fixture")
    adapter_factory = (
        FrozenNarrativeAdapterFactory.from_file(parsed.fixture)
        if parsed.fixture is not None
        else ProductionNarrativeAdapterFactory(model=parsed.model)
    )
    return _execute_v2(
        lambda: execute_trend_portfolio_stage(
            run_id=parsed.run_id,
            output_root=parsed.output_dir,
            wikimedia_evidence_path=parsed.wikimedia_evidence,
            cluster_adjudication_path=parsed.cluster_adjudication,
            adapter_factory=adapter_factory,
            progress_sink=_v2_progress_sink(parsed.progress_format),
            interrupt_before_completion=parsed.interrupt_before_completion,
        )
    )


def _run_publication_main(arguments: list[str]) -> int:
    from audience_trend_miner.v2.run_publication import execute_run_publication

    parser = argparse.ArgumentParser(
        prog="audience-trend-miner v2-run-publication"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wikimedia-evidence", type=Path)
    parser.add_argument("--semantic-audience-formation", type=Path)
    parser.add_argument("--cluster-adjudication", type=Path)
    parser.add_argument("--trend-portfolio", type=Path)
    parser.add_argument(
        "--progress-format", choices=("human", "json"), default="human"
    )
    parser.add_argument(
        "--interrupt-before-completion", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--fail-after-artifact",
        type=int,
        choices=(1, 2, 3),
        help=argparse.SUPPRESS,
    )
    parsed = parser.parse_args(arguments)
    supplied_paths = {
        stage: path
        for stage, path in {
            "wikimedia-evidence": parsed.wikimedia_evidence,
            "semantic-audience-formation": parsed.semantic_audience_formation,
            "cluster-adjudication": parsed.cluster_adjudication,
            "trend-portfolio": parsed.trend_portfolio,
        }.items()
        if path is not None
    }
    return _execute_v2(
        lambda: execute_run_publication(
            run_id=parsed.run_id,
            output_root=parsed.output_dir,
            progress_sink=_v2_progress_sink(parsed.progress_format),
            upstream_paths=supplied_paths,
            interrupt_before_completion=parsed.interrupt_before_completion,
            fail_after_artifact=parsed.fail_after_artifact,
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
    from audience_trend_miner.v2.shared import (
        human_progress_sink,
        json_progress_sink,
    )

    return (
        json_progress_sink(sys.stdout)
        if progress_format == "json"
        else human_progress_sink(sys.stdout)
    )


def _execute_v2(action: Callable[[], object]) -> int:
    from audience_trend_miner.v2.shared import V2ContractError

    try:
        action()
    except V2ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
