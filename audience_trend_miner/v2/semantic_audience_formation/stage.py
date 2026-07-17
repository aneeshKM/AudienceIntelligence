from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from audience_trend_miner.v2.semantic_audience_formation.categories import (
    CATEGORY_RULE_SET_VERSION,
    CategorySelection,
    select_categories,
)
from audience_trend_miner.v2.semantic_audience_formation.clustering import (
    EmbeddingAdapter,
    PreliminaryClusterArtifact,
    form_preliminary_clusters,
)
from audience_trend_miner.v2.shared import BoundedProgress, ProgressEvent, ProgressSink
from audience_trend_miner.v2.wikimedia_evidence import consume_wikimedia_evidence


STAGE = "semantic-audience-formation"


def execute_category_selection(
    *,
    run_id: str,
    output_root: Path,
    progress_sink: ProgressSink,
    wikimedia_evidence_path: Path | None = None,
) -> CategorySelection:
    """Validate Wikimedia Evidence and form deterministic Selected Categories."""
    evidence_path = wikimedia_evidence_path or (
        output_root / run_id / "wikimedia-evidence.json"
    )
    artifact = consume_wikimedia_evidence(evidence_path, run_id=run_id)
    payload = artifact["payload"]
    assert isinstance(payload, dict)
    canonical_pages = payload["canonical_pages"]
    assert isinstance(canonical_pages, list)
    selection = select_categories(canonical_pages)
    page_count = len(selection.pages)
    progress_sink(
        ProgressEvent(
            run_id=run_id,
            sequence=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            module=STAGE,
            operation="select-categories",
            level="info",
            message=(
                "selected meaningful categories with rule set "
                f"{CATEGORY_RULE_SET_VERSION}"
            ),
            progress=BoundedProgress(page_count, max(page_count, 1)),
        )
    )
    return selection


def execute_preliminary_clustering(
    *,
    run_id: str,
    output_root: Path,
    progress_sink: ProgressSink,
    embedding_adapter: EmbeddingAdapter,
    threshold: float,
    wikimedia_evidence_path: Path | None = None,
) -> PreliminaryClusterArtifact:
    """Select semantic evidence and form fixture- or production-backed clusters."""
    selection = execute_category_selection(
        run_id=run_id,
        output_root=output_root,
        progress_sink=progress_sink,
        wikimedia_evidence_path=wikimedia_evidence_path,
    )
    result = form_preliminary_clusters(
        selection.pages, embedding_adapter, threshold=threshold
    )
    page_count = len(selection.pages)
    progress_sink(
        ProgressEvent(
            run_id=run_id,
            sequence=2,
            timestamp=datetime.now(timezone.utc).isoformat(),
            module=STAGE,
            operation="form-preliminary-clusters",
            level="info",
            message=(
                f"formed {len(result.preliminary_clusters)} Preliminary Clusters; "
                f"discarded {result.singleton_count} singleton components; "
                f"using model {result.embedding_model!r}"
            ),
            progress=BoundedProgress(page_count, max(page_count, 1)),
        )
    )
    return result
