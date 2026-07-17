# AudienceIntelligence V2 domain language

This glossary defines the intended V2 design. The implemented V1 behavior is
documented separately in `research/audience-trend-miner.md` and ADR-0001.

## Nominal Window

One of two adjacent seven-day calendar periods ending two days before the run's
`as-of` date. A Nominal Window may contain fewer successful Analytics days.

## Effective Window

The successful daily Analytics observations inside a Nominal Window. An
Effective Window is usable only when it contains at least four days. Its traffic
is normalized to a seven-day equivalent for comparison with the other window.

## Analytics Page Observation

One published `views_ceil` record returned by the United States
`top-per-country` endpoint. Its identity is the pair `(project, article)`. An
absent record means not observed in the daily top 1,000, not known zero traffic.

## V2 Candidate Universe

The union of titles from every successful day in both Effective Windows after
filtering to `project == "en.wikipedia"`. Unlike V1, V2 does not discover from
the current window alone.

## Canonical Page

One English Wikipedia page identified by stable page ID and canonical title,
with a cleaned lead of at most 600 characters and at most five selected visible
categories. Redirect aliases resolving to the same page ID form one Canonical
Page.

## Selected Category

A visible Wikipedia category remaining after deterministic noise removal and
ranking by inverse document frequency across the V2 Candidate Universe. A
Canonical Page retains at most five Selected Categories.

## Content Representation

The local embedding of a Canonical Page's title and cleaned lead.

## Category Representation

The local embedding of a Canonical Page's Selected Category values.

## Combined Similarity

The weighted pair relationship used for preliminary clustering:

```text
0.70 * cosine(Content Representation)
+ 0.30 * cosine(Category Representation)
```

The qualifying threshold is configuration selected from a live V2 experiment.

## Preliminary Cluster

A multi-page connected component in the graph of Canonical Pages whose Combined
Similarity meets the configured threshold. Singleton components are discarded
and not persisted. A token-oversized component is subdivided locally before LLM
review.

## LLM Cluster Selection

The semantic-cohesion ordering of reviewable Preliminary Clusters before agent
adjudication. Cohesion is mean Combined Similarity across every page pair, with
larger page count and stable page IDs as deterministic tie-breakers.
`AUDIENCE_TREND_MINER_MAX_LLM_CLUSTERS` defaults to `10` and accepts a positive
integer or `all`.

## Cluster Adjudication

The isolated LangGraph run for one selected Preliminary Cluster. Through
LangChain model and tool integrations it proposes a structured decision,
receives deterministic validation and semantic/commercial critique, and may
revise once. Its final result may keep the cluster, split it internally, reject
members, or reject the whole cluster. It cannot merge separate Preliminary
Clusters or reconsider a terminal page.

## Terminal Page State

The final, monotonic state of a reviewed Canonical Page: assigned to exactly one
Final Audience Cluster or rejected. No page may contribute traffic twice.

## Final Audience Cluster

A semantically coherent, commercially meaningful, brand-safe group of at least
two Canonical Pages produced by Cluster Adjudication.

## Observed Cluster Traffic

The sum of published daily page observations belonging to accepted members of a
Final Audience Cluster. Traffic is attached only after semantic membership is
terminal.

## Seven-Day-Equivalent Traffic

Observed Cluster Traffic divided by successful days in its Effective Window and
multiplied by seven. It allows windows with different available-day counts to be
compared without favoring the longer Effective Window.

## Robust Growth

A Final Audience Cluster whose conservative current-window minimum exceeds its
previous-window maximum.

## Robust Shrinking

A Final Audience Cluster whose conservative current-window maximum is below its
previous-window minimum.

## Uncertain Direction

A Final Audience Cluster whose conservative window ranges overlap. Uncertain
Direction clusters are excluded from the V2 UI.

## Impact Score

The symmetric scale-and-change score used to rank Robust Growth and Robust
Shrinking together:

```text
ln(1 + max(previous_views, current_views))
* min(abs(log2((current_views + 1) / (previous_views + 1))), 10)
```

## Cluster Review Budget

The configured bound on Preliminary Clusters selected for Cluster Adjudication,
not a separate global model-call ceiling. The default ten clusters each use two
to three normal workflow calls. A larger integer or `all` deliberately raises
the budget; transient provider attempts are retried and audited.

## Narrative Budget

At most ten final narrative calls per run, one for each top-ranked Robust Growth
or Robust Shrinking cluster selected for the UI.

## Minimal Run Evidence

The deliberately small V2 record containing run facts, coverage, accepted page
metadata, accepted page observations, rejected page ID/title/reason, cluster
decisions, mathematical trend facts, and final narratives. It excludes full API
responses, full Wikipedia content, singleton records, embeddings, and complete
pairwise similarities.

## V2 Runnable Module

One of Wikimedia Evidence, Semantic Audience Formation, Cluster Adjudication,
Trend Portfolio, or Run Publication. Each has a small interface, consumes a
complete schema-versioned artifact for the shared `run_id`, publishes its own
artifact atomically, and may be invoked independently. The global command runs
the same module interfaces in dependency order.

## UI-facing Portfolio

The structured `portfolio.json` consumed by `report.html`. The basic report
shows dates, audience name and direction, trend summary, percentage change,
coverage, commercial interpretation, buying-power rating and rationale, the
Wikipedia-attention limitation note, and a plain empty state. Internal evidence,
member titles, brand categories, agent records, and degradation details remain
outside the basic UI.

## Deferred V2 Capability

A deliberately excluded feature: additional countries or languages, navigation
templates, cross-cluster merging, rejected-page recovery, uncertain-trend UI,
cross-run caching, brand-category display, a separate critic model, exhaustive
adjudication as the default, or predictive claims.
