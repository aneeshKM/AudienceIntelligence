# ADR-0002: Cluster-first, country-specific AudienceIntelligence V2

## Status

Proposed. This ADR records the design to implement after the V2 research and
live API experiments are complete.

## Context

V1 discovers global English Wikipedia pages in the current window, fetches exact
per-article project traffic, qualifies individual articles, and clusters only
the surviving signals. That order cannot recognize an audience whose attention
moves between a generic topic, products, and brands across adjacent windows.

The Wikimedia country endpoint provides United States top-page observations but
differs materially from the V1 project endpoint: it spans all Wikimedia
projects, publishes `views_ceil`, and censors pages outside the daily top 1,000.
The product therefore needs a new acquisition contract and must make its
uncertainty explicit.

## Decision

Build V2 as a separate cluster-first pipeline:

1. Request 14 days of `top-per-country/US` Analytics data.
2. Keep only `en.wikipedia` records and union both windows.
3. Resolve canonical content metadata with the best official MediaWiki request
   pattern selected by live implementation-phase testing.
4. Represent each page with title/lead and cleaned categories.
5. Combine content and category cosine similarity with `70/30` weights.
6. Form preliminary connected components at an empirically selected threshold.
7. Discard singletons.
8. Rank preliminary components by mean whole-component Combined Similarity and
   select at most `AUDIENCE_TREND_MINER_MAX_LLM_CLUSTERS`, default `10`.
9. Run one isolated LangGraph adjudicator per selected component. Use LangChain
   integrations for a bounded proposal, critique, and conditional single
   revision that may keep, split, or reject members and apply commercial/safety
   rules.
10. Attach country traffic only after membership becomes terminal.
11. Classify robust growth, robust shrinking, or uncertainty mathematically.
12. Rank both robust directions together and narrate at most ten clusters with
    one model call per cluster.

Implement the pipeline as five independently runnable modules: Wikimedia
Evidence, Semantic Audience Formation, Cluster Adjudication, Trend Portfolio,
and Run Publication. Each consumes a minimal, complete, schema-versioned
artifact for the same `run_id`. The normal global command runs them in order and
resumes at the first incomplete module; stage commands expose the same interfaces
for manual execution and recovery. All long-running work emits human-readable
progress and can emit schema-versioned structured JSON events containing the run
ID, sequence, timestamp, module, operation, level, message, and optional bounded
progress.

Run Publication atomically publishes `portfolio.json`, `audit.json`, and
`manifest.json`. `portfolio.json` is the UI-facing structured contract; Run
Publication does not render a static report.

After the five pipeline modules, provide a separate local interactive UI backed
by FastAPI. The UI invokes the existing global CLI rather than reimplementing
pipeline behavior. It starts or resumes a stable `run_id`, consumes the CLI's
structured event stream, and shows live installer-style progress in one
scrollable area. The backend owns the subprocess independently of a browser
connection and supports event replay after reconnection. On successful Run
Publication, the same page renders the final Audience Portfolio from
`portfolio.json`.

The browser connects when the subprocess starts. The CLI flushes each structured
event, and the backend reads and forwards it incrementally without waiting for a
module boundary, process exit, or Run Publication. Replay is used only to fill a
sequence gap after reconnection and does not delay delivery to a connected UI.

The final UI shows audience cards with direction, trend summary, percentage
change, coverage, commercial interpretation, buying-power assessment, the
limitation note, and a plain empty state. Operational and agent audit detail does
not appear in the final cards. Failed runs retain their progress history and can
resume through the same global CLI and `run_id`.

V2 stores minimal derived evidence rather than V1's complete raw response and
similarity artifact set. Cross-run caching, navigation templates, cross-cluster
merging, and uncertain-trend display are deferred.

## Consequences

- Audience identity is independent of the analysis window and traffic values.
- Brand or product pages can contribute to the same audience as a generic topic.
- The pipeline detects both increasing and decreasing observed attention.
- Country-specific values are censored and rounded, so the system reports robust
  direction only when conservative ranges do not overlap.
- LLM scale is controlled by `AUDIENCE_TREND_MINER_MAX_LLM_CLUSTERS`, which
  accepts a positive integer or `all`. The default ten selected components use
  two to three normal adjudication calls each; transient delivery attempts are
  retried and audited. There is no independent global 30-call ceiling.
- The proposal/critique/conditional-revision graph provides explicit agentic
  reasoning while preserving isolated, monotonic component membership.
- Independently runnable modules improve recovery and experimentation but
  require stable artifact schemas between stages.
- The interactive UI remains a replaceable consumer of the CLI event and
  `portfolio.json` contracts; it does not become a sixth pipeline
  implementation.
- Long-running local runs remain observable across browser reconnects, while
  process lifecycle and resume behavior stay owned by the backend and CLI.
- FastAPI adds a local application server and subprocess-lifecycle boundary that
  must default to loopback and validate run identifiers and CLI arguments.
- The result is less replayable than V1 because raw content, singleton records,
  embeddings, and full pairwise matrices are intentionally not retained.
- V1 implementation and documentation remain historical references until V2 is
  implemented and becomes the active README contract.

## Rejected alternatives

### Qualify articles before clustering

Rejected because different pages can represent the same audience across the two
windows, and early page-level qualification destroys that relationship.

### Use page-view gaps to decide semantic membership

Rejected because correlated traffic does not establish a shared audience.

### Send the full page universe to one LLM

Rejected because of context size, hallucination risk, cost, and poor membership
accounting.

### Iteratively reconcile clusters across LLM calls

Rejected for V2 because cross-cluster reconciliation is order-dependent, can
loop, and weakens the rule that each page has one terminal state. The bounded
within-component proposal, critique, and single revision do not move pages
between preliminary components.

### Use navigation templates in V2

Deferred because extracting reliable navboxes adds expensive per-page rendered
content requests. Title/lead and categories are sufficient for the initial V2
experiment.

### Abort when any one of 14 daily responses is unavailable

Rejected as too strict for development. V2 requires at least four successful
days in each window and normalizes unequal coverage to a seven-day equivalent.

### Use a static HTML report as the V2 UI

Rejected because a static report cannot start or resume the long-running CLI,
stream module progress, preserve installer-style failure context, or transition
from execution to results in one experience. The interactive UI renders the
published `portfolio.json` contract after following the structured CLI event
stream.
