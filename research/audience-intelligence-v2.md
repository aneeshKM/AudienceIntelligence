# AudienceIntelligence — V2 Research and Design

## 1. Objective

AudienceIntelligence V2 identifies recent changes in United States attention to
English Wikipedia topics and turns coherent groups of pages into market-facing
audience trends. It reports both growing and shrinking attention. It does not
claim to predict future behavior, identify individual readers, or infer that a
page view caused commercial intent.

V2 is a redesign, not an incremental change to the V1 acquisition order. V1
qualifies individual articles before clustering; V2 forms semantic audiences
first and measures traffic change only after membership is final.

## 2. Scope

- Country: United States (`US`) only.
- Content project: English Wikipedia (`en.wikipedia`) only.
- Analysis horizon: two adjacent seven-day windows.
- Portfolio: at most ten robust growing or shrinking audience clusters.
- Additional countries and languages are future scope.
- Uncertain trends are excluded from the UI in V2.
- A targetable audience requires at least two distinct canonical pages.

## 3. Data sources

### 3.1 Wikimedia Analytics API

Use the country-specific endpoint:

```text
GET /metrics/pageviews/top-per-country/US/all-access/{year}/{month}/{day}
```

The endpoint returns the top 1,000 pages viewed from the United States across
all Wikimedia projects. Keep only records whose `project` is exactly
`en.wikipedia`. Preserve the Analytics identity as `(project, article)` until
canonical resolution.

The response supplies published `views_ceil` observations. A page missing from
a daily response is **not observed**, not known to have zero traffic.

### 3.2 MediaWiki content API

Use official English Wikipedia APIs to resolve the Analytics title and retrieve:

- canonical page ID;
- canonical title;
- a plain-text lead capped locally at 600 characters; and
- visible Wikipedia categories.

During implementation, test official live API alternatives before choosing the
production request pattern. Compare canonical identity, lead quality, category
completeness, batching and continuation, request count, latency, and throttling.
The likely fast path is batched Action API `action=query` requests with up to 50
titles. Rendered HTML, Beautiful Soup parsing, and navigation-template evidence
are not required in V2.

Full article HTML, full article text, raw content responses, and navigation
templates are not stored.

### 3.3 Application architecture

V2 is organized as five independently runnable modules with small interfaces and
minimal, schema-versioned artifacts between them:

1. **Wikimedia Evidence** acquires Analytics observations and MediaWiki metadata,
   canonicalizes pages, and publishes the evidence required downstream.
2. **Semantic Audience Formation** consumes completed Wikimedia Evidence,
   selects categories, embeds pages, forms preliminary clusters, ranks them by
   semantic cohesion, and applies the configured LLM-cluster cap.
3. **Cluster Adjudication** consumes selected preliminary clusters and their page
   evidence, runs the bounded LangGraph workflow, and publishes terminal accepted
   and rejected membership.
4. **Trend Portfolio** consumes terminal membership plus the relevant Wikimedia
   observations, attaches traffic, classifies direction, ranks robust trends, and
   generates the structured portfolio narratives.
5. **Run Publication** validates and atomically publishes `portfolio.json`,
   `audit.json`, and `manifest.json`.

The modules share a `run_id`. Each module validates that its upstream artifact is
complete and schema-compatible, reads it without rerunning upstream work, and
writes its own artifact atomically. A module refuses to run when a prerequisite
is absent or incomplete. Domain records cross module interfaces; Wikimedia
response objects, database rows, embedding-library values, LangChain messages,
LangGraph state, and HTML concerns remain inside their owning implementations.

The existing global command remains the normal interface and runs all five
modules in dependency order, resuming from the first incomplete module. Small
stage-specific CLI commands expose the same module interfaces for manual runs,
experiments, and recovery; they do not introduce a second implementation.

Each module has an explicit package under `audience_trend_miner/v2/` and exposes
a small public interface. Shared run, artifact, progress, validation, and
atomic-write primitives live in `audience_trend_miner/v2/shared/`; adapters and
schemas stay with their owning module. Internal files are split only when the
implementation requires it, and tests mirror module ownership under `tests/v2/`.

## 4. Windows and partial availability

The CLI continues to accept `--as-of YYYY-MM-DD`. Derive two adjacent nominal
seven-day windows ending two days before `--as-of`.

Request all 14 daily Analytics responses. A run may continue with missing days
when both windows contain at least four successful days. Therefore the minimum
usable run contains eight successful daily responses.

For unequal coverage, normalize observed cluster traffic to a seven-day
equivalent:

```text
seven_day_equivalent = observed_total / successful_window_days * 7
```

Record successful-day counts for both windows. A run stops when either window
has fewer than four successful days.

## 5. Candidate formation and canonicalization

1. Union English Wikipedia titles from every successful day in both windows.
2. Resolve each Analytics `(project, article)` identity through the selected
   MediaWiki request pattern.
3. Group redirects and aliases by canonical page ID.
4. Omit pages whose metadata remains unavailable.
5. Remove deterministic technical targets such as Main Page and enumerated
   internal namespaces.

Metadata failures do not receive a qualitative run label. Record only a neutral
count such as `metadata_pages_unavailable`.

## 6. Category selection

Exclude hidden categories and deterministically remove category names matching
audited noise patterns, including:

```text
^\d{4} births$
^\d{4} deaths$
^Living people$
^Possibly living people$
^Year of birth missing
^Year of death missing
^Articles with
^All articles
^CS1
^Webarchive
^Wikipedia:
```

Calculate inverse document frequency across the canonical-page universe:

```text
idf(category) = log(total_pages / pages_containing_category)
```

For each page, sort remaining categories by descending IDF with alphabetical
tie-breaking and retain at most five.

## 7. Semantic representation and preliminary clusters

Use local Sentence Transformers with the configurable default embedding model:

```text
sentence-transformers/all-mpnet-base-v2
```

Create two representations per canonical page:

1. canonical title plus cleaned lead;
2. the selected category values.

For every page pair, compute:

```text
combined_similarity =
    0.70 * cosine(content_embeddings)
  + 0.30 * cosine(category_embeddings)
```

Embedding inference must be batched and similarity computation vectorized. Do
not persist embeddings or the complete pairwise matrix.

Keep graph edges whose combined similarity meets a configurable threshold.
Connected components define preliminary clusters. Select the threshold through
an implementation-phase experiment on real 14-day US data; do not assume the V1
threshold of `0.62` remains appropriate.

There is no semantic page-count cap. Enforce a model-input token guard only. If
a component exceeds it, subdivide locally using a stricter similarity boundary
or graph-community method before the LLM stage. The final number of reviewable
preliminary clusters is `n`.

Rank reviewable preliminary clusters by mean Combined Similarity across every
page pair in the component. Break ties by descending page count and then a
stable ordering of canonical page IDs. Apply
`AUDIENCE_TREND_MINER_MAX_LLM_CLUSTERS` after ranking. Its default is `10`; it
accepts a positive integer or `all`. Missing configuration uses `10`, while an
empty, zero, negative, or malformed value fails configuration validation. The
run artifact records the configured value, eligible count, selected count, and
count omitted by the cap. Traffic never affects this semantic selection.

Discard singleton components. Do not store their titles, metadata, embeddings,
or traffic. An aggregate singleton-discard count is sufficient.

## 8. Bounded agentic cluster adjudication

Cluster adjudication must use LangGraph with LangChain model and tool
integrations. Start one isolated, bounded graph run for each preliminary cluster
selected by `AUDIENCE_TREND_MINER_MAX_LLM_CLUSTERS`. A graph cannot see or alter
another preliminary cluster.

Each graph follows this workflow:

```text
propose adjudication
-> deterministic validation
-> semantic and commercial critique
-> conditionally revise once
-> deterministic final validation
-> accept or reject
```

The proposal and critique are always bounded to the supplied component. Skip the
revision only when validation passes and the critic approves the proposal. The
graph never loops more than once and never performs cross-cluster split/merge
work. With the default selection cap, cluster adjudication uses two to three
normal model calls for each of at most ten clusters. There is no separate global
30-call ceiling: changing the cluster cap deliberately changes the call budget.

The model receives only the pages in that component. For each page, provide:

- page ID;
- canonical title value;
- cleaned lead value;
- selected category values.

Do not provide window membership, page views, traffic thresholds, cosine
scores, or unrelated clusters. V2 does not merge separate preliminary clusters
and does not reconsider rejected pages.

The model may:

- keep the component as one group;
- split it internally into multiple groups;
- reject individual pages; or
- reject the whole component.

The same decision assesses commercial relevance and safety. Reject tragedy,
violent crime, death-driven attention, routine politics, isolated news without
a defensible audience, semantic mismatches, and topics without a coherent
consumer audience.

Required response shape:

```json
{
  "groups": [
    {
      "name": "Home Air Purification",
      "page_ids": [101, 102],
      "rationale": "Shared consumer interest in residential air-cleaning products."
    }
  ],
  "rejected": [
    {
      "page_id": 103,
      "reason": "routine_politics"
    }
  ]
}
```

Every supplied page must occur exactly once across accepted groups and rejected
members. Accepted groups require at least two distinct canonical pages. Unknown
or duplicate page IDs invalidate a candidate decision. Deterministic validation
owns these rules. The critic challenges semantic coherence, commercial meaning,
brand safety, audience naming, and unsupported rationales. After the one allowed
revision, an invalid final decision rejects the component.

Use `AUDIENCE_TREND_MINER_CLUSTER_MODEL` for the proposer, critic, and reviser,
with separate role-specific prompts. Bind the adjudication and validation tools
through LangChain. Transient delivery failures may use one initial attempt plus
two later attempts per model step. Record the framework and model identifiers,
proposal/critique/revision status, validation outcome, and provider attempts.
Never request or persist hidden chain-of-thought.

Use separate configurable model settings:

- cluster adjudication: `AUDIENCE_TREND_MINER_CLUSTER_MODEL`;
- final narrative: `AUDIENCE_TREND_MINER_NARRATIVE_MODEL`.

The intended cluster model is `groq/compound-mini`, using JSON Object Mode plus
local deterministic schema validation. Pace production requests below the
provider RPM ceiling and honor `Retry-After` on `429` responses.
Use `openai/gpt-oss-120b` for final narratives.

## 9. Terminal membership

Page membership is monotonic:

```text
pending -> assigned
pending -> rejected
```

A canonical page contributes traffic to at most one final cluster. Pages cannot
move between preliminary components, rejected pages cannot be resurrected, and
V2 has no iterative membership split/merge loop across components. The bounded
critique and single revision in Cluster Adjudication may refine only the current
component before its membership becomes terminal.

For rejected members, store only page ID, title, and rejection reason. For
accepted members, store only:

- page ID and canonical title;
- cleaned lead capped at 600 characters;
- up to five selected categories;
- daily observed page views; and
- source window dates.

## 10. Cluster traffic and direction

Attach traffic only after semantic membership is terminal. For each accepted
cluster and window:

1. sum published observations for accepted member pages;
2. record observed page-days and successful Analytics days;
3. normalize the observed total to a seven-day equivalent.

Missing page-days remain censored observations. Construct conservative window
ranges using the available daily top-1,000 cutoff as the upper bound for a known
member missing on that day. The exact handling of `views_ceil` rounding must be
verified against the live API specification during implementation.

Classify direction as:

```text
robust_growth:
    current minimum > previous maximum

robust_shrinking:
    current maximum < previous minimum

sudden_growth:
    previous observed total = 0 and current observed total > 0

uncertain:
    ranges overlap
```

Do not apply a minimum current-window view gate. For sudden growth, publish no
percentage because division by a zero observed baseline is undefined; label the
trend "Suddenly trending" instead. Do not send any of these values or
classifications to the cluster LLM.

Show robust growing, robust shrinking, and suddenly trending clusters in the UI.
Exclude uncertain clusters in V2; displaying them is future scope.

## 11. Ranking

Rank robust growing and shrinking clusters together using a symmetric impact
score:

```text
impact_score =
    ln(1 + max(previous_views, current_views))
    * min(abs(log2((current_views + 1) / (previous_views + 1))), 10)
```

Select at most the top ten clusters before narrative generation.

## 12. Final narrative generation

Make one `openai/gpt-oss-120b` call per selected directional cluster, for at most ten
narrative calls per run. Do not batch all clusters into one prompt.

Deterministic code owns direction, view totals, percentage change, coverage,
confidence, and impact score. The model may return only:

- audience name;
- trend summary;
- commercial interpretation;
- relevant brand categories;
- buying-power rating; and
- buying-power rationale.

The narrative must describe supplied evidence without inventing traffic or
claiming that Wikipedia attention proves causation, identity, income, or future
behavior.

### 12.1 UI-facing portfolio contract

`portfolio.json` is the UI-facing structured contract. Run Publication does not
render a static HTML report. A separate interactive UI consumes the contract
rather than reconstructing portfolio meaning from internal pipeline objects or
progress logs.

The final portfolio view shows:

- effective `as-of` date;
- previous and current window dates;
- one card per selected robust audience, without an explanation of rank;
- audience name and Growing or Shrinking direction badge;
- trend summary, percentage traffic change, and per-audience coverage;
- commercial interpretation; and
- buying-power rating and rationale.

The view includes a short statement that Wikipedia attention does not prove
reader identity, intent, income, causation, or future behavior. When no robust
audience qualifies, show the plain sentence `No robust audience trends qualified
for this run.`

Do not show absolute seven-day-equivalent traffic, member Wikipedia titles,
brand categories, rank explanations, run degradation details, rejected or
uncertain clusters, similarity evidence, prompts, agent state, validation
diagnostics, or retry histories in the final portfolio cards. Keep required
operational and audit facts in machine-readable artifacts.

### 12.2 Interactive run UI

After the five pipeline modules, provide a separate local UI backed by FastAPI.
It invokes the existing global CLI as a subprocess and never reimplements module
behavior. The user supplies an As-of Date and starts or resumes a stable
`run_id`.

Use one scrollable, installer-style area for the complete experience. While the
CLI runs, append live module events, bounded progress, retries, warnings, and
terminal failures. On successful Run Publication, transition the same page to
the final Audience Portfolio loaded from `portfolio.json`. A failed run retains
its event history and offers retry or resume through the same `run_id`.

Live means that the browser connects when the subprocess starts and receives
each structured event as soon as the CLI emits and flushes it. The backend reads
the running process incrementally and forwards events without waiting for a
module boundary, process exit, or Run Publication. The CLI must use unbuffered
event output so long-running operations remain observable while they are in
progress.

The FastAPI backend owns the subprocess independently of the browser connection,
allows at most one active process per `run_id`, and supports replay followed by
live streaming after reconnection. Invoke the CLI with an argument array and no
shell. Bind the local server to loopback by default. Cancellation requires
explicit confirmation, terminates only the owned subprocess, records a terminal
event, and does not delete completed artifacts.

## 13. Request scheduling and failures

Wikimedia work is queued. A transiently failed request moves to the back of the
queue. Allow one initial attempt and two later attempts, for three total. Honor
`429`, `Retry-After`, `maxlag`, and server errors with backoff. Use a descriptive
User-Agent.

Do not repeat completed work within a run. Cross-run caching and reuse are future
scope.

Every module emits concise progress logs so the global command is never silent
during long work. Default human-readable events identify the module, current
operation, and bounded progress such as Analytics day, metadata batch, embedding
batch, selected-cluster count, Groq proposal/critique/revision activity, trend
qualification, and publication.

Support schema-versioned structured JSON events as the stable UI integration
contract. Each event contains the run ID, monotonic sequence, timestamp, module,
operation, level, message, and optional bounded progress. Persist enough event
history to replay events only after a browser disconnect, then continue the live
stream from the next sequence. Event persistence and replay must not delay
delivery to an already connected browser. A malformed event is surfaced safely
without allowing the UI to infer pipeline state from human-readable text. Never
log secrets or hidden chain-of-thought.

## 14. Minimal artifacts

V2 intentionally stores less than V1. Do not persist:

- full Analytics responses;
- full MediaWiki JSON or HTML;
- full Wikipedia article text;
- singleton page records;
- embeddings; or
- the complete pairwise similarity matrix.

Persist only the run facts and minimal accepted/rejected evidence needed by the
UI, downstream module contracts, deterministic calculations, resumability, and
auditability. Intermediate module artifacts contain only the minimal data needed
by their declared downstream consumers. Secrets remain excluded.

## 15. Implementation validation before production coding

Before committing to a content endpoint and similarity threshold:

1. live-test official Wikimedia API alternatives;
2. verify canonical resolution, redirects, missing pages, batching, continuation,
   category completeness, and local 600-character truncation;
3. measure latency and throttling behavior;
4. run the 14-day US acquisition in a private temporary location;
5. inspect cluster size distributions and representative components at several
   similarity thresholds; and
6. choose the endpoint pattern and threshold from observed evidence.

Live testing happens after this research phase and before implementation.

## 16. Future scope

- Countries other than the United States.
- Wikimedia projects and languages other than English Wikipedia.
- Navigation-template evidence and rendered-HTML parsing.
- Cross-cluster reconciliation and whole-cluster merging.
- Re-clustering or recovery of rejected pages.
- Displaying uncertain trends.
- Displaying relevant brand categories in the final portfolio view.
- Remote multi-user UI deployment, authentication, authorization, or shared
  worker scheduling; the V2 UI is local and loopback-bound by default.
- Making exhaustive cluster adjudication the default by changing
  `AUDIENCE_TREND_MINER_MAX_LLM_CLUSTERS` from `10` to `all`; retain the setting
  as an operational safety override.
- Configuring a separate critic model or adding cross-cluster agent reasoning.
- Cross-run metadata and embedding caching.
- Approximate-nearest-neighbor indexing for a much larger universe.
- Global optimization or overlapping audience membership.
- Claims of predictive power; V2 remains an observed trend detector.
