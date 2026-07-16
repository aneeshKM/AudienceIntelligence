# Audience Trend Miner — Research and V1 Design

## 1. Project objective

Build a prototype that turns public Wikipedia attention signals into an **Emerging Audience Portfolio** for marketers. The system should identify topics experiencing meaningful growth, remove non-commercial noise, group related topics into coherent targetable audiences, and explain their commercial potential.

An audience segment is useful when it is:

1. **Coherent:** its members share a real interest, behavior, trait, or intent.
2. **Sizable:** the observed attention is large enough to matter.
3. **Commercially meaningful:** relevant brand and product categories can be named credibly.

The portfolio must include, for each accepted audience:

- A market-friendly audience name
- An evidence-aware description of the trend
- An Estimated Size Index
- A qualitative Potential Buying Power rating and brand-category rationale

## 2. V1 scope

- Analyze **English Wikipedia only**.
- Treat additional languages and market-specific portfolios as future work.
- Provide a repeatable Python 3.12 CLI pipeline.
- Produce up to ten accepted audiences, with no forced minimum. An empty portfolio is preferable to fabricated segments.
- Write three primary outputs:
  - Machine-readable portfolio JSON
  - Polished static HTML report
  - Detailed `audit.json`
- Save lightweight reproducibility artifacts in a timestamped run directory:
  - Raw Wikimedia responses
  - Processed article records
  - Prompts
  - Model outputs
- A full offline replay engine is future scope.

## 3. Time windows and candidate discovery

The CLI accepts:

```text
--as-of YYYY-MM-DD
```

It defaults to the current date in UTC. The run manifest records the supplied date and all derived dates.

To avoid incomplete Pageviews data, use two complete rolling seven-day windows ending two days before `--as-of`:

- **Current window:** the most recent complete seven-day analysis window
- **Previous window:** the immediately preceding seven days

Candidate discovery works as follows:

1. Fetch the daily top 1,000 English Wikipedia pages for each day in the **current window only**.
2. Take the union of all raw titles returned across those seven responses.
3. For every candidate title, call the per-article Pageviews API for exact daily traffic across both windows.
4. Sum each window separately.

The daily top-page discovery responses define the candidate universe. If any one of those responses remains unavailable after retries, abort the run because a partial universe would introduce selection bias.

## 4. Canonicalization and alias aggregation

Keep every raw Pageviews title, but normalize candidates before scoring and clustering:

1. Resolve each raw title to its canonical Wikipedia `page_id` and canonical title.
2. Group all candidate titles resolving to the same `page_id`.
3. Sum their current-window and previous-window traffic.
4. Generate one embedding and one clustering entry per canonical page.
5. Preserve every alias and its individual traffic values in `audit.json`.

Use Wikimedia's Action API to retrieve:

- Canonical title
- `page_id`
- Short lead extract
- Categories

Pageviews remains the sole behavioral signal. Article metadata supports canonicalization, filtering, semantic representation, and explanation only.

## 5. Trend qualification and scoring

A canonical article qualifies as a trend candidate only when all of the following are true:

- Current-window views are at least **100,000**.
- Current-window views exceed previous-window views.
- Trend score is positive.

The trend score combines scale and acceleration:

```text
trend_score = ln(1 + current_views)
              * min(log2((current_views + 1) / (previous_views + 1)), 10)
```

The `+1` terms handle zero-view baselines. The base-2 growth term is capped at 10 so a brand-new or previously negligible page does not overwhelm all other signals.

The same formula is recomputed from aggregated cluster traffic and used to rank accepted audiences. Estimated Size Index remains a separate measure.

## 6. Noise and safety filtering

Filtering is intentionally conservative and auditable.

### 6.1 Deterministic filter

Remove only unmistakable technical and navigational noise, such as internal utility pages. Do not automatically remove all list or index pages: a page such as a current-year film list may carry commercially useful attention.

Every deterministic exclusion records a reason.

### 6.2 Article-level semantic filter

Use structured LLM classification to assess commercial relevance and sensitive content. Reject:

- Tragedies
- Violent crime
- Death-driven attention
- Routine politics
- Isolated news events without a defensible consumer audience
- Other topics that cannot support coherent commercial targeting

### 6.3 Cluster-level safety veto

Run a second safety assessment after cluster refinement. Reject any cluster materially centered on tragedy or violent crime, even when it has adjacent commercial topics.

V1 has a zero-tolerance acceptance criterion: **no sensitive tragedy or violent-crime cluster may enter the portfolio**.

## 7. Embeddings and custom clustering

Use a local Sentence Transformers model:

```text
sentence-transformers/all-mpnet-base-v2
```

Embed one representation per canonical article using its title and relevant Wikipedia context.

Implement a clean custom grouping loop:

1. Compute pairwise cosine similarity between surviving canonical articles.
2. Add a graph edge when similarity is at least `0.62`.
3. Treat connected components as candidate clusters.
4. Send each candidate cluster to the LLM for semantic validation, splitting, or rejection.

The `0.62` threshold is configurable and should be calibrated against a frozen evaluation fixture. LLM refinement is responsible for detecting semantic chaining and clusters whose articles are individually related but do not describe one coherent audience.

Additional cluster rules:

- Require at least two meaningfully distinct canonical articles.
- Preserve rejected singletons in the audit trail as standalone signals, not audiences.
- Assign each canonical article to at most one accepted audience.
- Alternative matches may be audited but never double-counted.

## 8. Audience naming and descriptions

An audience name must identify a targetable group of people through a shared interest, behavior, or intent.

- Good: **Eco-Home Modernizers**
- Bad: **Renewable Energy**

Names should contain two to five words. Reject a cluster if it cannot honestly be expressed as a coherent group rather than merely a topic.

Descriptions must distinguish evidence from interpretation. Pageviews demonstrate attention, not causation. Use language such as:

> Traffic rose across the supporting pages, suggesting increased interest in…

Observed traffic values may be stated as facts. Causal explanations must be framed as hypotheses unless supported by retrieved article content.

## 9. Estimated Size Index

The Size Index represents an audience's share of traffic within the accepted commercial portfolio:

```text
cluster current-window views
----------------------------------------------- * 100
current-window views across all accepted clusters
```

Excluded noise, rejected clusters, and unassigned articles do not enter the denominator because they would dilute meaningful commercial audiences.

Allocate integer basis points from 0 to 10,000 using the largest-remainder method. Expose each value as a two-decimal number from 0 to 100 in JSON and HTML. The complete portfolio must total exactly **100.00**.

## 10. Potential Buying Power

The LLM scores four dimensions from 1 to 3:

1. **Purchase intent**
2. **Typical transaction value**
3. **Breadth of relevant product categories**
4. **Brand safety**

Typical transaction value is used instead of inferred disposable income. Wikipedia traffic cannot establish reader wealth, but the underlying topics can indicate inexpensive purchases, subscriptions, travel, vehicles, renovations, homes, or other transaction categories.

Map the scores as follows:

- **High:** total score of 10–12 and brand safety is 3
- **Medium:** total score of 7–11 and brand safety is at least 2
- **Low:** total score of 4–6, or brand safety is 1

This prevents an active but unsafe audience from receiving a High rating. The output also names relevant brand categories and explains the rating from the component scores.

## 11. Model architecture

Use two narrow internal interfaces so the pipeline remains testable without building a multi-provider framework:

- Structured generation
- Embeddings

### Structured generation

- Provider: Groq directly
- Default model: `openai/gpt-oss-120b`
- Model ID remains configurable; `openai/gpt-oss-20b` may support cheaper development runs.
- Use strict JSON-schema structured outputs.
- Validate every model response against the application schema.

### Embeddings

- Provider: local Sentence Transformers
- Default model: `sentence-transformers/all-mpnet-base-v2`

## 12. Retry and failure behavior

Use one initial attempt plus two retries, for three total attempts. Apply exponential backoff with jitter to transient Wikimedia and Groq failures.

Schema-invalid model responses receive the same retry allowance. If structured generation still fails:

- Reject the affected article or cluster.
- Record the failure and attempts in the audit output.
- Never accept incomplete or partially parsed model data.

Failure boundary:

- Missing daily top-1,000 discovery data aborts the entire run.
- After discovery succeeds, isolate article-level and cluster-level failures.
- Continue with unaffected items.
- Mark the final report as degraded and include failure counts and reasons.

## 13. Auditability

The audit output should make every portfolio decision traceable. It includes:

- Run parameters and resolved analysis windows
- Raw candidate titles
- Canonical `page_id` and title
- Alias-level daily and window traffic
- Enriched Wikipedia metadata
- Qualification gates and trend scores
- Deterministic exclusion reasons
- LLM prompts, structured outputs, and validation status
- Article-level relevance and safety decisions
- Embedding model and clustering threshold
- Similarity edges and initial components
- Cluster refinement, split, rejection, and membership decisions
- Rejected singletons and alternative matches
- Cluster-level safety decisions
- Size Index calculations
- Buying Power component scores and final mapping
- Retry attempts and degraded-run status

## 14. V1 quality gates

Create at least one frozen evaluation fixture. The prototype is acceptable only when:

- At least 80% of accepted articles are judged commercially relevant.
- No accepted cluster contains an obviously unrelated article.
- At least four of the top five audiences receive a human-editor thumbs-up for coherence, naming, and brand usefulness.
- No sensitive tragedy or violent-crime cluster is accepted.
- 100% of consumed model responses pass schema validation.
- Schema generation failures are rejected after configured retries rather than accepted with incomplete data.
- Size Index calculations are exact.
- Article-to-alias-to-cluster lineage is complete and auditable.

## 15. Future scope

- Additional Wikipedia languages
- Market- or locale-specific portfolios
- Full offline replay from stored run artifacts
- Overlapping or multi-membership audience models with explicit traffic attribution
- Hosted dashboard, persistence layer, scheduling, and user accounts
- Broader behavioral or commercial data sources beyond Wikipedia

