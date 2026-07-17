# V2 Semantic Formation Threshold and Subdivision Policy

> Status: Historical V2 research record supporting the active ADR-0002
> production threshold. References to V1 are experimental comparisons, not
> supported application behavior.

## Decision

V2 production Semantic Audience Formation uses an inclusive Combined
Similarity threshold of `0.76` with
`sentence-transformers/all-mpnet-base-v2` and the ADR-0002 content/category
weights of `0.70/0.30`.

An original connected component is oversized when its permitted Canonical Page
evidence plus fixed prompt allowance exceeds a conservative 16,384-token
input guard. The estimator counts one token for every UTF-8 byte in the compact
page-evidence serialization and reserves 2,048 tokens for the fixed prompt.
This deliberately overestimates normal model tokenization and does not require
the future adjudication adapter or a network tokenizer lookup.

Oversized components use the deterministic `stricter-boundary` policy. The
inclusive local boundary increases by `0.02` until connected subcomponents fit
the guard. Each original component is processed independently, so subdivision
cannot merge separate components. Every member is carried into exactly one
subdivision. If members remain indistinguishable at similarity `1.0`, stable
Canonical Page ID-order budget packing is the deterministic terminal fallback; it
partitions rather than truncates the tied members.

## Production experiment

The experiment used completed, real `top-per-country/US` evidence for the
fourteen nominal days anchored at as-of date 2026-07-17. Both Effective Windows
had 7/7 successful days. Canonicalization produced 3,654 Canonical Pages. The
production model embedded the complete Content Representations and Category
Representations locally; one vectorized Combined Similarity matrix was reused
for every threshold below.

| Threshold | Preliminary Clusters | Canonical Pages in Preliminary Clusters | Singletons | Largest component | Median cohesion | Minimum cohesion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 | 97 | 2,767 | 887 | 2,462 | 0.634 | 0.343 |
| 0.62 (V1) | 133 | 2,560 | 1,094 | 2,120 | 0.648 | 0.372 |
| 0.70 | 243 | 1,324 | 2,330 | 380 | 0.729 | 0.485 |
| 0.74 | 229 | 756 | 2,898 | 74 | 0.777 | 0.526 |
| 0.75 | 215 | 667 | 2,987 | 43 | 0.784 | 0.567 |
| **0.76** | **206** | **591** | **3,063** | **37** | **0.793** | **0.660** |
| 0.78 | 169 | 459 | 3,195 | 21 | 0.815 | 0.687 |
| 0.80 | 125 | 326 | 3,328 | 15 | 0.831 | 0.739 |

Cohesion is the mean Combined Similarity across every pair in a component, not
only graph edges. Thresholds through `0.70` showed graph percolation: one broad,
low-cohesion component dominated the retained Canonical Pages. The V1 value `0.62` was
therefore not transferable to the dual-representation V2 corpus.

At `0.76`, the last visibly weak chain present at `0.75` breaks: minimum
whole-component cohesion rises from `0.567` to `0.660`, while 591 Canonical
Pages remain available across 206 Preliminary Clusters. Representative largest groups were
recognizable tennis, national-football, calendar-date, Bosnia and Herzegovina,
and Ingalls-family neighborhoods. Raising the threshold to `0.78` or `0.80`
improved cohesion incrementally but discarded a further 132 or 265 Canonical
Pages as singletons. `0.76` is the selected balance between semantic coherence and
retained audience evidence; the token guard handles the remaining operationally
oversized cases without changing that global semantic boundary.

Running the selected production configuration through the stage applied the
guard to the 206 non-singleton connected components and produced 217 bounded
Preliminary Clusters. The increase of 11 Preliminary Clusters is subdivision only: the original 3,063
singleton count was unchanged and the stage retained every member of each
reviewable original component.

## Reproducibility and regression scope

The experiment artifact and embeddings remain local research evidence; raw
vectors and the full similarity matrix are not committed. Regression tests use
deterministic embedding adapters and fixed Canonical Page evidence. They exercise the
inclusive threshold, stricter-boundary subdivision, monotonic component
boundaries, and full membership retention without downloading a model or
calling Wikimedia.
