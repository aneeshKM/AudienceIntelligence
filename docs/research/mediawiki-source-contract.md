# MediaWiki production source contract

> Status: Historical V2 research record. The selected Action API contract is
> implemented in the active V2 Wikimedia Evidence module; the one-off comparison
> command used to collect this evidence was retired with V1 tooling.

## Decision

Use the English Wikipedia MediaWiki Action API for canonical identity and
semantic metadata:

```text
GET https://en.wikipedia.org/w/api.php
  ?action=query
  &format=json
  &formatversion=2
  &maxlag=5
  &redirects=1
  &converttitles=1
  &prop=extracts|categories
  &exintro=1
  &explaintext=1
  &clshow=!hidden
  &cllimit=max
  &titles=<up to 50 pipe-separated titles>
```

Follow the top-level `continue` object until absent. Merge category fragments by
page ID, deduplicate category titles, and remove the `Category:` prefix only
after the complete response is assembled. Preserve the Action API's
normalization and redirect maps as alias evidence. A page is usable only when it
has an integer `pageid`; a page carrying `missing: true` is a terminal missing
source, not a page with empty content.

This pattern is selected because one request can resolve up to 50 aliases and
return stable page IDs, canonical titles, plain-text lead sections, and visible
categories. It also represents redirects and missing titles in the same batch.
Neither REST alternative supplies that complete contract:

- RESTBase `page/summary/{title}` returns a good short plain-text lead and page
  ID, but no categories or batching; missing pages use HTTP 404.
- MediaWiki REST `page/{title}/with_html` returns stable identity and full
  rendered HTML, but no categories, no batched lookup, and a much larger body
  that would require additional lead extraction. Redirects can add an HTTP
  round trip.

Do not store broad experiment responses. Production Wikimedia Evidence stores
only the derived identity, lead, cleaned categories, alias lineage, and bounded
traffic facts required by its artifact schema.

## Live comparison

The retired bounded experiment emitted summaries only. On 2026-07-17 UTC it
queried `USA`, `Barack Obama`, and a deliberately missing
title, plus United States country traffic for 2025-07-15. The timings are one
sample and are evidence about request shape, not an availability SLA.

| Source | Result | Requests | Sample latency |
|---|---|---:|---:|
| Action API query | 2 pages plus 1 in-band missing title; redirect map; IDs, plain-text introductions (3,942 and 4,014 characters), visible categories | 1 | 0.2044 s |
| RESTBase summary | 2 HTTP 200 responses and 1 HTTP 404; same IDs and shorter plain-text leads (674 and 352 characters); no categories | 3 | 0.2088 s |
| MediaWiki REST with HTML | 2 HTTP 200 responses and 1 HTTP 404; same IDs and full rendered documents (about 2.6M and 2.4M characters); no categories | 5 including redirects | 0.9774 s |

`USA` resolved to page ID 3434750 and canonical title `United States` in the
Action response. The reproducible `--category-limit 10` comparison emitted
`clcontinue` and exhausted it in three requests, which verifies that category
continuation is real; the production `cllimit=max` sample completed in one
request. Action API documentation limits anonymous
`titles` queries to 50 titles and defines both redirect resolution and the
top-level continuation contract.

The country response contained exactly 1,000 cross-project records, including
778 `en.wikipedia` records. Every published `views_ceil` was divisible by 100;
the rank-1,000 daily cutoff was 4,600. The endpoint's public design records
ceiling-rounding to multiples of 100 and privacy filtering by daily actor count.
For an integer published value `C`, consume the conservative inclusive traffic
range `[max(0, C - 99), C]`. This is ceiling rounding, not nearest rounding and
not an exact count.

No tested response advertised a remaining request quota or `Retry-After`, and
the low-volume sequential experiment was not throttled. That does not imply an
unlimited contract. Wikimedia requires a descriptive User-Agent and compliance
with throttling instructions. Production reads should be sequential or
bounded, batch titles, use `maxlag`, honor `Retry-After`, and retry transient
429/5xx or `maxlag` failures with exponential backoff.

## Assumptions consumed downstream

### Wikimedia Evidence

- `top-per-country/US/all-access/YYYY/MM/DD` is one daily ranking across
  Wikimedia projects. Filter `project == "en.wikipedia"` only after recording
  the cutoff from all published records.
- The minimum published `views_ceil` is the conservative daily cutoff. On a
  successful day, a candidate page absent from the ranking has bounds
  `[0, cutoff]`; absence is censorship, not zero traffic.
- A present observation with `views_ceil = C` has bounds
  `[max(0, C - 99), C]`.
- The response article token is an alias. Action API page ID is the canonical
  identity; canonical title is display/source metadata. Redirect aliases that
  share a page ID remain separate traffic observations and are summed by date.
- Leads come from `exintro=1&explaintext=1`. Semantic categories come from
  `clshow=!hidden`; continuation must be exhausted before category cleaning.
- An unavailable country day is excluded from the Effective Window and is not
  zero-filled. Existing coverage policy still requires at least four successful
  days in each Nominal Window.

### Trend Portfolio

- Build cluster/day lower and upper traffic by summing member bounds, including
  `[0, cutoff]` for censored member-days.
- Sum only successful days in each Effective Window, then normalize both bounds
  by `7 / successful_day_count` for Seven-Day-Equivalent Traffic.
- Robust growth requires the current lower bound to exceed the previous upper
  bound; robust shrinking requires the current upper bound to be below the
  previous lower bound. Overlapping ranges produce Uncertain Direction.
- Use published values and their bounds consistently for Impact Score and
  narrative facts; never describe `views_ceil` as exact observed traffic.

## Official references

- [MediaWiki Action API query and continuation](https://www.mediawiki.org/wiki/API:Query)
- [MediaWiki Action API etiquette](https://www.mediawiki.org/wiki/API:Etiquette)
- [Wikimedia API Usage Guidelines](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines)
- [Wikimedia User-Agent Policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy)
- [Wikimedia Analytics page-view endpoints](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html)
- [Top-per-country design and privacy discussion](https://phabricator.wikimedia.org/T207171)
