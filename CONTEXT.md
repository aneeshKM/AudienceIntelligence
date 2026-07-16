# Audience Trend Miner domain language

## Candidate Universe

The complete union of raw English Wikipedia titles returned by the daily top-page discovery responses for every day in the current analysis window. It is complete only when all seven discovery days succeed within the retry allowance.

## Canonical Article

One resolved English Wikipedia page, identified by `page_id`, with its canonical title, lead extract, categories, and aggregated Alias Traffic. Exactly one Canonical Article is emitted for each successfully resolved `page_id`.

## Alias Traffic

The dated Pageviews observations and derived previous/current window totals belonging to one raw candidate title. Multiple aliases may contribute Alias Traffic to the same Canonical Article.

## Wikimedia Attention Acquisition

The operation that builds a Candidate Universe, retrieves exact Alias Traffic and Wikipedia metadata, resolves Canonical Articles, aggregates aliases, and returns traceable raw evidence and structured failures. It does not publish run artifacts.
