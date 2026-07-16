# Audience Trend Miner domain language

## Candidate Universe

The complete union of raw English Wikipedia titles returned by the daily top-page discovery responses for every day in the current analysis window. It is complete only when all seven discovery days succeed within the retry allowance.

## Canonical Article

One resolved English Wikipedia page, identified by `page_id`, with its canonical title, lead extract, categories, and aggregated Alias Traffic. Exactly one Canonical Article is emitted for each successfully resolved `page_id`.

## Alias Traffic

The dated Pageviews observations and derived previous/current window totals belonging to one raw candidate title. Multiple aliases may contribute Alias Traffic to the same Canonical Article.

## Wikimedia Attention Acquisition

The operation that builds a Candidate Universe, retrieves exact Alias Traffic and Wikipedia metadata, resolves Canonical Articles, aggregates aliases, and returns traceable raw evidence and structured failures. It does not publish run artifacts.

## Qualified Signal

A Canonical Article whose aggregated current-window traffic is at least 100,000, exceeds its previous-window traffic, has a positive capped scale-and-acceleration score, and is not explicit deterministic noise. A Qualified Signal is an input to later audience formation; it is not itself an accepted audience.

## Deterministic Noise

An unmistakable technical or navigational Wikipedia target: Main Page or a page in an explicitly enumerated technical namespace. List and index articles are not Deterministic Noise merely because of their title form.
