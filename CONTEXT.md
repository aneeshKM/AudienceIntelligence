# Audience Trend Miner domain language

## Candidate Universe

The complete union of raw English Wikipedia titles returned by the daily top-page discovery responses for every day in the current analysis window. It is complete only when all seven discovery days succeed within the retry allowance.

## Canonical Article

One resolved English Wikipedia page, identified by `page_id`, with its canonical title, lead extract, categories, and aggregated Alias Traffic. Exactly one Canonical Article is emitted for each successfully resolved `page_id`.

## Alias Traffic

The dated Pageviews observations and derived previous/current window totals belonging to one raw candidate title. Multiple aliases may contribute Alias Traffic to the same Canonical Article.

## Wikimedia Attention Acquisition

The operation that builds a Candidate Universe, retrieves exact Alias Traffic and Wikipedia metadata, resolves Canonical Articles, aggregates aliases, and returns traceable raw evidence and structured failures. It does not publish run artifacts.

## Effective Run Configuration

The immutable, non-secret facts resolved exactly once when a run starts. Shell
values take precedence over `.env` values, which take precedence over global
defaults. A normal run requires live LLM credentials; explicit test and CI runs
may select fixture adapters. Run artifacts record safe provenance such as model
name and adapter modes, never credentials, local paths, or secret-source details.

## Wikimedia Evidence Fetching

The resumable operation that completes discovery and retrieves raw Pageviews and
metadata evidence without forming Alias Traffic or Canonical Articles. It returns
only after the complete Candidate Universe has terminal Pageviews and metadata
evidence, projected as an immutable typed value. Fetch jobs are run-scoped,
idempotent, and leased through PostgreSQL. Raw Wikimedia evidence and terminal
failures are persisted so work can resume after interruption.

## Wikimedia Attention Transformation

The synchronous, deterministic operation that consumes the immutable terminal
evidence projected from persisted Wikimedia evidence, derives Alias Traffic, and
forms Canonical Articles. It has no database or worker dependency and is safely
replayed after interruption. Canonical Article formation receives every alias in
the complete Candidate Universe after fetching has completed or permanently
failed.

## Evidence Job

One PostgreSQL-backed, run-scoped item of Wikimedia fetching work: discovery,
Pageviews, or metadata. An Evidence Job is claimed with an expiring lease, has a
bounded attempt history, and is uniquely identified within its run so resumed
workers cannot duplicate completed work. Deterministic Wikimedia Attention
Transformation does not create Evidence Jobs.

## Qualified Signal

A Canonical Article whose aggregated current-window traffic is at least 100,000, exceeds its previous-window traffic, has a positive capped scale-and-acceleration score, and is not explicit deterministic noise. A Qualified Signal is an input to later audience formation; it is not itself an accepted audience.

## Classified Signal

A Qualified Signal retained by a strict article-level model judgment because it
supports a commercially meaningful, brand-safe consumer audience. Invalid,
partial, or unavailable judgments fail closed after three total attempts. Every
attempt remains traceable even when the signal is rejected.

## Deterministic Noise

An unmistakable technical or navigational Wikipedia target: Main Page or a page in an explicitly enumerated technical namespace. List and index articles are not Deterministic Noise merely because of their title form.

## Run Publication

The atomic operation that accepts finished domain results and effective run facts, assembles and validates the complete artifact bundle, renders the report, stages all files, and exposes the timestamped run directory only after every artifact is complete. Domain degradation is publishable; publication failure leaves no completed run directory.

## Accepted Refined Audience

A group of at least two distinct Canonical Articles retained from a candidate
component by a schema-valid validate or split decision and then cleared by a
separate cluster-level tragedy and violent-crime safety assessment. Canonical
Article traffic belongs to at most one Accepted Refined Audience; alternative
matches are audit evidence only.
