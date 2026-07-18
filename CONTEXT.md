# AudienceIntelligence glossary

Canonical domain terms used across AudienceIntelligence specifications, ADRs,
code, and product output.

## As-of Date

The date from which the analysis periods are derived. It anchors a run without
implying that data for that date is complete or included.

## Nominal Window

One of two adjacent seven-day calendar periods derived from the As-of Date. It
describes the intended period even when some daily observations are unavailable.

## Effective Window

The successful daily observations available within a Nominal Window. It is the
evidence actually used to measure attention for that period.

## Analytics Page Observation

A published daily attention value for a page identity in the source analytics
data. A missing observation means “not observed,” not “zero attention.”

## Candidate Universe

The complete set of page identities eligible for consideration in a run before
canonicalization, semantic grouping, or review.

## Canonical Page

One English Wikipedia page identified by its stable page ID and canonical title.
Aliases and redirects that resolve to the same page belong to one Canonical Page.

## Selected Category

A visible Wikipedia category retained as meaningful semantic evidence for a
Canonical Page after generic or administrative category noise is removed.

## Content Representation

The semantic representation of a Canonical Page derived from its canonical title
and cleaned lead text.

## Category Representation

The semantic representation of a Canonical Page derived from its Selected
Categories.

## Combined Similarity

The semantic relatedness between two Canonical Pages using both their Content
Representations and Category Representations.

## Preliminary Cluster

A group of Canonical Pages formed from semantic similarity before qualitative
review. It is a candidate grouping, not yet a market-facing audience.

## Cluster Adjudication

The review of a Preliminary Cluster for semantic coherence, commercial meaning,
brand safety, and valid audience membership.

## Terminal Page State

The final membership state of a reviewed Canonical Page: assigned to one Final
Audience Cluster or rejected. A page cannot occupy both states.

## Final Audience Cluster

A semantically coherent, commercially meaningful, brand-safe group containing
at least two distinct Canonical Pages.

## Observed Cluster Traffic

The sum of available Analytics Page Observations for the accepted Canonical Pages
in a Final Audience Cluster during an Effective Window.

## Seven-Day-Equivalent Traffic

Observed Cluster Traffic normalized to a seven-day period so Effective Windows
with different coverage can be compared on the same basis.

## Robust Growth

A trend direction assigned when the evidence supports that cluster attention
increased between the two windows despite observational uncertainty.

## Robust Shrinking

A trend direction assigned when the evidence supports that cluster attention
decreased between the two windows despite observational uncertainty.

## Uncertain Direction

A trend direction assigned when the available evidence does not reliably
distinguish growth from shrinking.

## Impact Score

A symmetric ranking measure that combines the scale of cluster attention with
the magnitude of change, regardless of direction.

## Audience Portfolio

The ranked collection of accepted audience trends produced by a run for product
consumption. It may be empty when no cluster meets the acceptance criteria.

## Run Evidence

The facts needed to interpret and audit a run, including its dates, coverage,
page observations, membership outcomes, trend facts, and published narratives.
