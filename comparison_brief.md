# Comparison Brief — Module 8 Lab

> ~300–500 words. Replace the placeholder text in each section with your analysis.

## Metrics Table

| Retriever | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 | _your number_ | _your number_ | _your number_ |
| Dense | _your number_ | _your number_ | _your number_ |
| Hybrid (α=0.5) | _your number_ | _your number_ | _your number_ |

Optionally include a per-`query_type` breakdown (factoid vs. paraphrastic).

## Where BM25 Wins

Identify 2 specific labeled queries where BM25 returns the gold doc but dense
does not (or ranks it much lower). For each: quote the query, name the gold
doc_id, and explain in 1–2 sentences why BM25's lexical signal works here
(rare token, exact identifier, etc.).

1. _query 1 — explanation_
2. _query 2 — explanation_

## Where Dense Wins

Same exercise, reversed: 2 queries where dense returns the gold doc but BM25
does not (or ranks it much lower).

1. _query 1 — explanation_
2. _query 2 — explanation_

## Alpha Recommendation

Recommend an alpha for this corpus and justify the choice. Consider the mix
of query types in the labeled set and the kinds of queries you expect from
real users of this corpus. A range (e.g., 0.4–0.6) with a default value is an
acceptable answer.
