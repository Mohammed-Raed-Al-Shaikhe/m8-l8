# Comparison Brief — Module 8 Lab

> ~300–500 words. Replace the placeholder text in each section with your analysis.

## Metrics Table

All columns are required. The per-query-type breakdown (factoid vs. paraphrastic recall@5) is the column that exposes the BM25-vs-dense divergence; the aggregate columns alone hide it.

| Retriever      | recall@5 | recall@10 |   MRR | factoid recall@5 | paraphrastic recall@5 |
| -------------- | -------: | --------: | ----: | ---------------: | --------------------: |
| BM25           |    0.567 |     0.650 | 0.549 |            1.000 |                 0.133 |
| Dense          |    0.900 |     0.933 | 0.670 |            0.833 |                 0.967 |
| Hybrid (α=0.5) |    0.850 |     0.983 | 0.705 |            0.967 |                 0.733 |

## Where BM25 Wins

1. **Query:** "What is the AskDeveloperIfThereIsDocumentationLyingSomewhereAround pseudocode about?"  
   **Gold doc_id:** programmers:121844  
   BM25 retrieves the correct document because the query contains the highly distinctive identifier `AskDeveloperIfThereIsDocumentationLyingSomewhereAround`. Exact token matching is very effective here because the identifier appears directly in the relevant document, while dense retrieval is less effective with such rare technical terms.

2. **Query:** "What is the EmissionFactorIndirectBackgroundVolatisation enum value in this nitrogen-loss code?"  
   **Gold doc_id:** programmers:136639  
   BM25 benefits from the exact enum name appearing in the query. The long and uncommon identifier acts as a strong lexical signal that uniquely identifies the relevant document, making keyword-based retrieval particularly effective.

## Where Dense Wins

1. **Query:** "How to make my page load faster?"  
   **Gold doc_id:** webmasters:15380  
   Dense retrieval succeeds because it captures the semantic meaning of website performance optimization even when the relevant document uses different wording. BM25 depends on direct term overlap and therefore ranks the correct document lower.

2. **Query:** "Getting Overwhelmed: Tips for noobs"  
   **Gold doc_id:** programmers:141427  
   This query is conversational and contains few distinctive keywords. Dense retrieval understands the broader meaning related to beginner developers, learning challenges, and feeling overwhelmed, while BM25 struggles because of limited lexical overlap.

## Alpha Recommendation

I recommend using a hybrid alpha of **0.5** for this corpus. The evaluation results show that BM25 performs exceptionally well on factoid queries, achieving a factoid recall@5 of **1.000**, while dense retrieval performs best on paraphrastic queries with a paraphrastic recall@5 of **0.967**. Since the corpus contains both query styles, a balanced alpha allows the retriever to benefit from exact keyword matching and semantic similarity at the same time.

Although dense retrieval achieved the highest recall@5 overall (**0.900**), hybrid retrieval achieved the highest recall@10 (**0.983**) and the highest MRR (**0.705**). These results suggest that α = 0.5 provides the best balance between lexical and semantic retrieval signals and is the most suitable default value for this dataset.
