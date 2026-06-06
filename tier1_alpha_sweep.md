## Tier 1 — Alpha Sweep and Per-Query Mode Selection

An alpha sweep was performed from 0.0 to 1.0 in increments of 0.1. Recall@5 increased steadily from 0.583 at α=0.0 to a maximum of 0.883 at α=0.6–0.8. Performance then declined slightly to 0.867 at α=0.9 and α=1.0. This indicates that a retrieval strategy biased toward dense retrieval performs best on this corpus, while still benefiting from some lexical BM25 signal.

The per-query analysis revealed three distinct patterns. First, identifier-heavy queries containing rare tokens, enum names, configuration directives, or class names favored lower alpha values. Examples include queries about "AskDeveloperIfThereIsDocumentationLyingSomewhereAround", "EmissionFactorIndirectBackgroundVolatisation", and ".htaccess FollowSymLinks". These queries benefit from BM25's exact token matching.

Second, paraphrastic natural-language queries favored higher alpha values. Examples include "Turn away a bug if no reproducible test case exists?" and "adb devices shows device as unauthorized". Dense retrieval successfully captures semantic similarity despite differences in wording.

Third, many queries produced identical results across nearly all alpha values. This suggests that both retrieval methods independently retrieve the correct document for a substantial portion of the corpus.

Based on these results, a global alpha between 0.6 and 0.8 is a strong default choice, with α=0.7 recommended. However, the variation between identifier-heavy and paraphrastic queries suggests that a per-query routing layer could outperform any single global alpha. Queries containing distinctive technical identifiers could be routed toward BM25-heavy retrieval, while natural-language questions could be routed toward dense-heavy retrieval. Such adaptive routing would likely achieve higher recall than a fixed alpha across the entire evaluation set.
