import json

import weaviate
from sentence_transformers import SentenceTransformer

from retrieval import hybrid_search

client = weaviate.Client("http://localhost:18080")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/retrieval_eval.jsonl") as f:
    rows = [json.loads(line) for line in f]

alphas = [i / 10 for i in range(11)]

for row in rows:

    query = row["query"]
    gold = row["gold_doc_id"]

    best_alphas = []

    for alpha in alphas:

        results = hybrid_search(
            client,
            query,
            5,
            embedder,
            alpha=alpha,
        )

        if gold in results:
            best_alphas.append(alpha)

    print()
    print("QUERY:", query)
    print("BEST ALPHAS:", best_alphas)