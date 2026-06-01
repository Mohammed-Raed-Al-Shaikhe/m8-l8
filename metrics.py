from sentence_transformers import SentenceTransformer
import weaviate

from retrieval import (
    bm25_search,
    dense_search,
    hybrid_search,
    evaluate_retriever,
)

client = weaviate.Client("http://localhost:18080")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

eval_path = "data/retrieval_eval.jsonl"

bm25_metrics = evaluate_retriever(
    eval_path,
    lambda query, k: bm25_search(client, query, k),
)

dense_metrics = evaluate_retriever(
    eval_path,
    lambda query, k: dense_search(client, query, k, embedder),
)

hybrid_metrics = evaluate_retriever(
    eval_path,
    lambda query, k: hybrid_search(client, query, k, embedder, alpha=0.5),
)

print("BM25")
print(bm25_metrics)

print("\nDense")
print(dense_metrics)

print("\nHybrid")
print(hybrid_metrics)