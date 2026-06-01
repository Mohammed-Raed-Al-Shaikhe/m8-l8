import json
import weaviate
from sentence_transformers import SentenceTransformer

from retrieval import bm25_search, dense_search

client = weaviate.Client("http://localhost:18080")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/retrieval_eval.jsonl", "r", encoding="utf-8") as f:
    eval_rows = [json.loads(line) for line in f]

for row in eval_rows:
    query = row["query"]
    gold = row["gold_doc_id"]

    bm25 = bm25_search(client, query, 10)
    dense = dense_search(client, query, 10, embedder)

    bm25_hit = gold in bm25[:5]
    dense_hit = gold in dense[:5]

    if bm25_hit and not dense_hit:
        print("=" * 60)
        print("BM25 WIN")
        print("Query:", query)
        print("Gold:", gold)

    if dense_hit and not bm25_hit:
        print("=" * 60)
        print("DENSE WIN")
        print("Query:", query)
        print("Gold:", gold)