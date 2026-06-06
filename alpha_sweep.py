import matplotlib.pyplot as plt
import weaviate
from sentence_transformers import SentenceTransformer

from retrieval import hybrid_search, evaluate_retriever

client = weaviate.Client("http://localhost:18080")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

alphas = [i / 10 for i in range(11)]
recalls = []

for alpha in alphas:

    metrics = evaluate_retriever(
        "data/retrieval_eval.jsonl",
        lambda query, k: hybrid_search(
            client,
            query,
            k,
            embedder,
            alpha=alpha,
        ),
    )

    recall = metrics["recall@5"]

    recalls.append(recall)

    print(
        f"alpha={alpha:.1f}",
        f"recall@5={recall:.3f}"
    )

plt.figure(figsize=(8,5))
plt.plot(alphas, recalls, marker="o")
plt.xlabel("Alpha")
plt.ylabel("Recall@5")
plt.title("Recall@5 vs Alpha")
plt.grid(True)

plt.savefig("alpha_sweep.png")