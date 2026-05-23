from pathlib import Path
from fastembed import TextEmbedding
from qdrant_edge import EdgeShard, QueryRequest, Query, Filter, FieldCondition, MatchValue

SHARD_DIR = "./shard"
TEXT_VECTOR_NAME = "text"

def main():
    if not Path(SHARD_DIR).exists():
        print("Index not found. Run index.py first.")
        return

    shard = EdgeShard.load(SHARD_DIR)
    embedder = TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")

    query_text = "butter dairy spread"
    print(f"Query: '{query_text}'")
    text_vecs = list(embedder.embed([query_text]))

    results = shard.query(
        QueryRequest(
            query=Query.Nearest(text_vecs[0].tolist(), using=TEXT_VECTOR_NAME),
            limit=3,
            with_payload=True,
            with_vector=False
        )
    )

    print(f"\nTop {len(results)} In-Stock Results:")
    print("\nSearch Results:")
    for res in results:
        payload = res.payload
        print(f"- {payload['name']} (Score: {res.score:.4f})")

    shard.close()

if __name__ == "__main__":
    main()
