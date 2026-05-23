from pathlib import Path
from fastembed import ImageEmbedding
from qdrant_edge import EdgeShard, SearchRequest, Query

SHARD_DIR = "./shard"
VECTOR_NAME = "image"
IMAGES_DIR = Path("data/images")

def main():
    if not Path(SHARD_DIR).exists():
        print("Index not found. Run index.py first.")
        return

    shard = EdgeShard.load(SHARD_DIR)
    embedder = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    
    example_img = IMAGES_DIR / "amul_butter" / "01.jpg"
    if not example_img.exists():
        print(f"Example image {example_img} not found.")
        return
        
    print(f"Querying using image: {example_img}")
    vecs = list(embedder.embed([example_img]))
    
    results = shard.search(
        SearchRequest(
            query=Query.Nearest(vecs[0].tolist(), using=VECTOR_NAME),
            limit=3,
            with_payload=True
        )
    )
    
    print("\nSearch Results:")
    for res in results:
        payload = res.payload
        print(f"- {payload['name']} (Score: {res.score:.4f})")

if __name__ == "__main__":
    main()