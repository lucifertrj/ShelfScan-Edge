import json
from pathlib import Path
from fastembed import ImageEmbedding, TextEmbedding
from qdrant_edge import (
    Distance, EdgeConfig, EdgeVectorParams, EdgeShard,
    Point, UpdateOperation, PayloadSchemaType
)

SHARD_DIR   = "./shard"
IMAGE_VECTOR_NAME = "image"
TEXT_VECTOR_NAME = "text"
VECTOR_DIM  = 512
BATCH_SIZE  = 8
IMAGES_DIR  = Path("data/images")
CATALOG     = Path("data/products.json")

def get_shard() -> EdgeShard:
    config = EdgeConfig(
        vectors={
            IMAGE_VECTOR_NAME: EdgeVectorParams(
                size=VECTOR_DIM,
                distance=Distance.Cosine
            ),
            TEXT_VECTOR_NAME: EdgeVectorParams(
                size=VECTOR_DIM,
                distance=Distance.Cosine
            )
        }
    )

    if Path(SHARD_DIR).exists() and any(Path(SHARD_DIR).iterdir()):
        return EdgeShard.load(SHARD_DIR)
    Path(SHARD_DIR).mkdir(parents=True, exist_ok=True)

    shard = EdgeShard.create(SHARD_DIR, config)
    # payload indexes for filtered search
    for field, schema in [
        ("category", PayloadSchemaType.Keyword),
        ("in_stock",  PayloadSchemaType.Bool),
        ("price",     PayloadSchemaType.Float),
    ]:
        shard.update(UpdateOperation.create_field_index(field, schema))

    return shard

def main():
    products = json.loads(CATALOG.read_text())
    image_embedder = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    text_embedder = TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")
    shard    = get_shard()

    for i in range(0, len(products), BATCH_SIZE):
        batch = products[i:i + BATCH_SIZE]
        
        points = []
        for p in batch:
            product_img_dir = IMAGES_DIR / p["image_dir"]
            image_paths = list(product_img_dir.glob("*.jpg"))
            if not image_paths:
                continue

            image_vecs = list(image_embedder.embed([image_paths[0]]))
            image_vector = image_vecs[0].tolist()

            text_description = f"{p['name']} {p['brand']} {p['category']} {' '.join(p['tags'])}"
            text_vecs = list(text_embedder.embed([text_description]))
            text_vector = text_vecs[0].tolist()

            points.append(
                Point(
                    id=p["id"],
                    vector={
                        IMAGE_VECTOR_NAME: image_vector,
                        TEXT_VECTOR_NAME: text_vector
                    },
                    payload={
                        "name":     p["name"],
                        "brand":    p["brand"],
                        "category": p["category"],
                        "price":    p["price"],
                        "aisle":    p["aisle"],
                        "shelf":    p["shelf"],
                        "in_stock": p["in_stock"],
                        "stock_qty":p["stock_qty"],
                        "barcode":  p["barcode"],
                        "tags":     p["tags"],
                    },
                )
            )
            
        if points:
            shard.update(UpdateOperation.upsert_points(points))
            
        print(f"Indexed {min(i + BATCH_SIZE, len(products))}/{len(products)}")

    shard.optimize()
    shard.close()
    print("Done.")

if __name__ == "__main__":
    main()