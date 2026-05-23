# ShelfScan Grocery Search

Retail shelf image search: Qdrant Edge + CLIP. Fully offline, no server.

## Install

```bash
pip install qdrant-edge-py fastembed Pillow
```

## Run

```bash
# 1. Index — embeds images with CLIP and stores in Qdrant Edge
python index.py

# 2. Search by image
python app.py

# 3. UI Interface using Streamlit with both Text and Image Search:

pip install streamlit
streamlit run ui.py
```

## Files

```
retail_edge/
├── data/
│   ├── products.json          ← 20-product mock catalog
│   └── images/                ← product images
└── scripts/
    ├── index.py  ← step 1
    ├── app.py    ← step 2
    └── ui.py     ← step 3
```

## Notes

- CLIP model (`Qdrant/clip-ViT-B-32-vision`, ~350 MB) downloads on first run via fastembed
- Index is persisted to `./shard/` — load across restarts with `EdgeShard.load()`
- Call `shard.optimize()` during low-traffic periods (merges segments, rebuilds indexes)