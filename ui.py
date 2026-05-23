import json
import time
from pathlib import Path

import streamlit as st
from fastembed import ImageEmbedding, TextEmbedding
from PIL import Image
from qdrant_edge import EdgeShard, Query, QueryRequest


SHARD_DIR = "./shard"
IMAGE_VECTOR_NAME = "image"
TEXT_VECTOR_NAME = "text"
IMAGES_DIR = Path("data/images")
CATALOG_PATH = Path("data/products.json")

SAMPLE_QUERIES = [
    "creamy dairy spread for breakfast toast",
    "chilled fizzy cola beverage",
    "quick cooking noodles with spicy masala",
    "toothpaste for strong teeth",
    "liquid detergent for washing machine",
    "crunchy potato chips snack",
]

SAMPLE_IMAGES = [
    ("amul_butter", "Butter"),
    ("orange_juice", "Juice"),
    ("lays_classic", "Chips"),
    ("maggi", "Noodles"),
    ("coca_cola", "Cola"),
    ("colgate", "Toothpaste"),
]


st.set_page_config(
    page_title="ShelfScan: Grocery Search",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; max-width: 1180px; }
        h1 { letter-spacing: 0; font-size: 2.3rem; margin-bottom: 0.15rem; }
        h2, h3, h4 { letter-spacing: 0; }
        .muted { color: #64748b; margin-bottom: 1.5rem; }
        .meta {
            color: #64748b;
            font-size: 0.9rem;
            margin: -0.25rem 0 0.75rem;
        }
        .price {
            font-size: 1.1rem;
            font-weight: 700;
        }
        .badge {
            border: 1px solid #dbe3ef;
            border-radius: 999px;
            color: #334155;
            display: inline-block;
            font-size: 0.78rem;
            margin: 0 0.25rem 0.35rem 0;
            padding: 0.2rem 0.55rem;
        }
        .score {
            color: #0f766e;
            font-size: 0.9rem;
            font-weight: 700;
        }
        [data-testid="stMetricValue"] { font-size: 1.4rem; }
        .stButton > button { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_catalog():
    if not CATALOG_PATH.exists():
        return []
    return json.loads(CATALOG_PATH.read_text())


@st.cache_resource
def load_resources():
    if not Path(SHARD_DIR).exists():
        return None, None, None

    shard = EdgeShard.load(SHARD_DIR)
    image_embedder = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    text_embedder = TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")
    return shard, image_embedder, text_embedder


def product_image(product):
    path = IMAGES_DIR / product.get("image_dir", "") / "01.jpg"
    return path if path.exists() else None


def search_text(query):
    vector = list(text_embedder.embed([query]))[0].tolist()
    return shard.query(
        QueryRequest(
            query=Query.Nearest(vector, using=TEXT_VECTOR_NAME),
            limit=3,
            with_payload=True,
            with_vector=False
        )
    )


def search_image(image):
    vector = list(image_embedder.embed([image]))[0].tolist()
    return shard.query(
        QueryRequest(
            query=Query.Nearest(vector, using=IMAGE_VECTOR_NAME),
            limit=3,
            with_payload=True,
            with_vector=False
        )
    )


def render_product(product, score=None):
    image_path = product_image(product)
    if image_path:
        st.image(Image.open(image_path), width="stretch")

    st.markdown(f"#### {product['name']}")
    st.markdown(f"<div class='meta'>{product['brand']} · {product['category']}</div>", unsafe_allow_html=True)

    stock = f"In stock: {product['stock_qty']}" if product.get("in_stock") else "Out of stock"
    st.markdown(
        f"""
        <span class="badge">{stock}</span>
        <span class="badge">Aisle {product['aisle']}</span>
        <span class="badge">Shelf {product['shelf']}</span>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    left.markdown(f"<div class='price'>Rs. {product['price']}</div>", unsafe_allow_html=True)
    if score is not None:
        right.markdown(f"<div class='score'>Score {score:.4f}</div>", unsafe_allow_html=True)


def render_grid(items, catalog_lookup):
    for row_start in range(0, len(items), 3):
        cols = st.columns(3)
        for col, item in zip(cols, items[row_start : row_start + 3]):
            payload = item.payload if hasattr(item, "payload") else item
            product = catalog_lookup.get(getattr(item, "id", None), payload)
            score = getattr(item, "score", None)
            with col:
                with st.container(border=True):
                    render_product({**product, **payload}, score)

catalog = load_catalog()
catalog_lookup = {product["id"]: product for product in catalog}

with st.spinner("Loading local shard and CLIP models..."):
    shard, image_embedder, text_embedder = load_resources()

st.title("ShelfScan Grocery Search")
st.markdown("<p class='muted'>Search the local qdrant-edge catalog with text or an image.</p>", unsafe_allow_html=True)

if shard is None:
    st.error("Index shard not found. Run `python index.py` first.")
    st.stop()

mode = st.segmented_control("Search mode", ["Text", "Image"], default="Text")
results = None
latency_ms = None

if mode == "Text":
    query = st.text_input("Query", placeholder="Try: spicy snack, dairy butter, orange drink")
    search_requested = False

    cols = st.columns(3)
    for index, sample in enumerate(SAMPLE_QUERIES):
        if cols[index % 3].button(sample, key=f"sample_query_{index}", width="stretch"):
            query = sample
            search_requested = True

    if st.button("Search", type="primary", disabled=not query) or search_requested:
        start = time.time()
        results = search_text(query)
        latency_ms = (time.time() - start) * 1000
else:
    uploaded = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png"])
    selected_image = None
    search_requested = False

    if uploaded:
        selected_image = Image.open(uploaded).convert("RGB")
        st.image(selected_image, width=180)

    st.markdown("##### Samples")
    cols = st.columns(len(SAMPLE_IMAGES))
    for index, (directory, label) in enumerate(SAMPLE_IMAGES):
        path = next((IMAGES_DIR / directory).glob("*.jpg"), None)
        if not path:
            continue
        with cols[index]:
            st.image(Image.open(path), width="stretch")
            if st.button(label, key=f"sample_image_{index}", width="stretch"):
                selected_image = path
                search_requested = True

    if st.button("Search image", type="primary", disabled=selected_image is None) or search_requested:
        start = time.time()
        results = search_image(selected_image)
        latency_ms = (time.time() - start) * 1000

st.divider()

if results is not None:
    metric_cols = st.columns(2)
    metric_cols[0].metric("top_k results", len(results))
    metric_cols[1].metric("Latency", f"{latency_ms:.1f} ms")

    if results:
        render_grid(results, catalog_lookup)
    else:
        st.info("No matching products found.")
else:
    st.subheader("Catalog")
    render_grid(catalog, catalog_lookup)