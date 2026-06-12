# scraper/embedder.py

import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from scraper.utils import logger

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
CHUNKS_FILE = "data/chunks/chunks.json"
EMBEDDINGS_DIR = "data/embeddings"
EMBEDDINGS_FILE = "data/embeddings/embeddings.npy"
CHUNKS_FILTERED_FILE = "data/embeddings/chunks_filtered.json"

# Multilingual model — handles German and English well
# 384-dimensional output vectors
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Drop chunks shorter than this — they're navigation fragments
MIN_TOKENS = 30

# How many chunks to embed in one batch
# Higher = faster but more RAM. 64 is safe for most laptops.
BATCH_SIZE = 64
# ──────────────────────────────────────────────


def load_and_filter_chunks() -> list[dict]:
    """
    Load chunks.json and drop anything too short to be useful.
    """
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    before = len(chunks)
    chunks = [c for c in chunks if c["token_count"] >= MIN_TOKENS]
    after = len(chunks)

    logger.info(f"Chunks loaded: {before}")
    logger.info(f"Chunks after filtering (>= {MIN_TOKENS} tokens): {after}")
    logger.info(f"Dropped: {before - after} fragments")

    return chunks


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Embed all chunks using the sentence transformer model.
    Returns a numpy array of shape (num_chunks, embedding_dim).

    Processes in batches so RAM doesn't spike.
    """
    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    texts = [chunk["text"] for chunk in chunks]
    all_embeddings = []

    logger.info(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
        batch = texts[i : i + BATCH_SIZE]
        batch_embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True   # normalize to unit length for cosine similarity
        )
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings(chunks: list[dict], embeddings: np.ndarray):
    """
    Save embeddings as a .npy file (fast binary format).
    Save filtered chunks as JSON (keeps metadata aligned with embeddings).

    IMPORTANT: chunk at index i corresponds to embeddings[i].
    This alignment must never be broken.
    """
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Save embeddings matrix
    np.save(EMBEDDINGS_FILE, embeddings)
    logger.info(f"Embeddings saved to: {EMBEDDINGS_FILE}")

    # Save the filtered chunks (metadata)
    with open(CHUNKS_FILTERED_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Filtered chunks saved to: {CHUNKS_FILTERED_FILE}")


def run_embeddings():
    """Full embeddings pipeline."""
    chunks = load_and_filter_chunks()
    embeddings = embed_chunks(chunks)
    save_embeddings(chunks, embeddings)
    logger.info("✓ Milestone 3 complete — embeddings ready.")
    return chunks, embeddings
