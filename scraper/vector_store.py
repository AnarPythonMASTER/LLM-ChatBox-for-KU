# scraper/vector_store.py

import os
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from scraper.utils import logger

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
EMBEDDINGS_FILE = "data/embeddings/embeddings.npy"
CHUNKS_FILE = "data/embeddings/chunks_filtered.json"
CHROMA_DIR = "data/chromadb"
COLLECTION_NAME = "ku_faculty"

# How many chunks to insert into ChromaDB at once
BATCH_SIZE = 500
# ──────────────────────────────────────────────


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Create a persistent ChromaDB client.
    Data is saved to disk at CHROMA_DIR so it
    survives between runs — no need to rebuild the index every time.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def build_vector_store():
    """
    Load embeddings + chunks and insert everything into ChromaDB.
    Builds the HNSW index automatically on insertion.
    """
    # Load embeddings matrix
    logger.info(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    embeddings = np.load(EMBEDDINGS_FILE)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Load chunk metadata
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    assert len(chunks) == len(embeddings), (
        f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings. "
        f"Something broke the index alignment."
    )

    # Connect to ChromaDB
    client = get_chroma_client()

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # use cosine similarity
    )

    logger.info(f"Inserting {len(chunks)} chunks into ChromaDB...")

    # Insert in batches
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Building index"):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_embeddings = embeddings[i : i + BATCH_SIZE]

        collection.add(
            ids=[c["chunk_id"] for c in batch_chunks],
            embeddings=batch_embeddings.tolist(),
            documents=[c["text"] for c in batch_chunks],
            metadatas=[
                {
                    "source_url": c["source_url"],
                    "title": c["title"],
                    "source_file": c["source_file"],
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                    "token_count": c["token_count"]
                }
                for c in batch_chunks
            ]
        )

    count = collection.count()
    logger.info(f"Vector store built. Total vectors indexed: {count}")
    logger.info(f"ChromaDB saved to: {CHROMA_DIR}")


def query_vector_store(
    query_text: str,
    n_results: int = 5,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> list[dict]:
    """
    Embed a query and retrieve the top-n most relevant chunks.
    Returns a list of result dicts with text and metadata.

    This is the function the LLM layer will call in Milestone 5.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    query_embedding = model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()

    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Format results cleanly
    formatted = []
    for i in range(len(results["documents"][0])):
        formatted.append({
            "text": results["documents"][0][i],
            "source_url": results["metadatas"][0][i]["source_url"],
            "title": results["metadatas"][0][i]["title"],
            "score": 1 - results["distances"][0][i],  # convert distance to similarity
            "chunk_index": results["metadatas"][0][i]["chunk_index"]
        })

    return formatted


def test_retrieval():
    """
    Run a few test queries so we can verify retrieval is working
    before connecting the LLM in Milestone 5.
    """
    test_queries = [
        "What are the exam registration deadlines?",
        "Wie bewerbe ich mich für ein Masterstudium?",
        "Data Science Studiengang Module",
        "Prüfungsordnung Bachelor"
    ]

    logger.info("\n" + "=" * 50)
    logger.info("RETRIEVAL TEST")
    logger.info("=" * 50)

    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        logger.info("-" * 40)
        results = query_vector_store(query, n_results=3)
        for r in results:
            logger.info(
                f"  Score: {r['score']:.3f} | "
                f"{r['title'][:50]} | "
                f"chunk {r['chunk_index']}"
            )
            logger.info(f"  Preview: {r['text'][:120]}...")
