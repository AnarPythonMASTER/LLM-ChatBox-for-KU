"""
build_index.py
Step 2 of the pipeline: parse our delimiter-format files, embed each
chunk body, and store everything in ChromaDB with full metadata.

Design decisions (Target B):
  - We split on ===CHUNK===, NOT on word count. Each clean chunk = one
    embedding. This preserves the semantic chunking we built by hand.
  - Embedding model: all-MiniLM-L6-v2 (English-only, 384-dim). Our data
    is all English, so an English-specialized model gives sharper
    embeddings than a multilingual one, and it's faster/lighter.
  - We embed the BODY ONLY for now (clean baseline). All metadata
    (type, name, programme, area, source_file) is stored alongside so
    we can add metadata filtering or name-prepending later WITHOUT
    re-embedding.
"""

import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer

from chunk_parser import parse_directory

# ---- Configuration ----
DATA_DIR = "."          # <-- adjust to where your clean .txt files live
CHROMA_DIR = "vector_db"
COLLECTION_NAME = "ku_ds_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def reset_vector_db():
    """Delete any existing database so we always rebuild from scratch."""
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"Removed existing database at {CHROMA_DIR}")


def main():
    print("=== Building vector index ===\n")

    reset_vector_db()

    # ---- Step 2a: parse all chunks ----
    print(f"Parsing chunks from: {DATA_DIR}")
    chunks = parse_directory(DATA_DIR)
    print(f"\nParsed {len(chunks)} chunks total.\n")

    if not chunks:
        print("ERROR: no chunks parsed. Check DATA_DIR path and file format.")
        return

    # ---- Step 2b: load embedding model ----
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ---- Step 2c: embed the bodies ----
    # NAME-PREPENDING: front-load the chunk name into the embedded text so
    # named-entity queries ("who is Marcel Oliver", "semester 1 courses")
    # match the right chunk. The stored DOCUMENT stays the clean body; only
    # the text we EMBED gets the name prefix.
    PREPEND_NAME = True

    def embed_text(ch):
        if PREPEND_NAME and ch["name"]:
            return f"{ch['name']}. {ch['text']}"
        return ch["text"]

    texts = [ch["text"] for ch in chunks]          # stored documents (clean body)
    embed_inputs = [embed_text(ch) for ch in chunks]  # what we actually embed
    print(f"Embedding {len(embed_inputs)} chunks (name-prepended)...")
    embeddings = model.encode(
        embed_inputs,
        normalize_embeddings=True,     # cosine similarity ready
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
    ).tolist()

    # ---- Step 2d: build ChromaDB ----
    print("\nStoring in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # cosine space to match our normalized embeddings
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    metadatas = []
    for i, ch in enumerate(chunks):
        # unique, readable id: sourcefile + index
        safe_source = ch["source_file"].replace(".txt", "")
        ids.append(f"{safe_source}__{i}")
        metadatas.append({
            "type": ch["type"],
            "name": ch["name"],
            "programme": ch["programme"],
            "area": ch["area"],
            "source_file": ch["source_file"],
        })

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"\nDone. Stored {collection.count()} chunks in collection "
          f"'{COLLECTION_NAME}' at '{CHROMA_DIR}'.")

    # ---- quick verification query ----
    print("\n--- Sanity query: 'who is professor oliver' ---")
    q_emb = model.encode("who is professor oliver",
                         normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=3,
                           include=["documents", "metadatas", "distances"])
    for doc, meta, dist in zip(res["documents"][0],
                               res["metadatas"][0],
                               res["distances"][0]):
        score = 1 - dist
        print(f"  [{score:.3f}] ({meta['type']}) {meta['name']}")


if __name__ == "__main__":
    main()
