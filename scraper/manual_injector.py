# scraper/manual_injector.py

import os
import json
import re
from scraper.utils import logger

MANUAL_DIR = "data/manual"
CHUNKS_FILE = "data/chunks/chunks.json"
FILTERED_CHUNKS_FILE = "data/embeddings/chunks_filtered.json"

MIN_TOKENS = 30


def count_tokens_simple(text: str) -> int:
    """Rough token count — good enough for manual files."""
    return len(text.split())


def make_chunk_id_manual(filename: str, index: int) -> str:
    import hashlib
    raw = f"manual_{filename}__chunk_{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_manual_text(text: str, chunk_size_words: int = 300, overlap: int = 30) -> list[str]:
    """Simple sentence-aware chunking for manual files."""
    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = []
    current_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_count + word_count > chunk_size_words and current:
            chunks.append(" ".join(current))
            # overlap: keep last few sentences
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current):
                overlap_count += len(s.split())
                if overlap_count > overlap:
                    break
                overlap_sentences.insert(0, s)
            current = overlap_sentences + [sentence]
            current_count = sum(len(s.split()) for s in current)
        else:
            current.append(sentence)
            current_count += word_count

    if current:
        chunks.append(" ".join(current))

    return chunks


def parse_manual_file(filepath: str) -> dict | None:
    """
    Parse a manual .txt file.
    Tries to extract a title from the first non-empty line.
    Handles both our formatted files (with SOURCE_URL header)
    and raw professor files (plain text).
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    if not content:
        return None

    filename = os.path.basename(filepath)

    # Check if it's our formatted file (has SOURCE_URL header)
    if "SOURCE_URL:" in content and "-" * 20 in content:
        parts = content.split("-" * 20, maxsplit=1)
        if len(parts) == 2:
            header, body = parts
            metadata = {}
            for line in header.strip().splitlines():
                if ": " in line:
                    key, value = line.split(": ", maxsplit=1)
                    metadata[key.strip()] = value.strip()
            return {
                "source_url": metadata.get("SOURCE_URL", "https://www.ku.de"),
                "title": metadata.get("TITLE", filename),
                "source_file": filename,
                "text": body.strip()
            }

    # Raw professor file — extract title from first line
    lines = content.splitlines()
    title = lines[0].strip() if lines else filename

    return {
        "source_url": "https://www.ku.de/en/mgf",
        "title": title,
        "source_file": filename,
        "text": content
    }


def inject_manual_data():
    """
    Load all .txt files from data/manual/, chunk them,
    and add to chunks_filtered.json (creating it if needed).
    Also updates chunks.json for consistency.
    Skips chunks that already exist by chunk_id.
    """
    if not os.path.exists(MANUAL_DIR):
        logger.info("No data/manual/ folder found — skipping.")
        return

    manual_files = [f for f in os.listdir(MANUAL_DIR) if f.endswith(".txt")]
    if not manual_files:
        logger.info("No manual .txt files found — skipping.")
        return

    # Load existing filtered chunks if available
    if os.path.exists(FILTERED_CHUNKS_FILE):
        with open(FILTERED_CHUNKS_FILE, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
    elif os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
    else:
        existing_chunks = []

    existing_ids = {c["chunk_id"] for c in existing_chunks}
    new_chunks = []

    for filename in sorted(manual_files):
        filepath = os.path.join(MANUAL_DIR, filename)
        parsed = parse_manual_file(filepath)

        if not parsed:
            logger.warning(f"Could not parse: {filename}")
            continue

        text_chunks = chunk_manual_text(parsed["text"])

        for i, chunk_text in enumerate(text_chunks):
            token_count = count_tokens_simple(chunk_text)
            if token_count < MIN_TOKENS:
                continue

            chunk_id = make_chunk_id_manual(filename, i)
            if chunk_id in existing_ids:
                continue

            new_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source_url": parsed["source_url"],
                "title": parsed["title"],
                "source_file": parsed["source_file"],
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "token_count": token_count
            })
            existing_ids.add(chunk_id)

    if not new_chunks:
        logger.info("No new manual chunks to inject (already up to date).")
        return

    all_chunks = existing_chunks + new_chunks

    # Save to both chunk files
    os.makedirs(os.path.dirname(FILTERED_CHUNKS_FILE), exist_ok=True)
    with open(FILTERED_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(CHUNKS_FILE), exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Injected {len(new_chunks)} manual chunks from {len(manual_files)} files.")
    logger.info(f"Total chunks now: {len(all_chunks)}")
