# scraper/chunker.py

import os
import re
import json
import hashlib
import tiktoken
from tqdm import tqdm

from scraper.utils import logger

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
CLEANED_DIR = "data/cleaned"
CHUNKS_DIR = "data/chunks"
OUTPUT_FILE = "data/chunks/chunks.json"

CHUNK_SIZE_TOKENS = 400
OVERLAP_TOKENS = 50
MIN_TOKENS_TO_SPLIT = 100
# ──────────────────────────────────────────────

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def make_chunk_id(source_file: str, chunk_index: int) -> str:
    raw = f"{source_file}__chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into individual sentences using punctuation boundaries.
    Handles German and English punctuation.
    Keeps the punctuation attached to the sentence it belongs to.
    """
    # Split on . ! ? followed by whitespace and a capital letter,
    # or on newlines which often indicate paragraph/section breaks
    sentence_endings = re.compile(
        r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÄÖÜ])'  # sentence-ending punctuation
        r'|\n{2,}'                            # paragraph breaks
    )
    raw_sentences = sentence_endings.split(text)

    # Clean up and remove empty/whitespace-only sentences
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences


def split_into_chunks(text: str, chunk_size: int, overlap_tokens: int) -> list[str]:
    """
    Sentence-aware chunking:
    1. Split text into complete sentences
    2. Pack sentences greedily into chunks up to chunk_size tokens
    3. When a chunk is full, carry over enough trailing sentences
       to cover overlap_tokens for the next chunk

    This ensures no chunk ever starts or ends mid-sentence.
    """
    if count_tokens(text) < MIN_TOKENS_TO_SPLIT:
        return [text]

    sentences = split_into_sentences(text)
    if not sentences:
        return [text]

    chunks = []
    current_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # If a single sentence is longer than chunk_size
        # (e.g. a very long table row or legal text),
        # just add it as its own chunk — we can't split further
        # without breaking meaning
        if sentence_tokens > chunk_size:
            # Flush current buffer first
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0
            chunks.append(sentence)
            continue

        # If adding this sentence would exceed the limit, flush
        if current_tokens + sentence_tokens > chunk_size:
            chunks.append(" ".join(current_sentences))

            # Build overlap: walk backwards through current_sentences
            # collecting sentences until we hit overlap_tokens
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_tokens = count_tokens(s)
                if overlap_count + s_tokens > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += s_tokens

            current_sentences = overlap_sentences + [sentence]
            current_tokens = sum(count_tokens(s) for s in current_sentences)
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    # Don't forget the last buffer
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def parse_cleaned_file(filepath: str) -> dict | None:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    parts = content.split("-" * 60, maxsplit=1)
    if len(parts) != 2:
        logger.warning(f"Unexpected format, skipping: {filepath}")
        return None

    header_block, text_body = parts
    metadata = {}
    for line in header_block.strip().splitlines():
        if ": " in line:
            key, value = line.split(": ", maxsplit=1)
            metadata[key.strip()] = value.strip()

    text = text_body.strip()
    if not text:
        return None

    return {
        "source_url": metadata.get("SOURCE_URL", ""),
        "title": metadata.get("TITLE", ""),
        "source_file": metadata.get("SOURCE_FILE", os.path.basename(filepath)),
        "text": text
    }


def chunk_all_files() -> list[dict]:
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    cleaned_files = [
        f for f in os.listdir(CLEANED_DIR)
        if f.endswith(".txt")
    ]

    if not cleaned_files:
        logger.warning(f"No cleaned files found in {CLEANED_DIR}.")
        return []

    logger.info(f"Chunking {len(cleaned_files)} cleaned files...")

    all_chunks = []
    total_skipped = 0

    for filename in tqdm(cleaned_files, desc="Chunking files"):
        filepath = os.path.join(CLEANED_DIR, filename)
        parsed = parse_cleaned_file(filepath)

        if parsed is None:
            total_skipped += 1
            continue

        text_chunks = split_into_chunks(
            parsed["text"],
            chunk_size=CHUNK_SIZE_TOKENS,
            overlap_tokens=OVERLAP_TOKENS
        )

        total = len(text_chunks)

        for i, chunk_text in enumerate(text_chunks):
            chunk_obj = {
                "chunk_id": make_chunk_id(parsed["source_file"], i),
                "text": chunk_text,
                "source_url": parsed["source_url"],
                "title": parsed["title"],
                "source_file": parsed["source_file"],
                "chunk_index": i,
                "total_chunks": total,
                "token_count": count_tokens(chunk_text)
            }
            all_chunks.append(chunk_obj)

    logger.info(f"Total chunks produced: {len(all_chunks)}")
    logger.info(f"Files skipped: {total_skipped}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Chunks saved to: {OUTPUT_FILE}")
    return all_chunks


def print_stats(chunks: list[dict]):
    if not chunks:
        return

    token_counts = [c["token_count"] for c in chunks]
    sources = set(c["source_file"] for c in chunks)

    logger.info("─" * 40)
    logger.info("CHUNKING STATS")
    logger.info(f"  Total chunks:      {len(chunks)}")
    logger.info(f"  Unique sources:    {len(sources)}")
    logger.info(f"  Avg tokens/chunk:  {sum(token_counts) / len(token_counts):.0f}")
    logger.info(f"  Min tokens/chunk:  {min(token_counts)}")
    logger.info(f"  Max tokens/chunk:  {max(token_counts)}")
    logger.info(f"  Total tokens:      {sum(token_counts):,}")
    logger.info("─" * 40)
