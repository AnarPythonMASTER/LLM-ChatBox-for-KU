"""
chunk_parser.py
Parses our delimiter-based data files into structured chunks.

File format expected:
    SOURCE_URL: ...
    TITLE: ...
    SOURCE_FILE: ...
    ------------------------------------------------------------

    ===CHUNK===
    type: <type>
    name: <name>
    programme: <programme>
    area: <area>            (optional)
    ---
    <body text, one self-contained paragraph>

    ===CHUNK===
    ...

Each ===CHUNK=== block becomes ONE chunk = ONE embedding, with its
metadata preserved for later filtering.
"""

import os

CHUNK_DELIMITER = "===CHUNK==="
META_BODY_SEPARATOR = "---"


def parse_file(filepath):
    """Parse a single delimiter-format file into a list of chunk dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(filepath)

    # Split on the chunk delimiter. Everything before the first
    # delimiter is the file-level header, which we discard for chunks
    # (but could capture if needed).
    parts = content.split(CHUNK_DELIMITER)
    raw_blocks = parts[1:]  # skip the header part

    chunks = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        # Separate metadata (above ---) from body (below ---)
        if META_BODY_SEPARATOR in block:
            meta_part, body_part = block.split(META_BODY_SEPARATOR, 1)
        else:
            # Fallback: no separator -> treat whole block as body
            meta_part, body_part = "", block

        # Parse metadata key: value lines
        metadata = {}
        for line in meta_part.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip().lower()] = value.strip()

        body = body_part.strip()
        if not body:
            continue  # skip empty bodies

        chunks.append({
            "text": body,
            "type": metadata.get("type", "unknown"),
            "name": metadata.get("name", ""),
            "programme": metadata.get("programme", "general"),
            "area": metadata.get("area", ""),
            "source_file": filename,
        })

    return chunks


def parse_directory(folder_path):
    """Parse all .txt files in a folder into a combined list of chunks."""
    all_chunks = []
    files = sorted(os.listdir(folder_path))
    for filename in files:
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(folder_path, filename)
        file_chunks = parse_file(filepath)
        all_chunks.extend(file_chunks)
        print(f"  {len(file_chunks):3d} chunks  <-  {filename}")
    return all_chunks


if __name__ == "__main__":
    import sys
    from collections import Counter

    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Parsing files in: {folder}\n")
    chunks = parse_directory(folder)

    print(f"\nTotal chunks parsed: {len(chunks)}")

    # Sanity checks
    print("\nBy type:")
    for t, c in Counter(ch["type"] for ch in chunks).most_common():
        print(f"  {c:3d}  {t}")

    print("\nBy programme:")
    for p, c in Counter(ch["programme"] for ch in chunks).most_common():
        print(f"  {c:3d}  {p}")

    # Flag any chunks missing critical metadata
    missing_type = [ch for ch in chunks if ch["type"] == "unknown"]
    missing_name = [ch for ch in chunks if not ch["name"]]
    empty_body = [ch for ch in chunks if len(ch["text"]) < 20]

    print(f"\nQuality checks:")
    print(f"  chunks missing type:  {len(missing_type)}")
    print(f"  chunks missing name:  {len(missing_name)}")
    print(f"  suspiciously short bodies (<20 chars): {len(empty_body)}")

    if missing_type:
        print("\n  Files with 'unknown' type chunks:")
        for f, c in Counter(ch["source_file"] for ch in missing_type).most_common():
            print(f"    {c:3d}  {f}")
