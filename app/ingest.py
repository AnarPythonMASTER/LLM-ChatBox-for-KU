import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer


RAW_DATA_DIR = "data/raw"
CHROMA_DIR = "vector_db"
COLLECTION_NAME = "ku_mids_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def read_txt_files(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append({
                "filename": filename,
                "text": text
            })

    return documents


def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def reset_vector_db():
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)


def main():
    print("Starting ingestion...")

    reset_vector_db()

    documents = read_txt_files(RAW_DATA_DIR)

    print(f"Found {len(documents)} text files.")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []

    chunk_id = 0

    for document in documents:
        filename = document["filename"]
        text = document["text"]

        chunks = chunk_text(text)

        print(f"{filename}: created {len(chunks)} chunks.")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}_chunk_{i}")
            all_metadatas.append({
                "source": filename,
                "chunk_number": i
            })

            chunk_id += 1

    print(f"Total chunks: {len(all_chunks)}")
    print("Creating embeddings...")

    embeddings = embedding_model.encode(all_chunks).tolist()

    print("Storing chunks in ChromaDB...")

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids
    )

    print("Ingestion completed successfully.")
    print(f"Vector database saved in: {CHROMA_DIR}")


if __name__ == "__main__":
    main()