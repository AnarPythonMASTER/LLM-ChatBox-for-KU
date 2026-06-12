# scraper/rag_pipeline.py

import os
import re
import requests
from sentence_transformers import SentenceTransformer
from scraper.vector_store import get_chroma_client
from scraper.utils import logger

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
OLLAMA_MODEL = "gemma3:1b"          # change to "llama3" if prefered
OLLAMA_URL = "http://localhost:11434/api/generate"

COLLECTION_NAME = "ku_faculty"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

N_RETRIEVE = 10    # how many chunks to fetch from ChromaDB
N_FINAL = 5        # how many to keep after reranking
TEMPERATURE = 0.0  # 0 = fully deterministic, factual
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful academic assistant for the Katholische Universität Eichstätt-Ingolstadt (KU).

Your job is to answer questions from students and staff about:
- Study programmes and modules
- Examination rules and deadlines
- Application and enrollment procedures
- Faculty information and contacts
- Campus life and services

Rules you must follow:
1. Answer ONLY using the context provided. Do not use outside knowledge.
2. If the answer is not in the context, say exactly: "I don't have that information in my knowledge base. Please check www.ku.de directly."
3. Answer in the same language the user asks in. German question → German answer. English → English.
4. Be concise. Students need clear answers, not long essays.
5. When helpful, use bullet points.
6. Never invent deadlines, names, rules, or requirements.
7. If the context contains a relevant source title, mention it so the user can verify."""


# ──────────────────────────────────────────────
# QUERY CLEANING
# Borrowed from your friend's approach —
# normalising before embedding reduces noise
# ──────────────────────────────────────────────
def clean_query(query: str) -> str:
    """
    Normalise query before embedding:
    - lowercase
    - remove punctuation except German characters
    - collapse whitespace
    """
    q = query.lower().strip()
    q = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß\s]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q


# ──────────────────────────────────────────────
# KEYWORD RERANKING
# After vector retrieval gives us N_RETRIEVE chunks,
# we re-score them by keyword overlap with the query.
# This is a cheap second pass that pushes chunks with
# exact term matches higher — helpful when the query
# contains specific words like "Data Science" or "APO".
# ──────────────────────────────────────────────
def keyword_score(query: str, chunk_text: str) -> int:
    """Count how many query words appear in the chunk."""
    query_tokens = set(query.lower().split())
    chunk_tokens = set(chunk_text.lower().split())
    return len(query_tokens.intersection(chunk_tokens))


def rerank_chunks(query: str, chunks: list[dict], final_k: int) -> list[dict]:
    """
    Re-score retrieved chunks by keyword overlap and return top final_k.
    Each chunk gets a combined score: vector similarity + keyword bonus.
    """
    for chunk in chunks:
        kw_score = keyword_score(query, chunk["text"])
        # Blend: vector score (0-1) + small keyword bonus
        # We scale keyword score to not dominate vector similarity
        chunk["combined_score"] = chunk["score"] + (kw_score * 0.02)

    reranked = sorted(chunks, key=lambda x: x["combined_score"], reverse=True)
    return reranked[:final_k]


class RAGPipeline:
    """
    Full local RAG pipeline:
    - Embedding model loaded once at startup
    - ChromaDB connection held open
    - Per query: clean → embed → retrieve → rerank → prompt → Ollama → return
    """

    def __init__(self):
        logger.info("Initialising RAG pipeline...")

        # Load embedding model ONCE — reused for every query
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        # Connect to ChromaDB ONCE
        client = get_chroma_client()
        self.collection = client.get_collection(COLLECTION_NAME)
        count = self.collection.count()
        logger.info(f"ChromaDB connected — {count} vectors indexed")
        logger.info(f"Ollama model: {OLLAMA_MODEL}")
        logger.info("RAG pipeline ready.\n")

    def retrieve(self, query: str) -> list[dict]:
        """
        Step 1 — Vector retrieval.
        Clean query, embed it, fetch top N_RETRIEVE chunks from ChromaDB.
        """
        cleaned = clean_query(query)

        query_embedding = self.embed_model.encode(
            cleaned,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RETRIEVE,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "title": results["metadatas"][0][i]["title"],
                "source_url": results["metadatas"][0][i]["source_url"],
                "score": round(1 - results["distances"][0][i], 3)
            })

        return chunks

    def build_prompt(self, query: str, chunks: list[dict]) -> str:
        """
        Step 3 — Assemble context + question into the final prompt.
        Each chunk is numbered and labelled with its source.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['title']}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )

    def generate(self, prompt: str) -> str:
        """
        Step 4 — Send prompt to local Ollama and return the response.
        Handles the case where Ollama isn't running with a clear message.
        """
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": TEMPERATURE,
                        "num_predict": 500
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"].strip()

        except requests.exceptions.ConnectionError:
            return (
                "Ollama is not running. "
                "Please start it with: ollama serve"
            )
        except requests.exceptions.Timeout:
            return "Request timed out. Ollama may be overloaded — try again."
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def ask(self, query: str, verbose: bool = False) -> dict:
        """
        Full RAG query — the single method you call from anywhere.

        Steps:
        1. Retrieve top N_RETRIEVE chunks by vector similarity
        2. Rerank by keyword overlap, keep top N_FINAL
        3. Build prompt with context
        4. Generate answer via Ollama
        5. Return structured result

        Returns dict with: query, answer, sources
        """
        # Step 1: Retrieve
        chunks = self.retrieve(query)

        # Step 2: Rerank
        chunks = rerank_chunks(query, chunks, final_k=N_FINAL)

        if verbose:
            logger.info(f"Top chunks after reranking:")
            for c in chunks:
                logger.info(
                    f"  [{c['combined_score']:.3f}] "
                    f"{c['title'][:50]} — "
                    f"{c['text'][:80]}..."
                )

        # Step 3: Build prompt
        prompt = self.build_prompt(query, chunks)

        # Step 4: Generate
        answer = self.generate(prompt)

        # Step 5: Format sources
        sources = [
            {
                "title": c["title"],
                "url": c["source_url"],
                "score": c["combined_score"]
            }
            for c in chunks
        ]

        return {
            "query": query,
            "answer": answer,
            "sources": sources
        }


def test_rag_pipeline():
    """End-to-end test with a few representative questions."""
    pipeline = RAGPipeline()

    questions = [
        "Wie bewerbe ich mich für den Master Data Science?",
        "What are the examination regulations for Bachelor students?",
        "Welche Module gibt es im Studiengang Data Science?",
    ]

    print("\n" + "=" * 60)
    print("RAG PIPELINE TEST")
    print("=" * 60)

    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 60)
        result = pipeline.ask(question, verbose=True)
        print(f"A: {result['answer']}")
        print("\nSources used:")
        for s in result["sources"][:3]:
            print(f"  [{s['score']:.3f}] {s['title']}")
        print()
