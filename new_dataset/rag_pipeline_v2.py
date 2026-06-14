"""
rag_pipeline_v2.py
Step 3: clean baseline RAG pipeline for the new delimiter-chunk dataset.

Deliberately simple: semantic retrieval -> light keyword rerank ->
score threshold -> LLM answer. No routing, no person-lookup yet.
We measure this baseline first, then add enhancements and prove each helps.

Reads from the ChromaDB built by build_index.py (collection 'ku_ds_chunks').
"""

import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ── Config ──
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CHROMA_DIR = "vector_db"
COLLECTION_NAME = "ku_ds_chunks"

N_RETRIEVE = 10
N_FINAL = 5
MIN_SCORE = 0.25
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 500

SYSTEM_PROMPT = """You are a helpful academic assistant for the Katholische Universität Eichstätt-Ingolstadt (KU).

Answer questions from students about study programmes, modules, examination rules, deadlines, application procedures, professors, and faculty information.

CRITICAL RULES — follow these exactly:
1. Answer ONLY using the context provided. Do not use outside knowledge.
2. If the answer is not clearly in the context, say exactly: "I don't have that information in my knowledge base. Please check www.ku.de or contact the faculty office."
3. Always answer in English.
4. Be concise. Use bullet points when listing multiple items.
5. Never invent names, deadlines, or requirements.
6. Note: the Mathematics programme is different from the Data Science programme; do not confuse them."""


def clean_query(query: str) -> str:
    q = query.lower().strip()
    q = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß\s]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q


def keyword_score(query: str, text: str) -> int:
    q_tokens = set(query.lower().split())
    t_tokens = set(text.lower().split())
    return len(q_tokens.intersection(t_tokens))


class RAGPipeline:
    def __init__(self, llm_enabled=True):
        print("Initialising RAG pipeline (v2, baseline)...")
        print(f"Embedding model: {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)
        print(f"ChromaDB connected — {self.collection.count()} chunks")

        self.llm_enabled = llm_enabled
        if llm_enabled:
            hf_token = os.environ.get("HF_TOKEN", "")
            self.llm_client = InferenceClient(token=hf_token)
            print(f"LLM: {LLM_MODEL}")
        print("Pipeline ready.\n")

    def retrieve(self, query: str) -> list[dict]:
        cleaned = clean_query(query)
        embedding = self.embed_model.encode(
            cleaned, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=N_RETRIEVE,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["documents"][0])):
            meta = results["metadatas"][0][i]
            chunks.append({
                "text": results["documents"][0][i],
                "type": meta.get("type", ""),
                "name": meta.get("name", ""),
                "programme": meta.get("programme", ""),
                "source_file": meta.get("source_file", ""),
                "score": round(1 - results["distances"][0][i], 3),
            })
        return chunks

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        for chunk in chunks:
            kw = keyword_score(query, chunk["text"])
            chunk["combined_score"] = chunk["score"] + (kw * 0.02)
        return sorted(chunks, key=lambda x: x["combined_score"], reverse=True)

    def build_prompt(self, query: str, chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            label = c["name"] or c["type"]
            parts.append(f"[Source {i}: {label}]\n{c['text']}")
        context = "\n\n---\n\n".join(parts)
        return f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER (in English only):"

    def generate(self, prompt: str) -> str:
        try:
            response = self.llm_client.chat_completion(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def ask(self, query: str) -> dict:
        chunks = self.retrieve(query)
        ranked = self.rerank(query, chunks)
        strong = [c for c in ranked if c["combined_score"] >= MIN_SCORE][:N_FINAL]

        if not strong:
            # fall back to top 3 so we can see what was close
            strong = ranked[:3]

        result = {
            "query": query,
            "retrieved": [
                {"name": c["name"], "type": c["type"],
                 "score": round(c["combined_score"], 3)}
                for c in strong
            ],
        }

        if self.llm_enabled:
            prompt = self.build_prompt(query, strong)
            result["answer"] = self.generate(prompt)
        else:
            result["answer"] = None

        return result
