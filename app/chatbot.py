import chromadb
import requests
from sentence_transformers import SentenceTransformer
import re

# ----------------------------
# CONFIG
# ----------------------------
CHROMA_PATH = "vector_db"
COLLECTION_NAME = "ku_mids_docs"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:1b"

TOP_K = 10
FINAL_K = 5  # after filtering


client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# ----------------------------
# QUERY CLEANING (IMPORTANT)
# ----------------------------
def clean_query(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^a-zA-Z0-9äöüß\s]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q


# ----------------------------
# SIMPLE KEYWORD BOOST (cheap reranking)
# ----------------------------
def score_chunk(query, text):
    q_tokens = set(query.split())
    t_tokens = set(text.lower().split())
    return len(q_tokens.intersection(t_tokens))


# ----------------------------
# OPTIONAL: RULE-BASED ROUTING (keep yours, but simplified fallback safe)
# ----------------------------
def detect_topic(question):
    q = question.lower()

    if any(w in q for w in ["fontaine", "pirmin", "logistics"]):
        return "prof_fontaine.txt"

    if any(w in q for w in ["janjic", "tijana"]):
        return "prof_janjic.txt"

    if any(w in q for w in ["pfander", "götz", "goetz"]):
        return "prof_pfander.txt"

    if any(w in q for w in ["voigtlaender", "felix"]):
        return "prof_voigtlaender.txt"

    if any(w in q for w in ["bachelor", "master", "mids", "data science"]):
        return None  # IMPORTANT: let vector DB handle general cases

    return None


# ----------------------------
# RETRIEVAL (IMPROVED)
# ----------------------------
def retrieve_context(question, n_results=TOP_K):
    query_clean = clean_query(question)

    query_embedding = embedding_model.encode(query_clean).tolist()

    topic = detect_topic(question)
    where_filter = {"source": topic} if topic else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # ----------------------------
    # RERANKING STEP (VERY IMPORTANT)
    # ----------------------------
    scored_chunks = []

    for doc, meta in zip(documents, metadatas):
        score = score_chunk(query_clean, doc)
        scored_chunks.append((score, doc, meta))

    # sort best chunks first
    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    # take top FINAL_K
    top_chunks = scored_chunks[:FINAL_K]

    context_parts = []
    sources_used = set()

    for score, doc, meta in top_chunks:
        source = meta.get("source", "unknown")
        sources_used.add(source)

        context_parts.append(
            f"[Source: {source} | relevance={score}]\n{doc}"
        )

    return "\n\n".join(context_parts), list(sources_used)


# ----------------------------
# LLM CALL (IMPROVED PROMPT)
# ----------------------------
def ask_ollama(question, context):

    prompt = f"""
You are a strict retrieval-based assistant for KU Eichstätt-Ingolstadt (MIDS).

You MUST follow these rules:

RULES:
- Use ONLY the provided context.
- If the answer is not in the context, say:
  "I do not have enough information in my current knowledge base."
- Do NOT guess.
- Do NOT use external knowledge.
- Keep answers short and precise.
- Prefer bullet points when helpful.

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 200
                }
            },
            timeout=60
        )

        response.raise_for_status()
        return response.json()["response"]

    except requests.exceptions.ConnectionError:
        return (
            "❌ Ollama is not running.\n"
            "Please start it with: `ollama serve`"
        )

    except Exception as e:
        return f"Error: {str(e)}"


# ----------------------------
# CLI TEST LOOP
# ----------------------------
if __name__ == "__main__":
    while True:
        question = input("\nAsk a question: ")

        if question.lower() in ["exit", "quit", "q"]:
            break

        context, sources = retrieve_context(question)

        print("\nSources used:")
        for s in sources:
            print("-", s)

        answer = ask_ollama(question, context)

        print("\n" + answer)