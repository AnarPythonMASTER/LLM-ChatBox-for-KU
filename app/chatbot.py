# import chromadb
# from sentence_transformers import SentenceTransformer

# CHROMA_DIR = "vector_db"
# COLLECTION_NAME = "ku_mids_docs"

# model = SentenceTransformer("all-MiniLM-L6-v2")

# client = chromadb.PersistentClient(path=CHROMA_DIR)

# collection = client.get_collection(
#     name=COLLECTION_NAME
# )


# def search(question, n_results=3):
#     query_embedding = model.encode(
#         [question]
#     ).tolist()

#     results = collection.query(
#         query_embeddings=query_embedding,
#         n_results=n_results
#     )

#     return results


# if __name__ == "__main__":

#     while True:
#         question = input("\nAsk a question: ")

#         if question.lower() == "exit":
#             break

#         results = search(question)

#         print("\nTop Results:\n")

#         for i, document in enumerate(
#             results["documents"][0]
#         ):
#             print("=" * 80)
#             print(f"Result {i+1}")
#             print(document[:1000])
#             print()


# import chromadb
# import requests
# from sentence_transformers import SentenceTransformer


# CHROMA_DIR = "vector_db"
# COLLECTION_NAME = "ku_mids_docs"

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# OLLAMA_MODEL = "gemma3:1b"


# model = SentenceTransformer(
#     EMBEDDING_MODEL
# )

# client = chromadb.PersistentClient(
#     path=CHROMA_DIR
# )

# collection = client.get_collection(
#     name=COLLECTION_NAME
# )


# def retrieve_context(
#     question,
#     n_results=4
# ):
#     query_embedding = model.encode(
#         [question]
#     ).tolist()

#     results = collection.query(
#         query_embeddings=query_embedding,
#         n_results=n_results
#     )

#     return results["documents"][0]


# def ask_llm(
#     question,
#     context_chunks
# ):
#     context = "\n\n".join(
#         context_chunks
#     )

#     prompt = f"""
# You are the KU MIDS Assistant.

# Answer ONLY using the provided context.

# If the answer is not contained in the context,
# say:

# 'I could not find this information in the KU MIDS knowledge base.'

# Context:

# {context}

# Question:

# {question}

# Answer:
# """

#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": OLLAMA_MODEL,
#             "prompt": prompt,
#             "stream": False
#         }
#     )

#     return response.json()["response"]


# def answer_question(question):
#     chunks = retrieve_context(
#         question
#     )

#     answer = ask_llm(
#         question,
#         chunks
#     )

#     return answer


# if __name__ == "__main__":

#     while True:

#         question = input(
#             "\nAsk a question: "
#         )

#         if question.lower() == "exit":
#             break

#         answer = answer_question(
#             question
#         )

#         print("\n")
#         print(answer)





import chromadb
import requests
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "ku_mids_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:1b"

model = SentenceTransformer(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME)


def retrieve_context(question, n_results=2):
    query_embedding = model.encode([question]).tolist()

    question_lower = question.lower()

    if any(name in question_lower for name in [
        "stöger", "stoeger", "dominik", "oliver", "marcel",
        "pfander", "fontaine", "voigtlaender", "voigtländer",
        "janjic", "ray", "setzer"
    ]):
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where={"source": "professors.txt"}
        )
    else:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for doc, meta in zip(documents, metadatas):
        context_parts.append(
            f"Source: {meta['source']}\n{doc}"
        )

    return "\n\n".join(context_parts)


def ask_llm(question, context):
    prompt = f"""
You are the KU MIDS Assistant.

Answer the question using ONLY the context below.
Be concise and precise.
If the answer is not in the context, say:
"I could not find this information in the KU MIDS knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 120
            },
            "keep_alive": "10m"
        }
    )

    return response.json()["response"]


def answer_question(question):
    context = retrieve_context(question)
    return ask_llm(question, context)


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            break

        answer = answer_question(question)
        print("\n" + answer)