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





# import chromadb
# import requests
# from sentence_transformers import SentenceTransformer

# CHROMA_DIR = "vector_db"
# COLLECTION_NAME = "ku_mids_docs"
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# OLLAMA_MODEL = "gemma3:1b"

# model = SentenceTransformer(EMBEDDING_MODEL)
# client = chromadb.PersistentClient(path=CHROMA_DIR)
# collection = client.get_collection(name=COLLECTION_NAME)


# def retrieve_context(question, n_results=2):
#     query_embedding = model.encode([question]).tolist()

#     question_lower = question.lower()

#     if any(name in question_lower for name in [
#         "stöger", "stoeger", "dominik", "oliver", "marcel",
#         "pfander", "fontaine", "voigtlaender", "voigtländer",
#         "janjic", "ray", "setzer"
#     ]):
#         results = collection.query(
#             query_embeddings=query_embedding,
#             n_results=n_results,
#             where={"source": "professors.txt"}
#         )
#     else:
#         results = collection.query(
#             query_embeddings=query_embedding,
#             n_results=n_results
#         )

#     documents = results["documents"][0]
#     metadatas = results["metadatas"][0]

#     context_parts = []

#     for doc, meta in zip(documents, metadatas):
#         context_parts.append(
#             f"Source: {meta['source']}\n{doc}"
#         )

#     return "\n\n".join(context_parts)


# def ask_llm(question, context):
#     prompt = f"""
# You are the KU MIDS Assistant.

# Answer the question using ONLY the context below.
# Be concise and precise.
# If the answer is not in the context, say:
# "I could not find this information in the KU MIDS knowledge base."

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
#             "stream": False,
#             "options": {
#                 "temperature": 0.1,
#                 "num_predict": 120
#             },
#             "keep_alive": "10m"
#         }
#     )

#     return response.json()["response"]


# def answer_question(question):
#     context = retrieve_context(question)
#     return ask_llm(question, context)


# if __name__ == "__main__":
#     while True:
#         question = input("\nAsk a question: ")

#         if question.lower() == "exit":
#             break

#         answer = answer_question(question)
#         print("\n" + answer)






import chromadb
import requests
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "ku_mids_docs"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:1b"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# def detect_topic(question):
#     q = question.lower()

#     if any(word in q for word in ["professor", "prof", "stöger", "stoeger", "müller", "mueller"]):
#         return "professors.txt"

#     if any(word in q for word in ["bachelor", "bsc", "undergraduate"]):
#         return "bachelor_data_science.txt"

#     if any(word in q for word in ["master", "msc", "graduate"]):
#         return "master_data_science.txt"

#     if any(word in q for word in ["mids", "mathematical institute", "machine learning and data science"]):
#         return "mids.txt"

#     return None

def detect_topic(question):
    q = question.lower()

    if any(word in q for word in [
        "professor", "prof", "teacher", "team", "chair", "research group",
        "stöger", "stoeger", "pfander", "fontaine", "janjic", "janjić",
        "oliver", "ray", "setzer", "voigtlaender", "voigtländer",
        "götz", "goetz", "pirmin", "dominik", "felix", "tijana"
    ]):
        return "mids_the_team.txt"

    if any(word in q for word in [
        "contact", "email", "phone", "address", "room", "secretary",
        "management", "where is mids", "location", "visiting address",
        "mailing address"
    ]):
        return "mids_contact.txt"

    if any(word in q for word in [
        "job", "jobs", "vacancy", "position", "phd", "student assistant",
        "research assistant", "application", "apply", "salary"
    ]):
        return "mids_jobs.txt"

    if any(word in q for word in [
        "seminar", "talk", "speaker", "wednesday", "georgianum",
        "presentation", "lecture", "sommerfest"
    ]):
        return "mids_seminar.txt"

    if any(word in q for word in [
        "publication", "publications", "paper", "journal", "doi",
        "article", "research output"
    ]):
        return "mids_publications.txt"

    if any(word in q for word in [
        "news", "event", "conference", "new scientific director",
        "half marathon", "data4good", "stukon"
    ]):
        return "mids_news.txt"

    if any(word in q for word in [
        "mids", "about", "institute", "mathematical institute",
        "machine learning and data science", "bachelor", "bsc",
        "master", "msc", "degree program", "data science program"
    ]):
        return "mids_about.txt"

    return None


def retrieve_context(question, n_results=6):
    query_embedding = embedding_model.encode(question).tolist()
    topic = detect_topic(question)

    where_filter = {"source": topic} if topic else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc}")

    return "\n\n".join(context_parts)



def ask_ollama(question, context):
    q = question.lower()

    if "what is mids" in q or "what does mids mean" in q:
        return (
            "MIDS stands for the Mathematical Institute for Machine Learning "
            "and Data Science at KU Eichstätt-Ingolstadt. It is located in "
            "Ingolstadt and focuses on machine learning, data science, "
            "mathematical foundations, and digitalization."
        )



    prompt = f"""
You are a helpful chatbot for KU Eichstätt-Ingolstadt, especially the MIDS / Data Science department.

Important rules:
- Use ONLY the given context.
- MIDS belongs to KU Eichstätt-Ingolstadt. Never call it Ingolstadt University of Technology.
- Do NOT invent university names, locations, professors, admission rules, or program details.
- If the context does not clearly say something, say you do not have enough information.
- Keep answers short, factual, and student-friendly.

Answer the user's question using ONLY the context below.
If the answer is not in the context, say:
"I do not have enough information in my current knowledge base."

Keep the answer clear, concise, and student-friendly.

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
                "temperature": 0.0, #changed 0.1 to 0.0
                "num_predict": 180
            }
        }
    )

    return response.json()["response"]


# while True:
#     question = input("\nAsk a question: ")

#     if question.lower() in ["exit", "quit", "q"]:
#         break

#     context = retrieve_context(question)
#     answer = ask_ollama(question, context)

#     print("\n" + answer)

while True:
    question = input("\nAsk a question: ")

    if question.lower() in ["exit", "quit", "q"]:
        break

    context = retrieve_context(question)

    print("\n================ CONTEXT ================\n")
    print(context)
    print("\n=========================================\n")

    answer = ask_ollama(question, context)

    print("\n" + answer)