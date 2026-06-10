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





import os
from openai import OpenAI
import chromadb
import requests
from sentence_transformers import SentenceTransformer


USE_HF = False
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OLLAMA_MODEL = "gemma3:1b"

hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)


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

# def detect_topic(question):
#     q = question.lower()

#     if any(word in q for word in [
#         "professor", "prof", "teacher", "team", "chair", "research group",
#         "stöger", "stoeger", "pfander", "fontaine", "janjic", "janjić",
#         "oliver", "ray", "setzer", "voigtlaender", "voigtländer",
#         "götz", "goetz", "pirmin", "dominik", "felix", "tijana"
#     ]):
#         return "mids_the_team.txt"

#     if any(word in q for word in [
#         "contact", "email", "phone", "address", "room", "secretary",
#         "management", "where is mids", "location", "visiting address",
#         "mailing address"
#     ]):
#         return "mids_contact.txt"

#     if any(word in q for word in [
#         "job", "jobs", "vacancy", "position", "phd", "student assistant",
#         "research assistant", "application", "apply", "salary"
#     ]):
#         return "mids_jobs.txt"

#     if any(word in q for word in [
#         "seminar", "talk", "speaker", "wednesday", "georgianum",
#         "presentation", "lecture", "sommerfest"
#     ]):
#         return "mids_seminar.txt"

#     if any(word in q for word in [
#         "publication", "publications", "paper", "journal", "doi",
#         "article", "research output"
#     ]):
#         return "mids_publications.txt"

#     if any(word in q for word in [
#         "news", "event", "conference", "new scientific director",
#         "half marathon", "data4good", "stukon"
#     ]):
#         return "mids_news.txt"

#     if any(word in q for word in [
#         "mids", "about", "institute", "mathematical institute",
#         "machine learning and data science", "bachelor", "bsc",
#         "master", "msc", "degree program", "data science program"
#     ]):
#         return "mids_about.txt"

#     return None

def set_llm_mode(use_hf):
    global USE_HF
    USE_HF = use_hf
def detect_topic(question):
    q = question.lower()

    # ----------------
    # Individual professors
    # ----------------
    if any(w in q for w in ["fontaine", "pirmin", "logistics", "operations analytics"]):
        return "prof_fontaine.txt"

    if any(w in q for w in ["janjic", "janjić", "tijana", "data assimilation"]):
        return "prof_janjic.txt"

    if any(w in q for w in ["oliver", "marcel", "applied mathematics", "trr 181", "climate science"]):
        return "prof_oliver.txt"

    if any(w in q for w in ["pfander", "götz", "goetz", "scientific computing", "sampling theory", "speaker of the institute"]):
        return "prof_pfander.txt"

    if any(w in q for w in ["ray", "nadja", "geomatics", "geomathematics", "geosciences"]):
        return "prof_ray.txt"

    if any(w in q for w in ["setzer", "thomas", "business informatics", "business analytics", "digital and data-driven business"]):
        return "prof_setzer.txt"

    if any(w in q for w in ["stöger", "stoeger", "stoger", "dominik", "compressed sensing", "low-rank", "signal processing"]):
        return "prof_stoeger.txt"

    if any(w in q for w in ["voigtlaender", "voigtländer", "felix", "reliable machine learning", "adversarial", "robustness"]):
        return "prof_voigtlaender.txt"

    # ----------------
    # International applicants / general admission
    # ----------------
    if any(w in q for w in [
        "international student", "foreign qualification", "bildungsausländer",
        "uni-assist", "uni assist", "aps", "studienkolleg",
        "preparatory college", "foreign university entrance qualification",
        "grade conversion", "anabin", "daad database",
        "language requirements international", "international application"
    ]):
        return "application_admission_international_students.txt"

    # ----------------
    # General FAQ
    # ----------------
    if any(w in q for w in [
        "faq", "visa late", "late visa", "when should i arrive",
        "arrive in ingolstadt", "self-study", "self study",
        "online", "hybrid", "accommodation", "housing",
        "german lessons", "english in ingolstadt", "monthly expenses",
        "private or public university", "is ku private", "gym", "swimming pool",
        "semester abroad", "preliminary course"
    ]):
        return "FAQ.txt"

    # ----------------
    # MIDS general
    # ----------------
    if any(w in q for w in [
        "what is mids", "about mids", "about the institute",
        "mathematical institute", "institute for machine learning",
        "machine learning and data science institute"
    ]):
        return "mids_about.txt"

    if any(w in q for w in [
        "mids contact", "mids address", "where is mids", "mids located",
        "mids location", "mids management", "mids secretary",
        "mids phone", "mids email", "visiting address", "mailing address"
    ]):
        return "mids_contact.txt"

    if any(w in q for w in [
        "mids job", "jobs at mids", "vacancy", "vacancies",
        "phd position", "student assistant", "research assistant"
    ]):
        return "mids_jobs.txt"

    if any(w in q for w in [
        "mids seminar", "seminar", "talk", "speaker", "georgianum",
        "wednesday", "sommerfest"
    ]):
        return "mids_seminar.txt"

    if any(w in q for w in [
        "mids news", "news", "event", "conference", "data4good",
        "scientific director", "stukon", "half marathon"
    ]):
        return "mids_news.txt"

    if any(w in q for w in [
        "publication", "publications", "paper", "journal", "doi",
        "research output"
    ]):
        return "mids_publications.txt"

    # ----------------
    # Bachelor Data Science
    # ----------------
    if any(w in q for w in [
        "internship",
        "internships",
        "practical connection",
        "industrial internship",
        "research internship",
        "audi",
        "continental",
        "airbus"
    ]):
        return "bachelor_internship.txt"
    
    if any(w in q for w in [
        "bachelor deadline", "bachelor application deadline",
        "bachelor general", "bachelor degree", "semester fee bachelor",
        "standard length bachelor", "place of study bachelor",
        "language of instruction bachelor", "bachelor start"
    ]):
        return "bachelor_general_and_application_deadlines.txt"

    if any(w in q for w in [
        "bachelor apply", "bachelor application", "uni-assist bachelor",
        "aptitude assessment bachelor", "bachelor requirements",
        "bachelor admission", "bachelor documents", "english skills bachelor",
        "german a2 bachelor"
    ]):
        return "bachelor_application.txt"

    if any(w in q for w in [
        "bachelor structure", "bachelor ects", "180 ects",
        "required area", "required modules", "bachelor modules",
        "bachelor thesis", "bachelor seminar"
    ]):
        return "bachelor_structure.txt"

    if any(w in q for w in [
        "bachelor specialization", "specialization", "focus area",
        "applied mathematics and scientific computing",
        "business analytics and operations",
        "digital transformation of society",
        "environmental sciences", "finance and economics",
        "machine learning and statistics"
    ]):
        return "bachelor_in_detail_specializations.txt"

    if any(w in q for w in [
        "bachelor study program", "what is bachelor data science",
        "bsc data science", "bachelor data science about",
        "data science bachelor about"
    ]):
        return "bachelor_study_program.txt"

    if any(w in q for w in [
        "bachelor internship", "industrial internship",
        "research internship", "audi", "continental", "airbus",
        "practical connection", "career center"
    ]):
        return "bachelor_internship.txt"

    if any(w in q for w in [
        "bachelor career", "career possibilities", "job after bachelor",
        "occupational fields", "graduates", "data scientist job"
    ]):
        return "bachelor_career_possibilities.txt"

    if any(w in q for w in [
        "bachelor consulting", "bachelor advisory", "consultation hours",
        "office hour", "sarah eberle", "armelle langenwald",
        "international office"
    ]):
        return "bachelor_consulting.txt"

    if any(w in q for w in [
        "bachelor contact", "bachelor contact person",
        "examination office", "program spokesperson",
        "subject advisor", "examinations committee"
    ]):
        return "bachelor_contact_persons.txt"

    if any(w in q for w in [
        "ranking", "rankings", "studycheck", "popular university",
        "student satisfaction", "recommend studying"
    ]):
        return "bachelor_rankings_and_assesments.txt"

    if any(w in q for w in [
        "testimonial", "testimonials", "student opinion",
        "student experience", "fantastic four", "jan stüwe",
        "alena", "aditi", "shizhen", "oleksandr"
    ]):
        return "bachelor_testimonials.txt"

    if any(w in q for w in [
        "bachelor faq", "bachelor question", "bachelor visa",
        "bachelor accommodation", "bachelor housing",
        "bachelor online", "bachelor hybrid",
        "bachelor german lessons"
    ]):
        return "bachelor_faq.txt"

    # ----------------
    # Master Data Science
    # ----------------
    
    if any(w in q for w in [
        "master admission",
        "master requirements",
        "master application",
        "admission requirements",
        "requirements for the master",
        "apply for master",
        "master documents",
        "master aptitude test"
    ]):
        return "master_application.txt"

    if any(w in q for w in [
        "study abroad",
        "semester abroad",
        "internationalization",
        "internationalisation",
        "global network",
        "abroad"
    ]):
        return "master_internatiolization_study_abroad.txt"
    
    if any(w in q for w in [
        "ects",
        "120 ects",
        "credits",
        "credit points",
        "program structure",
        "master structure"
    ]):
        return "master_in_detail_and_program_structure.txt"
    

    if any(w in q for w in [
        "master general", "msc general", "master degree",
        "what is master data science", "msc data science",
        "master data science about", "standard length master",
        "place of study master", "language of instruction master"
    ]):
        return "master_general.txt"

    if any(w in q for w in [
        "master study program", "master course", "msc study program",
        "winter semester 2025", "master focuses", "data analytics",
        "operations research", "weather and climate"
    ]):
        return "master_study_program.txt"

    if any(w in q for w in [
        "master application", "master apply", "master admission",
        "master requirements", "master deadline", "master documents",
        "aptitude test master", "uni-assist master", "lisbon convention",
        "gre", "module prerequisites"
    ]):
        return "master_application.txt"

    if any(w in q for w in [
        "master structure", "master program structure", "120 ects",
        "master ects", "mathematics for data science",
        "advanced programming", "database management",
        "applied data science project", "studium.pro"
    ]):
        return "master_in_detail_and_program_structure.txt"

    if any(w in q for w in [
        "master abroad", "study abroad master", "internationalization",
        "internationalisation", "semester abroad master",
        "international focus", "global network"
    ]):
        return "master_internatiolization_study_abroad.txt"

    if any(w in q for w in [
        "master consulting", "master advice", "master advisory",
        "master consultation"
    ]):
        return "master_advice_and_consulting.txt"

    if any(w in q for w in [
        "master contact", "master contact person", "contact persons master"
    ]):
        return "master_contact_persons.txt"

    # ----------------
    # Fallback broad routing
    # ----------------
    if "bachelor" in q or "bsc" in q:
        return "bachelor_general_and_application_deadlines.txt"

    if "master" in q or "msc" in q:
        return "master_general.txt"

    if "professor" in q or "prof" in q or "chair" in q or "research group" in q:
        return "mids_the_team.txt"

    return None

def retrieve_context(question, n_results=8): # i changed 6 to 8 because we have now more small txt files
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

    #return "\n\n".join(context_parts)
    sources_used = list(
        set(
            meta.get("source", "unknown")
            for meta in metadatas
        )
    )

    return (
        "\n\n".join(context_parts),
        sources_used
    )



# def ask_ollama(question, context):
def ask_local_ollama(question, context):
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


def ask_hf(question, context):
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
- MIDS belongs to KU Eichstätt-Ingolstadt.
- Never call it Ingolstadt University of Technology.
- Do NOT invent university names, locations, professors, admission rules, or program details.
- If the context does not clearly say something, say:
"I do not have enough information in my current knowledge base."
- If the question is not related to KU/MIDS, answer:
"This chatbot is designed only for KU/MIDS-related questions."
- Keep answers short, factual, and student-friendly.

Context:
{context}

Question:
{question}

Answer:
"""

    response = hf_client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=180,
        temperature=0.1
    )

    return response.choices[0].message.content

def ask_llm(question, context):

    if USE_HF:
        return ask_hf(question, context)

    return ask_local_ollama(question, context)


# while True:
#     question = input("\nAsk a question: ")

#     if question.lower() in ["exit", "quit", "q"]:
#         break

#     context = retrieve_context(question)
#     answer = ask_ollama(question, context)

#     print("\n" + answer)

# while True:
#     question = input("\nAsk a question: ")

#     if question.lower() in ["exit", "quit", "q"]:
#         break

#     #context = retrieve_context(question)
#     context, sources = retrieve_context(question)

#     # print("\n================ CONTEXT ================\n")
#     # print(context)
#     # print("\n=========================================\n")
#     print("\nSources used:")

#     for source in sources:
#         print("-", source)
#     answer = ask_ollama(question, context)

#     print("\n" + answer)
    
if __name__ == "__main__":
    while True:
        question = input("\nAsk a question: ")

        if question.lower() in ["exit", "quit", "q"]:
            break

        context, sources = retrieve_context(question)

        print("\nSources used:")
        for source in sources:
            print("-", source)

        # answer = ask_ollama(question, context)
        answer = ask_llm(question, context)
        print("\n" + answer)
