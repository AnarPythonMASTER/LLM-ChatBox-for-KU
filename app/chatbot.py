import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "ku_mids_docs"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_collection(
    name=COLLECTION_NAME
)


def search(question, n_results=3):
    query_embedding = model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    return results


if __name__ == "__main__":

    while True:
        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            break

        results = search(question)

        print("\nTop Results:\n")

        for i, document in enumerate(
            results["documents"][0]
        ):
            print("=" * 80)
            print(f"Result {i+1}")
            print(document[:1000])
            print()