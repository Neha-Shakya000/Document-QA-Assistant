from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


def build_vector_store(documents, embeddings):

    if not documents:
        raise ValueError("No document chunks available.")

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store


def retrieve_relevant_chunks(vector_store, question, top_k=4):

    results = vector_store.similarity_search_with_score(
        question,
        k=min(top_k, vector_store.index.ntotal)
    )

    retrieved = []

    for document, distance in results:

        retrieved.append({
            "chunk": document.page_content,
            "page": document.metadata["page"],
            "chunk_id": document.metadata["chunk_id"],
            "distance": float(distance)
        })

    return retrieved
