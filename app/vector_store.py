from langchain_community.vectorstores import FAISS

from app.embeddings import get_embedding_model


def create_vector_store(chunks):
    """
    Create a FAISS vector store from document chunks.
    """
    if not chunks:
        raise ValueError("No document chunks provided.")

    embedding_model = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    return vector_store


def get_retriever(vector_store, k=4):
    """
    Return a retriever that fetches the top-k most relevant chunks.
    """
    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )
