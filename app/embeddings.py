from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Load the sentence-transformer embedding model used
    to convert text chunks into vector representations.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
