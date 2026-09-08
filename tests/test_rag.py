from app.rag_engine import split_documents
from langchain_core.documents import Document


def test_split_documents():
    documents = [
        Document(
            page_content="This is a sample document. " * 100
        )
    ]

    chunks = split_documents(documents)

    assert len(chunks) > 0
    assert all(chunk.page_content for chunk in chunks)
