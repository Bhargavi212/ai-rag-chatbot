from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from app.document_loader import load_pdf
from app.vector_store import create_vector_store, get_retriever
from app.llm_service import get_llm


def split_documents(documents):
    """
    Split loaded PDF documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return text_splitter.split_documents(documents)


def build_rag_pipeline(file_path: str):
    """
    Build the RAG pipeline from a PDF file.
    """
    documents = load_pdf(file_path)
    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)
    retriever = get_retriever(vector_store)

    return retriever


def ask_question(retriever, question: str):
    """
    Retrieve relevant document chunks and generate
    a grounded answer using the LLM.
    """
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant answering questions from provided documents.

Use only the context below to answer the question.

If the answer cannot be found in the context, say:
"I could not find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = get_llm()

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    sources = [
        {
            "page": doc.metadata.get("page"),
            "source": doc.metadata.get("source"),
        }
        for doc in relevant_docs
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
