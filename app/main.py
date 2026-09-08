from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil

from app.rag_engine import build_rag_pipeline, ask_question


app = FastAPI(
    title="AI-Powered RAG Chatbot",
    description="Upload a PDF and ask questions using Retrieval-Augmented Generation.",
    version="1.0.0",
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

retriever = None


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "AI-Powered RAG Chatbot API is running."
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global retriever

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        retriever = build_rag_pipeline(file_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(exc)}",
        )

    return {
        "message": "PDF uploaded and indexed successfully.",
        "filename": file.filename,
    }


@app.post("/ask")
def ask(request: QuestionRequest):
    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF before asking a question.",
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = ask_question(
            retriever=retriever,
            question=request.question,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(exc)}",
        )

    return result
