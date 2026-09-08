import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def get_llm():
    """
    Initialize and return the Groq LLM used for answer generation.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    return ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )
