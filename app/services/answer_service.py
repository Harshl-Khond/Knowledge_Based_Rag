import os
import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """Use Groq LLM to generate a concise answer from retrieved document chunks."""

    context = "\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions based strictly on the provided context. "
                "Give a clear, concise, and direct answer. If the answer is not found in the context, "
                "say 'Sorry, I couldn’t find the exact information right now. For detailed information, please visit our office:  Mukta Plaza, KITS Square, Income Tax Chowk,  Gaurakshan Road, Akola, Maharashtra 444001, India Our team will be happy to assist you!'"
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 300,
    }

    response = httpx.post(GROQ_API_URL, headers=headers, json=payload, timeout=30.0)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]
