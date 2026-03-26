import os
from functools import lru_cache

from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# ========= CONFIG =========
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "devops_rag_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openrouter/auto"


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def load_chroma_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_collection(name=COLLECTION_NAME)


@lru_cache(maxsize=1)
def load_llm_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    # Match the original headers you had in the Streamlit app.
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "DevOps RAG Assistant",
        },
    )


def retrieve_chunks(question: str, top_k: int = 5):
    model = load_embedding_model()
    collection = load_chroma_collection()

    query_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)

    return f"""
You are a DevOps troubleshooting assistant.

Answer the question using ONLY the provided context.
If the context is insufficient or not relevant, say that clearly.

For troubleshooting questions:
- explain what the context suggests
- list possible causes if available
- mention commands or checks if present in the context

For definition questions:
- explain simply
- mention why it matters in practice

Keep the answer clear and structured.

Context:
{context}

Question:
{question}
"""


def ask_llm(prompt: str) -> str:
    client = load_llm_client()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content

