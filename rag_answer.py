import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ===== CONFIG =====
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "devops_rag_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"

# OpenRouter model (FREE)
LLM_MODEL = "openrouter/auto"


def retrieve_chunks(question: str, top_k: int = 5):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    model = SentenceTransformer(EMBED_MODEL)

    query_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return results["documents"][0]


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)

    return f"""
You are a DevOps assistant.

Answer the question using ONLY the provided context.
If the context is not relevant, say that clearly.

For definitions:
- explain simply
- give real-world meaning

For troubleshooting:
- possible causes
- commands to run
- suggested fix

Context:
{context}

Question:
{question}
"""


def ask_llm(prompt: str) -> str:
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "DevOps RAG Assistant"
        }
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def main():
    question = input("Ask a DevOps question: ")

    chunks = retrieve_chunks(question)
    prompt = build_prompt(question, chunks)

    answer = ask_llm(prompt)

    print("\n===== FINAL ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    main()