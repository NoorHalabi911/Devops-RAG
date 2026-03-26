import os
from dotenv import load_dotenv

import chromadb
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# ========= CONFIG =========
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "devops_rag_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openrouter/auto"


# ========= PAGE =========
st.set_page_config(
    page_title="OpsRecall",
    page_icon="🛠️",
    layout="wide"
)

# ========= STYLES =========
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #94a3b8;
    margin-bottom: 1.5rem;
}
.answer-box {
    padding: 1rem;
    border-radius: 14px;
    background-color: #0f172a;
    border: 1px solid #1e293b;
}
.small-label {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ========= CACHED LOADERS =========
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def load_chroma_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_collection(name=COLLECTION_NAME)


@st.cache_resource
def load_llm_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "DevOps RAG Assistant"
        }
    )


# ========= RAG FUNCTIONS =========
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
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# ========= SIDEBAR =========
with st.sidebar:
    st.title("⚙️ Settings")
    top_k = st.slider("Retrieved chunks", 3, 8, 5)
    show_debug = st.toggle("Show retrieval debug", value=True)

    st.markdown("---")
    st.markdown("### Example questions")
    st.markdown("- What is Docker?")
    st.markdown("- What is Kubernetes?")
    st.markdown("- How does Docker Compose work?")
    st.markdown("- What is CI/CD?")
    st.markdown("- Why is my port not working?")

    st.markdown("---")
    st.caption("OpsRecall • Personal DevOps RAG Assistant")
    if st.button("Clear chat", use_container_width=True, type="secondary"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! Ask me a DevOps question and I’ll answer using your PDF knowledge base.",
            }
        ]
        st.rerun()


# ========= HEADER =========
st.markdown('<div class="main-title">🛠️ OpsRecall</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A DevOps RAG assistant powered by your PDF knowledge base</div>',
    unsafe_allow_html=True
)

# ========= CHAT STATE =========
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me a DevOps question and I’ll answer using your PDF knowledge base.",
        }
    ]

# ========= MAIN CHAT =========
use_chat_ui = hasattr(st, "chat_input") and hasattr(st, "chat_message")

for msg in st.session_state.messages:
    if use_chat_ui:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Render assistant answers inside the existing "answer-box" container.
                st.markdown(
                    f"<div class='answer-box'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"])

            retrieval = msg.get("retrieval")
            if msg["role"] == "assistant" and retrieval:
                best_distance = retrieval.get("best_distance")
                n_chunks = retrieval.get("n_chunks", 0)

                if best_distance is not None:
                    st.caption(f"Retrieved {n_chunks} chunks • Best distance: {best_distance:.4f}")
                else:
                    st.caption(f"Retrieved {n_chunks} chunks")

                if best_distance is not None and best_distance > 1.4:
                    st.warning("Retrieval quality looks weak. The answer may be based on poor context.")

                if show_debug and retrieval.get("chunks"):
                    with st.expander("Retrieved source chunks"):
                        chunks = retrieval["chunks"]
                        metadatas = retrieval.get("metadatas", [])
                        distances = retrieval.get("distances", [])

                        for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances), start=1):
                            title = f"Chunk {i} • distance={dist:.4f}"
                            with st.expander(title):
                                st.caption(f"chunk_index: {meta.get('chunk_index', 'N/A')}")
                                st.write(chunk)

                    if retrieval.get("prompt"):
                        with st.expander("Prompt sent to the LLM"):
                            st.code(retrieval["prompt"])
    else:
        # Fallback UI for older Streamlit versions.
        if msg["role"] == "assistant":
            st.markdown(f"<div class='answer-box'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

        retrieval = msg.get("retrieval")
        if msg["role"] == "assistant" and retrieval:
            best_distance = retrieval.get("best_distance")
            n_chunks = retrieval.get("n_chunks", 0)

            if best_distance is not None:
                st.caption(f"Retrieved {n_chunks} chunks • Best distance: {best_distance:.4f}")
            else:
                st.caption(f"Retrieved {n_chunks} chunks")

            if best_distance is not None and best_distance > 1.4:
                st.warning("Retrieval quality looks weak. The answer may be based on poor context.")

            if show_debug and retrieval.get("chunks"):
                with st.expander("Retrieved source chunks"):
                    chunks = retrieval["chunks"]
                    metadatas = retrieval.get("metadatas", [])
                    distances = retrieval.get("distances", [])

                    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances), start=1):
                        title = f"Chunk {i} • distance={dist:.4f}"
                        with st.expander(title):
                            st.caption(f"chunk_index: {meta.get('chunk_index', 'N/A')}")
                            st.write(chunk)

                if retrieval.get("prompt"):
                    with st.expander("Prompt sent to the LLM"):
                        st.code(retrieval["prompt"])

new_question = None
if use_chat_ui:
    new_question = st.chat_input(placeholder="e.g. What is Docker? or Why is my port not working?")
else:
    question = st.text_input(
        "Ask a DevOps question",
        placeholder="e.g. What is Docker? or Why is my port not working?",
    )
    ask = st.button("Ask", use_container_width=True)
    if ask:
        new_question = question

if new_question and new_question.strip():
    question = new_question.strip()
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Retrieving context and generating answer..."):
        try:
            retrieved = retrieve_chunks(question, top_k=top_k)
            chunks = retrieved["documents"]
            metadatas = retrieved["metadatas"]
            distances = retrieved["distances"]

            prompt = build_prompt(question, chunks)
            answer = ask_llm(prompt)

            best_distance = distances[0] if distances else None
            assistant_msg = {
                "role": "assistant",
                "content": answer,
                "retrieval": {
                    "n_chunks": len(chunks),
                    "best_distance": best_distance,
                },
            }

            # Store extra details only when debug is enabled to avoid large session state.
            if show_debug:
                assistant_msg["retrieval"].update(
                    {
                        "chunks": chunks,
                        "metadatas": metadatas,
                        "distances": distances,
                        "prompt": prompt,
                    }
                )

            st.session_state.messages.append(assistant_msg)
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Sorry—something went wrong while generating the answer. Please try again.",
                }
            )

    st.rerun()