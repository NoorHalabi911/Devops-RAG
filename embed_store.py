from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.txt"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "devops_rag_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"


def is_useful_chunk(text: str) -> bool:
    t = text.lower().strip()

    bad_patterns = [
        "copyright",
        "packt publishing",
        "about the author",
        "about the reviewers",
        "preface",
        "table of contents",
        "index",
    ]

    if len(t) < 120:
        return False

    if any(p in t for p in bad_patterns):
        return False

    # Too many figure/screenshot references usually means noisy visual content
    if t.count("figure") >= 2:
        return False

    # Drop chunks that are mostly page markers / formatting
    if t.count("--- page") >= 2 and len(t) < 300:
        return False

    return True


def clean_chunk_text(text: str) -> str:
    text = re.sub(r"--- PAGE \d+ ---", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_chunks(chunks_file: str) -> list[str]:
    if not Path(chunks_file).exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    with open(chunks_file, "r", encoding="utf-8") as f:
        content = f.read()

    raw_parts = content.split("===== CHUNK ")
    chunks = []

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        lines = part.splitlines()
        if len(lines) > 1:
            chunk_text = "\n".join(lines[1:])
            chunk_text = clean_chunk_text(chunk_text)

            if chunk_text and is_useful_chunk(chunk_text):
                chunks.append(chunk_text)

    return chunks


def create_embeddings(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    print(f"Creating embeddings with local model: {EMBED_MODEL}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def store_in_chroma(chunks: list[str], embeddings: list[list[float]]) -> None:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    existing_collections = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        chroma_client.delete_collection(COLLECTION_NAME)

    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = []

    for i, chunk in enumerate(chunks):
        metadatas.append({
            "source": "devops_pdf",
            "chunk_index": i,
            "has_docker": "docker" in chunk.lower(),
            "has_kubernetes": "kubernetes" in chunk.lower(),
            "has_terraform": "terraform" in chunk.lower(),
            "has_linux": any(x in chunk.lower() for x in ["linux", "systemctl", "journalctl", "ss ", "lsof"]),
        })

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} chunks in Chroma collection '{COLLECTION_NAME}'")


def main() -> None:
    chunks = load_chunks(CHUNKS_FILE)
    print(f"Filtered chunks: {len(chunks)}")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = create_embeddings(model, chunks, batch_size=32)
    print(f"Created {len(embeddings)} embeddings")

    store_in_chroma(chunks, embeddings)


if __name__ == "__main__":
    main()




# from pathlib import Path

# import chromadb
# from sentence_transformers import SentenceTransformer

# CHUNKS_FILE = "chunks.txt"
# CHROMA_PATH = "chroma_db"
# COLLECTION_NAME = "devops_rag_chunks"
# EMBED_MODEL = "all-MiniLM-L6-v2"

# def is_useful_chunk(text: str) -> bool:
#     text = text.lower()

#     # remove garbage
#     if "copyright" in text:
#         return False
#     if "packt publishing" in text:
#         return False
#     if "figure" in text:
#         return False
#     if "page" in text:
#         return False
#     if len(text) < 100:
#         return False

#     return True

# def load_chunks(chunks_file: str) -> list[str]:
#     if not Path(chunks_file).exists():
#         raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

#     with open(chunks_file, "r", encoding="utf-8") as f:
#         content = f.read()

#     raw_parts = content.split("===== CHUNK ")
#     chunks = []

#     for part in raw_parts:
#         part = part.strip()
#         if not part:
#             continue

#         lines = part.splitlines()
#         if len(lines) > 1:
#             chunk_text = "\n".join(lines[1:]).strip()
#             if chunk_text and is_useful_chunk(chunk_text):
#                 chunks.append(chunk_text)
#         print(f"Filtered chunks: {len(chunks)}")
#     return chunks


# def create_embeddings(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> list[list[float]]:
#     print(f"Creating embeddings with local model: {EMBED_MODEL}")
#     embeddings = model.encode(
#         texts,
#         batch_size=batch_size,
#         show_progress_bar=True,
#         convert_to_numpy=True,
#         normalize_embeddings=True,
#     )
#     return embeddings.tolist()


# def store_in_chroma(chunks: list[str], embeddings: list[list[float]]) -> None:
#     chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

#     existing_collections = [c.name for c in chroma_client.list_collections()]
#     if COLLECTION_NAME in existing_collections:
#         chroma_client.delete_collection(COLLECTION_NAME)

#     collection = chroma_client.create_collection(name=COLLECTION_NAME)

#     ids = [f"chunk_{i}" for i in range(len(chunks))]
#     metadatas = [{"source": "devops_pdf", "chunk_index": i} for i in range(len(chunks))]

#     collection.add(
#         ids=ids,
#         documents=chunks,
#         embeddings=embeddings,
#         metadatas=metadatas,
#     )

#     print(f"Stored {len(chunks)} chunks in Chroma collection '{COLLECTION_NAME}'")


# def main() -> None:
#     chunks = load_chunks(CHUNKS_FILE)
#     print(f"Loaded {len(chunks)} chunks")

#     model = SentenceTransformer(EMBED_MODEL)

#     embeddings = create_embeddings(model, chunks, batch_size=32)
#     print(f"Created {len(embeddings)} embeddings")

#     store_in_chroma(chunks, embeddings)


# if __name__ == "__main__":
#     main()