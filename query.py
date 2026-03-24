import re
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "devops_rag_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"


def extract_keywords(question: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9\-\._]+", question.lower())
    stopwords = {
        "what", "is", "the", "how", "why", "do", "does", "a", "an", "to",
        "for", "of", "in", "on", "my", "and", "with", "can", "i"
    }
    return [w for w in words if w not in stopwords and len(w) > 2]


def keyword_score(text: str, keywords: list[str]) -> int:
    t = text.lower()
    score = 0
    for kw in keywords:
        if kw in t:
            score += 1
    return score


def search_chunks(question: str, top_k: int = 5, candidate_k: int = 20) -> None:
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
        n_results=candidate_k,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    keywords = extract_keywords(question)

    reranked = []
    for doc, meta, dist in zip(docs, metas, distances):
        kscore = keyword_score(doc, keywords)

        # lower distance is better, so we subtract keyword bonus
        final_score = dist - (0.15 * kscore)

        reranked.append((final_score, dist, kscore, doc, meta))

    reranked.sort(key=lambda x: x[0])
    top_results = reranked[:top_k]

    print(f"\nQuestion: {question}")
    print(f"Keywords: {keywords}\n")

    for i, (final_score, dist, kscore, doc, meta) in enumerate(top_results, start=1):
        print(f"===== RESULT {i} =====")
        print(f"Chunk index: {meta['chunk_index']}")
        print(f"Distance: {dist:.4f}")
        print(f"Keyword score: {kscore}")
        print(f"Final score: {final_score:.4f}")
        print(doc[:1200])
        print()


if __name__ == "__main__":
    user_question = input("Ask a DevOps question: ").strip()
    search_chunks(user_question, top_k=5, candidate_k=20)



# import chromadb
# from sentence_transformers import SentenceTransformer

# CHROMA_PATH = "chroma_db"
# COLLECTION_NAME = "devops_rag_chunks"
# EMBED_MODEL = "all-MiniLM-L6-v2"


# def search_chunks(question: str, top_k: int = 5) -> None:
#     chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
#     collection = chroma_client.get_collection(name=COLLECTION_NAME)

#     model = SentenceTransformer(EMBED_MODEL)
#     query_embedding = model.encode(
#         question,
#         convert_to_numpy=True,
#         normalize_embeddings=True,
#     ).tolist()

#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k,
#     )

#     docs = results["documents"][0]
#     metas = results["metadatas"][0]
#     distances = results["distances"][0]

#     print(f"\nQuestion: {question}\n")
#     print(f"Top {top_k} results:\n")

#     for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
#         print(f"===== RESULT {i} =====")
#         print(f"Chunk index: {meta['chunk_index']}")
#         print(f"Distance: {dist}")
#         print(doc[:1000])
#         print()


# if __name__ == "__main__":
#     user_question = input("Ask a DevOps question: ").strip()
#     search_chunks(user_question, top_k=5)