import fitz  # PyMuPDF
import re
from pathlib import Path


PDF_PATH = "data/devops_notes.pdf"


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF."""
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            full_text.append(f"\n--- PAGE {page_num} ---\n{text}")

    doc.close()
    return "\n".join(full_text)


def clean_text(text: str) -> str:
    """Basic cleanup to reduce noisy spacing."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping character-based chunks.
    Good starting point for RAG MVP.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def save_chunks(chunks: list[str], output_path: str = "chunks.txt") -> None:
    """Save chunks to a text file so you can inspect them."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"===== CHUNK {i} =====\n")
            f.write(chunk)
            f.write("\n\n")


def main() -> None:
    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    raw_text = extract_text_from_pdf(PDF_PATH)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text, chunk_size=800, overlap=150)

    print(f"Extracted characters: {len(cleaned_text)}")
    print(f"Total chunks: {len(chunks)}")
    print("\nFirst chunk preview:\n")
    print(chunks[0][:1000])

    save_chunks(chunks, "chunks.txt")
    print("\nChunks saved to chunks.txt")


if __name__ == "__main__":
    main()