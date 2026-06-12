import pathlib
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"


def ingest_docs(docs_dir: str, collection_name: str) -> int:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_or_create_collection(collection_name)

    total_chunks = 0

    for path in pathlib.Path(docs_dir).rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 80]
        if not chunks:
            continue

        embeddings = embedder.encode(chunks).tolist()
        col.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{path.stem}_{i}" for i, _ in enumerate(chunks)],
            metadatas=[{"source": str(path)} for _ in chunks],
        )
        total_chunks += len(chunks)

    return total_chunks