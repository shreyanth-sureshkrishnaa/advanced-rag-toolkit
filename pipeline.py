import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from rich.console import Console
from decompose import decompose
from hyde import hyde_embed
from compress import compress_chunk

CHROMA_PATH = "./chroma_db"
console = Console()


class AdvancedRAG:
    def __init__(self, collection_name: str, debug: bool = False):
        self.debug = debug
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma.get_collection(collection_name)

    def query(self, question: str, top_k: int = 4) -> str:
        result = self._query_with_metadata(question, top_k=top_k)
        return result["answer"]

    def _query_with_metadata(self, question: str, top_k: int = 4) -> dict:
        """Full pipeline returning answer + retrieval metadata for eval."""
        sub_queries = decompose(question)

        if self.debug:
            console.print(f"[dim]Sub-queries: {sub_queries}[/dim]")

        compressed_contexts = []
        seen_ids = set()
        total_retrieved = 0
        total_kept = 0

        for sq in sub_queries:
            embedding = hyde_embed(sq, self.embedder)
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
            )
            for doc, doc_id in zip(results["documents"][0], results["ids"][0]):
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                total_retrieved += 1
                compressed = compress_chunk(sq, doc)

                if self.debug:
                    console.print(f"[dim]Compressed [{doc_id}]: {compressed}[/dim]")

                if compressed:
                    total_kept += 1
                    compressed_contexts.append(f"[Re: {sq}]\n{compressed}")

        if not compressed_contexts:
            return {
                "answer": "I couldn't find relevant information.",
                "context": "",
                "sub_queries": sub_queries,
                "total_retrieved": total_retrieved,
                "total_kept": total_kept,
            }

        context = "\n\n---\n\n".join(compressed_contexts)
        resp = ollama.generate(
            model="llama3.2",
            prompt=f"""Answer the question using only the provided context.

Context:
{context}

Question: {question}
Answer:""",
        )

        return {
            "answer": resp.response,
            "context": context,
            "sub_queries": sub_queries,
            "total_retrieved": total_retrieved,
            "total_kept": total_kept,
        }