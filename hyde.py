# hyde.py
import ollama

HYDE_PROMPT = """Write a short, factual passage (3-5 sentences) that would
directly answer the following question. Write as if it's from a technical doc.

Question: {query}"""

def generate_hypothetical_doc(query: str) -> str:
    resp = ollama.generate(model="llama3.2", prompt=HYDE_PROMPT.format(query=query))
    return resp.response

def hyde_embed(query: str, embedder) -> list[float]:
    hyp_doc = generate_hypothetical_doc(query)
    return embedder.encode(hyp_doc).tolist()