import ollama
import json
import re

DECOMPOSE_PROMPT = """You are a query decomposition engine.
Break the user query into 2-4 atomic sub-queries that, when answered independently,
can be combined to fully answer the original.
Return ONLY a JSON array of strings. No explanation. No markdown.

Query: {query}"""


def decompose(query: str) -> list[str]:
    resp = ollama.generate(
        model="qwen2.5-coder:3b",
        prompt=DECOMPOSE_PROMPT.format(query=query),
    )
    raw = resp.response.strip()
    try:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            sub_queries = json.loads(match.group())
            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                return sub_queries
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: return the original query unchanged
    return [query]