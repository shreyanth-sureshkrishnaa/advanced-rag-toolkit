import ollama

COMPRESS_PROMPT = """Given a question and a retrieved passage, extract sentences
from the passage relevant to answering the question.

Be INCLUSIVE — if a sentence provides useful background or partial information,
keep it. Only return IRRELEVANT if the passage has absolutely no connection
to the question topic.

Question: {query}
Passage: {passage}

Extracted sentences:"""


def compress_chunk(query: str, chunk: str) -> str | None:
    """Return only the relevant sentences from a chunk, or None if irrelevant."""
    resp = ollama.generate(
        model="qwen2.5-coder:3b",
        prompt=COMPRESS_PROMPT.format(query=query, passage=chunk),
    )
    result = resp.response.strip()
    if result.upper() == "IRRELEVANT" or not result:
        return None
    return result