import ollama
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

_embedder = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are a strict factual auditor.

Read the context and the answer. Check every factual claim in the answer.
A claim is supported if it can be directly inferred from the context.
A claim is unsupported if it introduces facts not present in the context.

Score:
  1.0 = all claims supported
  0.7 = mostly supported, 1-2 minor unsupported details
  0.3 = many unsupported claims
  0.0 = answer ignores or contradicts the context

You MUST pick a specific score. Do not return 0.5 unless exactly half the claims are unsupported.
Return ONLY valid JSON: {{"score": float, "reason": "one sentence"}}

Context:
{context}

Answer:
{answer}

JSON:"""


def score_faithfulness(answer: str, context: str) -> dict:
    """LLM judge: is the answer grounded in the retrieved context?"""
    if not context.strip():
        return {"score": 0.0, "reason": "No context was retrieved."}

    resp = ollama.generate(
        model="llama3.2",
        prompt=FAITHFULNESS_PROMPT.format(context=context, answer=answer),
    )
    raw = resp.response.strip()

    try:
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: try to extract a float from the response
    numbers = re.findall(r'\d+\.\d+|\d+', raw)
    score = float(numbers[0]) if numbers else 0.5
    score = min(max(score, 0.0), 1.0)
    return {"score": score, "reason": raw[:120]}


# ---------------------------------------------------------------------------
# Answer relevance
# ---------------------------------------------------------------------------

def score_answer_relevance(question: str, answer: str) -> dict:
    """Cosine similarity between question and answer embeddings."""
    embedder = _get_embedder()
    q_emb = embedder.encode([question])
    a_emb = embedder.encode([answer])
    sim = float(cosine_similarity(q_emb, a_emb)[0][0])
    sim = round(min(max(sim, 0.0), 1.0), 4)
    return {
        "score": sim,
        "reason": f"Cosine similarity between question and answer embeddings: {sim:.4f}",
    }


# ---------------------------------------------------------------------------
# Context precision
# ---------------------------------------------------------------------------

def score_context_precision(total_retrieved: int, total_kept: int) -> dict:
    """Fraction of retrieved chunks that survived compression."""
    if total_retrieved == 0:
        return {"score": 0.0, "reason": "No chunks were retrieved."}
    precision = round(total_kept / total_retrieved, 4)
    return {
        "score": precision,
        "reason": f"{total_kept} of {total_retrieved} chunks kept after compression.",
    }


# ---------------------------------------------------------------------------
# Ground truth similarity (optional reference score)
# ---------------------------------------------------------------------------

def score_ground_truth_similarity(answer: str, ground_truth: str) -> dict:
    """Cosine similarity between generated answer and ground truth."""
    embedder = _get_embedder()
    a_emb = embedder.encode([answer])
    g_emb = embedder.encode([ground_truth])
    sim = float(cosine_similarity(a_emb, g_emb)[0][0])
    sim = round(min(max(sim, 0.0), 1.0), 4)
    return {
        "score": sim,
        "reason": f"Cosine similarity between answer and ground truth: {sim:.4f}",
    }