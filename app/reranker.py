from sentence_transformers import CrossEncoder
from functools import lru_cache
from typing import List, Tuple

@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, passages: List[Tuple[str, dict]], top_k: int = 5):
    if not passages:
        return []
    model = get_reranker()
    pairs = [(query, p[0]) for p in passages]
    scores = model.predict(pairs)
    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    # Keep full (text, payload) along with score
    return [((text, payload), float(s)) for (text, payload), s in ranked[:top_k]]
