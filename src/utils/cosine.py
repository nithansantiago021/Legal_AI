"""
cosine.py — Simple cosine similarity for plain Python float lists.
Used by the RAGAS answer relevancy metric to avoid a numpy/torch dependency
in the evaluation module itself.
"""
from __future__ import annotations
import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return round(dot / (mag_a * mag_b), 6)
