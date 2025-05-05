import numpy as np
from typing import List

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a_array = np.array(a, dtype=np.float64)
    b_array = np.array(b, dtype=np.float64)
    return np.dot(a_array, b_array) / (
        np.linalg.norm(a_array) * np.linalg.norm(b_array)
    )