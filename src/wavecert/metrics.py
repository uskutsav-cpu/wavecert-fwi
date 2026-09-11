from __future__ import annotations

import numpy as np


def relative_l2(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-15) -> float:
    ref = np.asarray(reference)
    est = np.asarray(estimate)
    return float(np.linalg.norm(est - ref) / max(np.linalg.norm(ref), eps))


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    denom = max(float(np.linalg.norm(x) * np.linalg.norm(y)), eps)
    return float(np.dot(x, y) / denom)
