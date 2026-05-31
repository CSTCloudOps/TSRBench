from __future__ import annotations

import numpy as np


def resample_to_length(x, length: int) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32).reshape(-1)
    if values.size == length:
        return values
    if values.size == 0:
        return np.zeros((length,), dtype=np.float32)
    if values.size == 1:
        return np.full((length,), float(values[0]), dtype=np.float32)
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, length)
    return np.interp(dst, src, values).astype(np.float32)


def align_pair(a, b, mode: str = "min") -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if mode == "truncate":
        length = min(a.size, b.size)
        return a[:length], b[:length]
    if mode == "min":
        length = max(2, min(a.size, b.size))
    elif mode == "max":
        length = max(2, max(a.size, b.size))
    else:
        raise ValueError("align mode must be min, max, or truncate")
    return resample_to_length(a, length), resample_to_length(b, length)


def cosine_distance(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(1.0 - float(np.dot(a, b)) / denom)


def l2_distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))

