from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _valid_scores_labels(scores: Sequence[float], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)
    mask = np.isfinite(scores_arr)
    return scores_arr[mask], labels_arr[mask]


def _ordered_labels(scores: Sequence[float], labels: Sequence[int], reverse: bool = False) -> np.ndarray:
    scores_arr, labels_arr = _valid_scores_labels(scores, labels)
    order = np.argsort(scores_arr, kind="mergesort")
    if reverse:
        order = order[::-1]
    return labels_arr[order]


def hit_at_k(scores: Sequence[float], labels: Sequence[int], k: int, reverse: bool = False) -> float:
    ordered = _ordered_labels(scores, labels, reverse=reverse)
    return float(np.any(ordered[:k] == 1))


def precision_at_k(scores: Sequence[float], labels: Sequence[int], k: int, reverse: bool = False) -> float:
    ordered = _ordered_labels(scores, labels, reverse=reverse)[:k]
    return float(np.mean(ordered == 1)) if ordered.size else 0.0


def recall_at_k(scores: Sequence[float], labels: Sequence[int], k: int, reverse: bool = False) -> float:
    ordered = _ordered_labels(scores, labels, reverse=reverse)
    _, valid_labels = _valid_scores_labels(scores, labels)
    positives = int(np.sum(valid_labels == 1))
    if positives == 0:
        return 0.0
    return float(np.sum(ordered[:k] == 1) / positives)


def average_precision(scores: Sequence[float], labels: Sequence[int], k: int | None = None, reverse: bool = False) -> float:
    ordered = _ordered_labels(scores, labels, reverse=reverse)
    if k is not None:
        ordered = ordered[:k]
    if np.sum(ordered == 1) == 0:
        return 0.0
    hits = 0
    values = []
    for rank, label in enumerate(ordered, start=1):
        if label == 1:
            hits += 1
            values.append(hits / rank)
    return float(np.mean(values)) if values else 0.0


def ndcg_at_k(scores: Sequence[float], labels: Sequence[int], k: int, reverse: bool = False) -> float:
    ordered = _ordered_labels(scores, labels, reverse=reverse)[:k].astype(np.float64)
    if ordered.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, ordered.size + 2))
    dcg = float(np.sum(ordered * discounts))
    _, valid_labels = _valid_scores_labels(scores, labels)
    ideal = np.sort(valid_labels.astype(np.float64))[::-1][:k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal.size + 2))
    idcg = float(np.sum(ideal * ideal_discounts))
    return float(dcg / (idcg + 1e-12))


def adaptive_ndcg(scores: Sequence[float], labels: Sequence[int], reverse: bool = False) -> float:
    _, valid_labels = _valid_scores_labels(scores, labels)
    positives = int(np.sum(valid_labels == 1))
    if positives <= 0:
        return 0.0
    return ndcg_at_k(scores, labels, k=positives, reverse=reverse)


def _weights_for_scheme(scheme: str, kstar: int, c: float = 5.0, eps: float = 1e-12) -> np.ndarray:
    if kstar <= 0:
        return np.zeros((0,), dtype=np.float64)
    idx = np.arange(1, kstar + 1, dtype=np.float64)
    if scheme == "log":
        return 1.0 / np.log(idx + 1.0)
    if scheme == "lin":
        return 1.0 / idx
    if scheme == "exp":
        alpha_q = float(c) / (float(kstar) + eps)
        return np.exp(-alpha_q * (idx - 1.0))
    raise ValueError(f"unknown scheme: {scheme}")


def _query_base_from_pool(
    labels: Sequence[int],
    alpha: float = 1.0,
    b_min: float = 2.0,
    b_max: float = 20.0,
) -> float:
    rels = np.asarray(labels, dtype=np.int64)
    positives = int(np.sum(rels == 1))
    total = int(rels.size)
    distractors = max(0, total - positives)
    if positives <= 0:
        return float(b_min)
    hardness = distractors / max(1, positives)
    base = 1.0 + alpha * np.log1p(hardness)
    return float(np.clip(base, b_min, b_max))


def _weights_for_scheme_pool(scheme: str, n: int, base: float, eps: float = 1e-12) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    idx = np.arange(1, n + 1, dtype=np.float64)
    if scheme == "log":
        return 1.0 / ((np.log(idx + 1.0) / (np.log(base) + eps)) + eps)
    if scheme == "lin":
        return 1.0 / (base + idx)
    if scheme == "exp":
        return np.exp(-idx / (base + eps))
    raise ValueError(f"unknown scheme: {scheme}")


def ab_ndcg(
    scores: Sequence[float],
    labels: Sequence[int],
    scheme: str,
    reverse: bool = False,
    gain_mode: str = "binary",
    c: float = 5.0,
    eps: float = 1e-12,
) -> float:
    rels = _ordered_labels(scores, labels, reverse=reverse).astype(np.int64)
    if rels.size == 0:
        return 0.0
    pos_idx = np.where(rels == 1)[0]
    if pos_idx.size == 0:
        return 0.0
    kstar = int(pos_idx[-1] + 1)
    w = _weights_for_scheme(scheme, kstar, c=c, eps=eps)
    if gain_mode == "binary":
        gains = rels[:kstar].astype(np.float64)
    else:
        gains = (2.0 ** rels[:kstar].astype(np.float64)) - 1.0
    dcg = float(np.sum(gains * w))
    positives = int(np.sum(rels))
    idcg = float(np.sum(w[:positives]))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def ab_ndcg_pool(
    scores: Sequence[float],
    labels: Sequence[int],
    scheme: str,
    reverse: bool = False,
    gain_mode: str = "binary",
    alpha: float = 1.0,
    b_min: float = 2.0,
    b_max: float = 20.0,
    eps: float = 1e-12,
) -> float:
    rels = _ordered_labels(scores, labels, reverse=reverse).astype(np.int64)
    if rels.size == 0:
        return 0.0
    positives = int(np.sum(rels == 1))
    if positives <= 0:
        return 0.0
    base = _query_base_from_pool(labels, alpha=alpha, b_min=b_min, b_max=b_max)
    weights = _weights_for_scheme_pool(scheme, n=rels.size, base=base, eps=eps)
    if gain_mode == "binary":
        gains = rels.astype(np.float64)
    elif gain_mode == "graded":
        gains = (2.0 ** rels.astype(np.float64)) - 1.0
    else:
        raise ValueError(f"unknown gain_mode: {gain_mode}")
    ideal_gains = np.sort(gains)[::-1]
    dcg = float(np.sum(gains * weights))
    idcg = float(np.sum(ideal_gains * weights))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def ab_map(
    scores: Sequence[float],
    labels: Sequence[int],
    scheme: str,
    reverse: bool = False,
    c: float = 5.0,
    eps: float = 1e-12,
) -> float:
    rels = _ordered_labels(scores, labels, reverse=reverse).astype(np.int64)
    if rels.size == 0:
        return 0.0
    pos_idx = np.where(rels == 1)[0]
    if pos_idx.size == 0:
        return 0.0
    kstar = int(pos_idx[-1] + 1)
    positives = int(np.sum(rels))
    w = _weights_for_scheme(scheme, max(kstar, positives), c=c, eps=eps)
    cum = np.cumsum(rels).astype(np.float64)
    precision = cum / np.arange(1, rels.size + 1, dtype=np.float64)
    numerator = float(np.sum(w[:kstar] * precision[:kstar] * rels[:kstar]))
    denominator = float(np.sum(w[:positives]))
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def ab_map_pool(
    scores: Sequence[float],
    labels: Sequence[int],
    scheme: str,
    reverse: bool = False,
    alpha: float = 1.0,
    b_min: float = 2.0,
    b_max: float = 20.0,
    eps: float = 1e-12,
) -> float:
    rels = _ordered_labels(scores, labels, reverse=reverse).astype(np.int64)
    if rels.size == 0:
        return 0.0
    positives = int(np.sum(rels == 1))
    if positives <= 0:
        return 0.0
    base = _query_base_from_pool(labels, alpha=alpha, b_min=b_min, b_max=b_max)
    weights = _weights_for_scheme_pool(scheme, n=rels.size, base=base, eps=eps)
    cum_relevant = np.cumsum(rels).astype(np.float64)
    precision = cum_relevant / np.arange(1, rels.size + 1, dtype=np.float64)
    numerator = float(np.sum(weights * precision * rels))
    denominator = float(np.sum(weights[:positives]))
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def ranking_summary(
    scores: Sequence[float],
    labels: Sequence[int],
    ks: Iterable[int] = (1, 3, 5, 10),
    reverse: bool = False,
    base_mode: str = "effective",
    ab_alpha: float = 1.0,
    ab_b_min: float = 2.0,
    ab_b_max: float = 20.0,
    ab_c: float = 5.0,
    ab_eps: float = 1e-12,
    gain_mode: str = "binary",
) -> dict[str, float]:
    if base_mode not in {"effective", "pool"}:
        raise ValueError("base_mode must be 'effective' or 'pool'")
    out: dict[str, float] = {}
    for k in ks:
        out[f"avg_hit@{k}"] = hit_at_k(scores, labels, k, reverse=reverse)
        out[f"map@{k}"] = average_precision(scores, labels, k=k, reverse=reverse)
        out[f"avg_precision@{k}"] = precision_at_k(scores, labels, k, reverse=reverse)
        out[f"avg_recall@{k}"] = recall_at_k(scores, labels, k, reverse=reverse)
        out[f"avg_ndcg@{k}"] = ndcg_at_k(scores, labels, k, reverse=reverse)
    out["MAP"] = average_precision(scores, labels, reverse=reverse)
    out["avg_adaptive_ndcg"] = adaptive_ndcg(scores, labels, reverse=reverse)
    out["NDCG@10"] = ndcg_at_k(scores, labels, k=10, reverse=reverse)
    out["MAP@10"] = average_precision(scores, labels, k=10, reverse=reverse)
    for scheme in ("log", "lin", "exp"):
        if base_mode == "pool":
            out[f"AB-NDCG_{scheme}"] = ab_ndcg_pool(
                scores,
                labels,
                scheme,
                reverse=reverse,
                gain_mode=gain_mode,
                alpha=ab_alpha,
                b_min=ab_b_min,
                b_max=ab_b_max,
                eps=ab_eps,
            )
        else:
            out[f"AB-NDCG_{scheme}"] = ab_ndcg(
                scores,
                labels,
                scheme,
                reverse=reverse,
                gain_mode=gain_mode,
                c=ab_c,
                eps=ab_eps,
            )
    for scheme in ("log", "lin", "exp"):
        if base_mode == "pool":
            out[f"AB-MAP_{scheme}"] = ab_map_pool(
                scores,
                labels,
                scheme,
                reverse=reverse,
                alpha=ab_alpha,
                b_min=ab_b_min,
                b_max=ab_b_max,
                eps=ab_eps,
            )
        else:
            out[f"AB-MAP_{scheme}"] = ab_map(
                scores,
                labels,
                scheme,
                reverse=reverse,
                c=ab_c,
                eps=ab_eps,
            )
    return out
