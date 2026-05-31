from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd

from metrics import ranking_summary


def _load_method_cache(path: Path) -> tuple[str, list[dict]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and len(payload) == 1:
        method, rows = next(iter(payload.items()))
        return str(method), list(rows)
    if isinstance(payload, list):
        return path.stem, payload
    raise ValueError(f"Unsupported cache format: {path}")


def compute_metrics_from_cache(
    distance_cache_path: str,
    out_csv: str,
    topk: Iterable[int] = (1, 3, 5, 10),
    base_mode: str = "effective",
    ab_alpha: float = 1.0,
    ab_b_min: float = 2.0,
    ab_b_max: float = 20.0,
    ab_c: float = 5.0,
    ab_eps: float = 1e-12,
    gain_mode: str = "binary",
) -> pd.DataFrame:
    cache_dir = Path(distance_cache_path)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Distance cache directory does not exist: {cache_dir}")

    rows: list[dict] = []
    for cache_file in sorted(cache_dir.glob("*.pkl")):
        method, queries = _load_method_cache(cache_file)
        metric_rows = []
        for query in queries:
            distances = query.get("distances")
            labels = query.get("labels")
            if distances is None or labels is None:
                continue
            metric_rows.append(
                ranking_summary(
                    distances,
                    labels,
                    ks=topk,
                    reverse=False,
                    base_mode=base_mode,
                    ab_alpha=ab_alpha,
                    ab_b_min=ab_b_min,
                    ab_b_max=ab_b_max,
                    ab_c=ab_c,
                    ab_eps=ab_eps,
                    gain_mode=gain_mode,
                )
            )
        if not metric_rows:
            continue
        frame = pd.DataFrame(metric_rows)
        row = {"method": method}
        row.update({col: float(frame[col].mean()) for col in frame.columns})
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty and "MAP" in result:
        result = result.sort_values("MAP", ascending=False).reset_index(drop=True)

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    return result
