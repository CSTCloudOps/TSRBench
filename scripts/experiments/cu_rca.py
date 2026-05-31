from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from distances import build_cu_rca_registry
from distances.utils import align_pair
from loaders import load_cu_rca_files


@dataclass(frozen=True)
class CURCAConfig:
    data_dir: str
    output: str
    cache_dir: str
    methods: str | None = None
    max_files: int | None = None
    align_mode: str = "truncate"


def _method_filter(all_names: list[str], spec: str | None) -> list[str]:
    if not spec:
        return all_names
    requested = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in requested if name not in all_names]
    if unknown:
        raise KeyError(f"Unknown methods: {unknown}. Available: {all_names}")
    return requested


def _hit(labels: list[int], k: int) -> float:
    return float(any(label == 1 for label in labels[:k]))


def run_cu_rca(config: CURCAConfig) -> dict[str, dict[str, float]]:
    files = load_cu_rca_files(config.data_dir, max_files=config.max_files)
    registry = build_cu_rca_registry()
    methods = _method_filter(registry.names(), config.methods)
    baselines = (0, 4, 6)

    cache: dict[str, list[dict]] = {method: [] for method in methods}
    summary: dict[str, dict[str, float]] = {
        method: {"files": 0, "files_with_anomaly": 0, "hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0}
        for method in methods
    }

    for item in files:
        for method_name in methods:
            method = registry.get(method_name)
            per_baseline_hits = []
            records = []
            for baseline_idx in baselines:
                if baseline_idx >= len(item.sensors):
                    continue
                baseline = item.sensors[baseline_idx]
                distances = []
                labels = []
                names = []
                for idx, sensor in enumerate(item.sensors):
                    if idx == baseline_idx:
                        continue
                    a, b = align_pair(baseline.series, sensor.series, mode=config.align_mode)
                    try:
                        score = float(method.func(a, b))
                    except Exception:
                        score = float("-inf")
                    if not np.isfinite(score):
                        score = float("-inf")
                    distances.append(score)
                    labels.append(int(sensor.label))
                    names.append(sensor.name)
                order = np.argsort(np.asarray(distances))[::-1]
                ordered_labels = [labels[i] for i in order.tolist()]
                per_baseline_hits.append(
                    {
                        "hit@1": _hit(ordered_labels, 1),
                        "hit@3": _hit(ordered_labels, 3),
                        "hit@5": _hit(ordered_labels, 5),
                    }
                )
                records.append(
                    {
                        "file": str(item.path),
                        "baseline_index": baseline_idx,
                        "distances": distances,
                        "labels": labels,
                        "names": names,
                    }
                )
            if not per_baseline_hits:
                continue
            cache[method_name].extend(records)
            stats = summary[method_name]
            stats["files"] += 1
            if item.has_anomaly:
                stats["files_with_anomaly"] += 1
                for metric in ("hit@1", "hit@3", "hit@5"):
                    stats[metric] += float(np.mean([h[metric] for h in per_baseline_hits]))

    for method_name, stats in summary.items():
        denom = stats["files_with_anomaly"] or stats["files"] or 1
        for metric in ("hit@1", "hit@3", "hit@5"):
            stats[metric] = float(stats[metric] / denom)

    if config.cache_dir:
        cache_dir = Path(config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for method_name, records in cache.items():
            with (cache_dir / f"{method_name}.pkl").open("wb") as f:
                pickle.dump({method_name: records}, f, protocol=pickle.HIGHEST_PROTOCOL)

    if config.output:
        out = Path(config.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
