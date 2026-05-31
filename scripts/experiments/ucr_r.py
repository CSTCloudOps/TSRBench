from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from distances import build_ucr_r_registry
from distances.utils import align_pair
from embeddings import build_embedding_banks
from embeddings.banks import EmbeddingConfig
from loaders import load_ucr_r
from metrics import ranking_summary

logger = logging.getLogger(__name__)
Key = Tuple[int, int]


@dataclass(frozen=True)
class UCRRConfig:
    data_dir: str
    out_csv: str
    distance_cache_path: str
    device: str = "cpu"
    cuda_devices: str = ""
    samples_per_class: int = 3000
    random_seed: int = 42
    max_classes: int | None = None
    test_mode: bool = False
    methods: str | None = None
    align_mode: str = "min"
    query_per_class: int = 2
    ts2vec_ckpt: str | None = None
    cost_ckpt: str | None = None
    timesfm_dir: str | None = None
    chronos2_dir: str | None = None
    timer_dir: str | None = None
    sundial_dir: str | None = None
    moirai_dir: str | None = None
    timemoe_dir: str | None = None
    tabpfn_use: bool = False
    base_mode: str = "effective"
    ab_alpha: float = 1.0
    ab_b_min: float = 2.0
    ab_b_max: float = 20.0
    ab_c: float = 5.0
    ab_eps: float = 1e-12
    gain_mode: str = "binary"


def _keys(classes: Dict[int, List[np.ndarray]]) -> list[Key]:
    return [(cid, idx) for cid, items in classes.items() for idx in range(len(items))]


def _method_filter(all_names: list[str], spec: str | None) -> list[str]:
    if not spec:
        return all_names
    requested = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in requested if name not in all_names]
    if unknown:
        raise KeyError(f"Unknown methods: {unknown}. Available: {all_names}")
    return requested


def _needs_embedding_bank(methods: str | None, bank_name: str) -> bool:
    if not methods:
        return True
    prefix = f"{bank_name}_"
    return any(name.strip().startswith(prefix) for name in methods.split(",") if name.strip())


def _requested_embedding_banks(methods: str | None) -> set[str]:
    if not methods:
        return set()
    banks = {"TS2Vec", "CoST", "TimesFM", "Chronos2", "Timer", "Sundial", "Moirai", "TimeMoE", "TabPFN"}
    requested = set()
    for name in (item.strip() for item in methods.split(",") if item.strip()):
        for bank in banks:
            if name.startswith(f"{bank}_"):
                requested.add(bank)
    return requested


def _save_distance_cache(path: str, cache: Dict[str, list[dict]]) -> None:
    if not path:
        return
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    for method, rows in cache.items():
        with (out / f"{method}.pkl").open("wb") as f:
            pickle.dump({method: rows}, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_ucr_r(config: UCRRConfig) -> pd.DataFrame:
    max_classes = config.max_classes
    samples_per_class = config.samples_per_class
    if config.test_mode:
        max_classes = max_classes or 2
        samples_per_class = min(samples_per_class, 10)

    dataset = load_ucr_r(
        config.data_dir,
        samples_per_class=samples_per_class,
        random_seed=config.random_seed,
        max_classes=max_classes,
    )

    emb_cfg = EmbeddingConfig(
        device=config.device,
        cuda_devices=config.cuda_devices,
        ts2vec_ckpt=config.ts2vec_ckpt if _needs_embedding_bank(config.methods, "TS2Vec") else None,
        cost_ckpt=config.cost_ckpt if _needs_embedding_bank(config.methods, "CoST") else None,
        timesfm_dir=config.timesfm_dir if _needs_embedding_bank(config.methods, "TimesFM") else None,
        chronos2_dir=config.chronos2_dir if _needs_embedding_bank(config.methods, "Chronos2") else None,
        timer_dir=config.timer_dir if _needs_embedding_bank(config.methods, "Timer") else None,
        sundial_dir=config.sundial_dir if _needs_embedding_bank(config.methods, "Sundial") else None,
        moirai_dir=config.moirai_dir if _needs_embedding_bank(config.methods, "Moirai") else None,
        timemoe_dir=config.timemoe_dir if _needs_embedding_bank(config.methods, "TimeMoE") else None,
        use_tabpfn=config.tabpfn_use,
    )
    banks = build_embedding_banks(dataset.classes, emb_cfg)
    missing_banks = sorted(_requested_embedding_banks(config.methods) - set(banks))
    if missing_banks:
        raise RuntimeError(
            "Requested embedding methods could not be built: "
            f"{missing_banks}. Check checkpoint/model paths and file compatibility."
        )
    registry = build_ucr_r_registry(banks)
    methods = _method_filter(registry.names(), config.methods)

    rng = np.random.RandomState(config.random_seed)
    all_keys = _keys(dataset.classes)
    rows: list[dict] = []
    distance_cache: Dict[str, list[dict]] = {method: [] for method in methods}

    for method_name in methods:
        method = registry.get(method_name)
        query_metrics: list[dict[str, float]] = []
        for class_id, items in dataset.classes.items():
            if not items:
                continue
            query_count = min(config.query_per_class, len(items))
            query_indices = rng.choice(len(items), size=query_count, replace=False).tolist()
            for query_index in query_indices:
                query_key = (class_id, query_index)
                candidate_keys = [key for key in all_keys if key != query_key]
                labels = [1 if key[0] == class_id else 0 for key in candidate_keys]
                scores: list[float] = []
                for candidate_key in candidate_keys:
                    if method.input_type == "key":
                        try:
                            score = method.func(query_key, candidate_key)
                        except KeyError:
                            score = float("inf")
                    else:
                        query_series = dataset.classes[query_key[0]][query_key[1]]
                        candidate_series = dataset.classes[candidate_key[0]][candidate_key[1]]
                        if method.requires_equal_length:
                            query_series, candidate_series = align_pair(query_series, candidate_series, mode=config.align_mode)
                        try:
                            score = method.func(query_series, candidate_series)
                        except Exception:
                            score = float("inf")
                    if not np.isfinite(score):
                        score = float("inf")
                    scores.append(float(score))
                query_metrics.append(
                    ranking_summary(
                        scores,
                        labels,
                        ks=(1, 3, 5, 10),
                        reverse=False,
                        base_mode=config.base_mode,
                        ab_alpha=config.ab_alpha,
                        ab_b_min=config.ab_b_min,
                        ab_b_max=config.ab_b_max,
                        ab_c=config.ab_c,
                        ab_eps=config.ab_eps,
                        gain_mode=config.gain_mode,
                    )
                )
                distance_cache[method_name].append(
                    {
                        "query_key": query_key,
                        "candidate_keys": candidate_keys,
                        "distances": scores,
                        "labels": labels,
                    }
                )

        if query_metrics:
            frame = pd.DataFrame(query_metrics)
            row = {"method": method_name}
            row.update({col: float(frame[col].mean()) for col in frame.columns})
            rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty and "MAP" in result:
        result = result.sort_values("MAP", ascending=False).reset_index(drop=True)

    if config.out_csv:
        out = Path(config.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
    _save_distance_cache(config.distance_cache_path, distance_cache)
    return result
