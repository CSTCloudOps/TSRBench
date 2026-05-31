from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalDataset:
    name: str
    classes: Dict[int, List[np.ndarray]]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_series(self) -> int:
        return sum(len(items) for items in self.classes.values())


def _read_ucr_r_tsv(path: Path) -> Dict[int, List[np.ndarray]]:
    df = pd.read_csv(path, sep="\t", header=None)
    classes: Dict[int, List[np.ndarray]] = {}
    for _, row in df.iterrows():
        try:
            class_id = int(float(row.iloc[0]))
        except Exception:
            continue
        values = pd.to_numeric(row.iloc[1:], errors="coerce").to_numpy(dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        classes.setdefault(class_id, []).append(values)
    return classes


def load_ucr_r(
    data_dir: str | Path,
    samples_per_class: int = 3000,
    random_seed: int = 42,
    max_classes: int | None = None,
) -> RetrievalDataset:
    """Load the UCR-R benchmark layout used in TSRBench.

    The release layout is nested by original UCR dataset and contains TSV files
    whose first column is the class label and remaining columns are time-series
    values.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"UCR-R directory not found: {root}")

    merged: Dict[int, List[np.ndarray]] = {}
    for path in sorted(root.rglob("*.tsv")):
        for class_id, series_list in _read_ucr_r_tsv(path).items():
            merged.setdefault(class_id, []).extend(series_list)
        if max_classes is not None and len(merged) >= max_classes:
            keep = sorted(merged)[:max_classes]
            merged = {cid: merged[cid] for cid in keep}
            break

    if not merged:
        raise RuntimeError(f"No UCR-R series found under {root}")

    rng = np.random.RandomState(random_seed)
    sampled: Dict[int, List[np.ndarray]] = {}
    for class_id in sorted(merged):
        items = merged[class_id]
        if samples_per_class and len(items) > samples_per_class:
            idx = rng.choice(len(items), size=samples_per_class, replace=False)
            sampled[class_id] = [items[i] for i in idx.tolist()]
        else:
            sampled[class_id] = list(items)
        logger.info("UCR-R class %s: %d series", class_id, len(sampled[class_id]))

    return RetrievalDataset(name="UCR-R", classes=sampled)

