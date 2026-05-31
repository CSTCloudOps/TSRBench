from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CURCASensor:
    name: str
    label: int
    series: np.ndarray


@dataclass(frozen=True)
class CURCAFile:
    path: Path
    sensors: List[CURCASensor]

    @property
    def has_anomaly(self) -> bool:
        return any(sensor.label == 1 for sensor in self.sensors)


def _read_file(path: Path) -> CURCAFile:
    df = pd.read_csv(path, sep="\t")
    sensors: List[CURCASensor] = []
    for _, row in df.iterrows():
        name = str(row.iloc[0])
        try:
            label = int(float(row.iloc[1]))
        except Exception:
            continue
        values = pd.to_numeric(row.iloc[2:], errors="coerce").to_numpy(dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size:
            sensors.append(CURCASensor(name=name, label=label, series=values))
    return CURCAFile(path=path, sensors=sensors)


def load_cu_rca_files(data_dir: str | Path, max_files: int | None = None) -> List[CURCAFile]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"CU-RCA directory not found: {root}")
    files: List[CURCAFile] = []
    for path in sorted(root.rglob("*.tsv")):
        item = _read_file(path)
        if len(item.sensors) >= 2:
            files.append(item)
        if max_files is not None and len(files) >= max_files:
            break
    if not files:
        raise RuntimeError(f"No CU-RCA TSV files found under {root}")
    return files

