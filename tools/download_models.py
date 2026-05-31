from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class ModelSpec:
    local_dir: str
    candidates: tuple[str, ...]


MODEL_SPECS = {
    "timesfm": ModelSpec("timesfm-2.0-500m-pytorch", ("google/timesfm-2.0-500m-pytorch",)),
    "chronos2": ModelSpec("chronos-2", ("amazon/chronos-2",)),
    "timer": ModelSpec("timer", ("thuml/timer-base-84m", "thuml/timer-large-84m")),
    "sundial": ModelSpec("sundial", ("thuml/sundial-base-128m",)),
    "moirai": ModelSpec("moirai", ("Salesforce/moirai-moe-1.0-R-base", "Salesforce/moirai-1.0-R-base")),
    "timemoe": ModelSpec("time-moe", ("Maple728/TimeMoE-50M", "Maple728/TimeMoE-200M")),
    "tabpfn": ModelSpec("tabpfn", ("Prior-Labs/TabPFN-v2-reg",)),
}


def parse_models(value: str) -> Iterable[str]:
    if value.strip().lower() == "all":
        return MODEL_SPECS.keys()
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in MODEL_SPECS]
    if unknown:
        raise SystemExit(f"Unknown model names: {unknown}. Available: {sorted(MODEL_SPECS)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretrained models used by TSRBench.")
    parser.add_argument("--models", default="all", help="all or comma-separated names")
    parser.add_argument("--pretrained_root", default="pretrained_models")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.pretrained_root)
    root.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for name in parse_models(args.models):
        spec = MODEL_SPECS[name]
        out_dir = root / spec.local_dir
        if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
            print(f"[skip] {name}: {out_dir}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        last_error = None
        for repo_id in spec.candidates:
            try:
                print(f"[download] {name}: {repo_id} -> {out_dir}")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(out_dir),
                    token=args.token,
                    revision=args.revision,
                    resume_download=True,
                )
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                print(f"[failed] {repo_id}: {exc}")
        if last_error is not None:
            failures.append((name, str(last_error)))

    if failures:
        for name, error in failures:
            print(f"[error] {name}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

