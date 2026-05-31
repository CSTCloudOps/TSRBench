from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_PATHS = [
    "artifacts",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove generated TSRBench artifacts.")
    parser.add_argument("--paths", nargs="*", default=DEFAULT_PATHS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if root not in path.parents and path != root:
            raise SystemExit(f"Refusing to remove path outside repository: {path}")
        if path.exists():
            print(f"[remove] {path}")
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch()


if __name__ == "__main__":
    main()
