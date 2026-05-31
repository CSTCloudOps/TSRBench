from __future__ import annotations

import importlib.util
import platform
import sys


REQUIRED = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "matplotlib",
    "openpyxl",
]

OPTIONAL = [
    "transformers",
    "huggingface_hub",
    "tabpfn",
    "fastdtw",
    "dtw",
]


def status(name: str) -> str:
    return "ok" if importlib.util.find_spec(name) else "missing"


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print("\nrequired:")
    missing = []
    for name in REQUIRED:
        item_status = status(name)
        print(f"  {name}: {item_status}")
        if item_status != "ok":
            missing.append(name)
    print("\noptional:")
    for name in OPTIONAL:
        print(f"  {name}: {status(name)}")
    if missing:
        raise SystemExit(f"Missing required packages: {missing}")


if __name__ == "__main__":
    main()

