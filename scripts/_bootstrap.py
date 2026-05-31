from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def ensure_repo_root() -> None:
    for path in (str(REPO_ROOT),):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(REPO_ROOT)
