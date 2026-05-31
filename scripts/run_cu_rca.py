from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_repo_root, repo_path

ensure_repo_root()

from experiments import CURCAConfig, run_cu_rca


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TSRBench CU-RCA industrial retrieval experiment.")
    parser.add_argument("--data_dir", default=repo_path("data", "CU-RCA"))
    parser.add_argument("--cache_dir", default=repo_path("artifacts", "cu_rca_distance_cache"))
    parser.add_argument("--output", default=repo_path("results", "cu_rca_hit.json"))
    parser.add_argument("--methods", help="Comma-separated method names. Defaults to all available methods.")
    parser.add_argument("--max_files", type=int)
    parser.add_argument("--align_mode", default="truncate", choices=["min", "max", "truncate"])
    parser.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    parser.add_argument("--cuda_devices", default="", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = CURCAConfig(
        data_dir=str(args.data_dir),
        output=str(args.output),
        cache_dir=str(args.cache_dir),
        methods=args.methods,
        max_files=args.max_files,
        align_mode=args.align_mode,
    )
    result = run_cu_rca(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
