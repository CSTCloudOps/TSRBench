from __future__ import annotations

import argparse

from _bootstrap import ensure_repo_root, repo_path

ensure_repo_root()

from experiments import compute_metrics_from_cache


def _parse_topk(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute UCR-R ranking metrics from cached distances.")
    parser.add_argument("--distance_cache_path", default=repo_path("artifacts", "main_distance_cache"))
    parser.add_argument("--topk", default="1,3,5,10")
    parser.add_argument("--out_csv", default=repo_path("results", "metrics_from_cache.csv"))
    parser.add_argument("--base_mode", default="effective", choices=["effective", "pool"])
    parser.add_argument("--ab_alpha", type=float, default=1.0)
    parser.add_argument("--ab_b_min", type=float, default=2.0)
    parser.add_argument("--ab_b_max", type=float, default=20.0)
    parser.add_argument("--ab_c", type=float, default=5.0)
    parser.add_argument("--ab_eps", type=float, default=1e-12)
    parser.add_argument("--gain_mode", default="binary", choices=["binary", "graded"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = compute_metrics_from_cache(
        distance_cache_path=str(args.distance_cache_path),
        out_csv=str(args.out_csv),
        topk=_parse_topk(args.topk),
        base_mode=args.base_mode,
        ab_alpha=args.ab_alpha,
        ab_b_min=args.ab_b_min,
        ab_b_max=args.ab_b_max,
        ab_c=args.ab_c,
        ab_eps=args.ab_eps,
        gain_mode=args.gain_mode,
    )
    print(result.to_string(index=False) if not result.empty else "No cached metrics were produced.")


if __name__ == "__main__":
    main()
