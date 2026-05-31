from __future__ import annotations

import argparse

from _bootstrap import ensure_repo_root, repo_path

ensure_repo_root()

from experiments import UCRRConfig, run_ucr_r


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TSRBench UCR-R retrieval experiment.")
    parser.add_argument("--data_dir", default=repo_path("data", "UCR-R"))
    parser.add_argument("--distance_cache_path", default=repo_path("artifacts", "main_distance_cache"))
    parser.add_argument("--out_csv", default=repo_path("results", "ucr_r_metrics.csv"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cuda_devices", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples_per_class", type=int, default=3000)
    parser.add_argument("--query_per_class", type=int, default=2)
    parser.add_argument("--max_classes", type=int, default=2)
    parser.add_argument("--methods", help="Comma-separated method names. Defaults to all available methods.")
    parser.add_argument("--align_mode", default="min", choices=["min", "max", "truncate"])
    parser.add_argument("--base_mode", default="effective", choices=["effective", "pool"])
    parser.add_argument("--ab_alpha", type=float, default=1.0)
    parser.add_argument("--ab_b_min", type=float, default=2.0)
    parser.add_argument("--ab_b_max", type=float, default=20.0)
    parser.add_argument("--ab_c", type=float, default=5.0)
    parser.add_argument("--ab_eps", type=float, default=1e-12)
    parser.add_argument("--gain_mode", default="binary", choices=["binary", "graded"])
    parser.add_argument("--smoke", action="store_true", default=True, help="Run a small CPU-friendly subset.")
    parser.add_argument("--full", action="store_true", help="Run the full included UCR-R dataset.")
    parser.add_argument("--tabpfn_use", action="store_true")
    parser.add_argument("--ts2vec_ckpt", default=repo_path("models", "ts2vec", "ts2vec", "ts2vec_model.pkl"))
    parser.add_argument("--cost_ckpt", default=repo_path("models", "CoST", "cost_ckpt", "cost_ckpt.pkl"))
    parser.add_argument("--timesfm_dir", default=repo_path("pretrained_models", "timesfm-2.0-500m-pytorch"))
    parser.add_argument("--chronos2_dir", default=repo_path("pretrained_models", "chronos-2"))
    parser.add_argument("--timer_dir", default=repo_path("pretrained_models", "timer"))
    parser.add_argument("--sundial_dir", default=repo_path("pretrained_models", "sundial"))
    parser.add_argument("--moirai_dir", default=repo_path("pretrained_models", "moirai"))
    parser.add_argument("--timemoe_dir", default=repo_path("pretrained_models", "time-moe"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = UCRRConfig(
        data_dir=str(args.data_dir),
        out_csv=str(args.out_csv),
        distance_cache_path=str(args.distance_cache_path),
        device=args.device,
        cuda_devices=args.cuda_devices,
        samples_per_class=args.samples_per_class,
        random_seed=args.seed,
        max_classes=None if args.full else args.max_classes,
        test_mode=args.smoke and not args.full,
        methods=args.methods,
        align_mode=args.align_mode,
        query_per_class=args.query_per_class,
        ts2vec_ckpt=str(args.ts2vec_ckpt) if args.ts2vec_ckpt else None,
        cost_ckpt=str(args.cost_ckpt) if args.cost_ckpt else None,
        timesfm_dir=str(args.timesfm_dir) if args.timesfm_dir else None,
        chronos2_dir=str(args.chronos2_dir) if args.chronos2_dir else None,
        timer_dir=str(args.timer_dir) if args.timer_dir else None,
        sundial_dir=str(args.sundial_dir) if args.sundial_dir else None,
        moirai_dir=str(args.moirai_dir) if args.moirai_dir else None,
        timemoe_dir=str(args.timemoe_dir) if args.timemoe_dir else None,
        tabpfn_use=args.tabpfn_use,
        base_mode=args.base_mode,
        ab_alpha=args.ab_alpha,
        ab_b_min=args.ab_b_min,
        ab_b_max=args.ab_b_max,
        ab_c=args.ab_c,
        ab_eps=args.ab_eps,
        gain_mode=args.gain_mode,
    )
    result = run_ucr_r(config)
    print(result.to_string(index=False) if not result.empty else "No results were produced.")


if __name__ == "__main__":
    main()
