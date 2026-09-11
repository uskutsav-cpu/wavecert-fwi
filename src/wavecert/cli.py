from __future__ import annotations

import argparse
import json

from wavecert.experiments.ablation import run_surrogate_quality_ablation
from wavecert.experiments.runner import run_demo
from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.physics.verification import verify_reference_problem


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wavecert", description="WaveCert-FWI research harness")
    sub = p.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the self-contained certified Helmholtz demo")
    demo.add_argument("--output", default="results/demo", help="output directory")
    demo.add_argument("--no-exact-validation", action="store_true")

    ab = sub.add_parser("ablation", help="sweep surrogate quality and fallback demand")
    ab.add_argument("--output", default="results/ablation", help="output directory")

    verify = sub.add_parser("verify-reference", help="run dot-product, FD, and Taylor tests")
    verify.add_argument("--trials", type=int, default=5)
    verify.add_argument("--seed", type=int, default=20260910)

    train = sub.add_parser("train-fno", help="train the forward-only Phase-2 FNO baseline")
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--train", type=int, default=256)
    train.add_argument("--val", type=int, default=48)
    train.add_argument("--test", type=int, default=80)
    train.add_argument("--shape", type=int, default=16)
    train.add_argument("--width", type=int, default=20)
    train.add_argument("--modes", type=int, default=5)
    train.add_argument("--depth", type=int, default=3)
    train.add_argument("--seed", type=int, default=20260910)

    failure = sub.add_parser("failure-study", help="run held-out forward-vs-gradient study")
    failure.add_argument("--cases", type=int, default=80)
    failure.add_argument("--seed", type=int, default=20260930)

    cert = sub.add_parser("certificate-study", help="run the Phase-4 deterministic certificate study")
    cert.add_argument("--cases", type=int, default=80)
    cert.add_argument("--seed", type=int, default=20260930)

    cal = sub.add_parser("calibrate-certificate", help="run Phase-5 split-conformal calibration")
    cal.add_argument("--alpha", type=float, default=0.10)

    adaptive = sub.add_parser("adaptive-study", help="run Phase-6 adaptive fallback study")
    adaptive.add_argument("--cases", type=int, default=30)
    adaptive.add_argument("--seed", type=int, default=20261010)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        summary = run_demo(args.output, validate_exact=not args.no_exact_validation)
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "ablation":
        rows = run_surrogate_quality_ablation(args.output)
        print(json.dumps(rows, indent=2))
        return 0
    if args.command == "verify-reference":
        problem = build_synthetic_problem(
            shape=(18, 18), frequencies_hz=(3.0,), n_sources=1, n_receivers=18
        )
        block = problem.blocks[0]
        report = verify_reference_problem(
            problem.physics,
            problem.m0,
            problem.physics.source(block.source_index),
            block.observed,
            problem.geometry.receiver_indices,
            block.frequency_hz,
            seed=args.seed,
            n_trials=args.trials,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0
    if args.command == "train-fno":
        from wavecert.experiments.train_fno import run_training
        from wavecert.surrogates.fno import FNOConfig

        summary = run_training(
            shape=(args.shape, args.shape),
            n_train=args.train,
            n_val=args.val,
            n_test=args.test,
            epochs=args.epochs,
            seed=args.seed,
            config=FNOConfig(
                modes_z=args.modes,
                modes_x=args.modes,
                width=args.width,
                depth=args.depth,
            ),
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "failure-study":
        from wavecert.experiments.failure_study import run_failure_study

        summary = run_failure_study(n_cases=args.cases, seed=args.seed)
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "certificate-study":
        from wavecert.experiments.certificate_study import run_certificate_study

        summary = run_certificate_study(n_cases=args.cases, seed=args.seed)
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "calibrate-certificate":
        from wavecert.experiments.certificate_calibration import run_certificate_calibration

        summary = run_certificate_calibration(alpha=args.alpha)
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "adaptive-study":
        from wavecert.experiments.adaptive_study import run_adaptive_study

        summary = run_adaptive_study(n_cases=args.cases, seed=args.seed)
        print(json.dumps(summary, indent=2))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
