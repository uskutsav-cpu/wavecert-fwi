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

    cert = sub.add_parser(
        "certificate-study", help="run the Phase-4 deterministic certificate study"
    )
    cert.add_argument("--cases", type=int, default=80)
    cert.add_argument("--seed", type=int, default=20260930)

    cal = sub.add_parser("calibrate-certificate", help="run Phase-5 split-conformal calibration")
    cal.add_argument("--alpha", type=float, default=0.10)

    adaptive = sub.add_parser("adaptive-study", help="run Phase-6 adaptive fallback study")
    adaptive.add_argument("--cases", type=int, default=30)
    adaptive.add_argument("--seed", type=int, default=20261010)

    end_to_end = sub.add_parser(
        "end-to-end-study", help="run Phase-7 complete inversion trajectories"
    )
    end_to_end.add_argument("--cases", type=int, default=4)
    end_to_end.add_argument("--iterations", type=int, default=8)
    end_to_end.add_argument("--seed", type=int, default=20261101)
    end_to_end.add_argument("--output", default="results/phase7")
    end_to_end.add_argument("--step-acceptance", choices=("surrogate", "exact-decrease", "exact-armijo"), default="surrogate")

    evidence = sub.add_parser("evidence-study", help="reproduce a cost-accounted, multi-policy development study")
    evidence.add_argument("--checkpoint", default="checkpoints/fno_phase2.pt")
    evidence.add_argument("--output", default="results/evidence_study")
    evidence.add_argument("--seeds", type=int, nargs="+", default=[731, 1907, 8101])
    evidence.add_argument("--cases-per-seed", type=int, default=4)
    evidence.add_argument("--iterations", type=int, default=12)
    evidence.add_argument("--calibration-cases", type=int, default=30)
    evidence.add_argument("--alpha", type=float, default=0.10)
    evidence.add_argument("--scenarios", nargs="+", default=["clean", "noise-5pct", "sparse-receivers", "frequency-shift"])
    evidence.add_argument("--step-acceptance", choices=("surrogate", "exact-decrease", "exact-armijo"), default="exact-decrease")
    evidence.add_argument("--threads", type=int, default=1)

    independent = sub.add_parser("verify-physics", help="check sparse physics against a spectral solver and analytic continuum solution")
    independent.add_argument("--output", default="results/independent_physics")

    production = sub.add_parser(
        "production-study", help="run Phase-8 production/Marmousi readiness study"
    )
    production.add_argument("--iterations", type=int, default=6)
    production.add_argument("--marmousi", default=None, help="optional public Marmousi binary path")

    ood = sub.add_parser("ood-study", help="run Phase-9 geology/OOD stress study")
    ood.add_argument("--cases-per-setting", type=int, default=5)
    ood.add_argument("--seed", type=int, default=20261120)

    real_train = sub.add_parser(
        "train-real-fno", help="train the stronger SubsurfaceGen real-geology FNO"
    )
    real_train.add_argument("--dataset", default="data/external/subsurfacegen-wavecert")
    real_train.add_argument("--checkpoint", default="checkpoints/fno_subsurfacegen_v2.pt")
    real_train.add_argument("--epochs", type=int, default=100)
    real_train.add_argument("--batch-size", type=int, default=32)
    real_train.add_argument("--patience", type=int, default=15)
    real_train.add_argument("--width", type=int, default=48)
    real_train.add_argument("--modes", type=int, default=12)
    real_train.add_argument("--depth", type=int, default=4)
    real_train.add_argument("--output", default="results/real_geology_v2")
    real_train.add_argument("--seed", type=int, default=20260911)
    real_train.add_argument("--threads", type=int, default=1)
    real_train.add_argument("--jacobian-weight", type=float, default=0.0)
    real_train.add_argument("--jacobian-seed", type=int, default=20261107)

    rw = sub.add_parser(
        "real-wavecert-study", help="run real-geology ID/OOD gradient and certificate study"
    )
    rw.add_argument("--dataset", default="data/external/subsurfacegen-wavecert")
    rw.add_argument("--checkpoint", default="checkpoints/fno_subsurfacegen_v2.pt")
    rw.add_argument("--alpha", type=float, default=0.10)
    rw.add_argument("--output", default="results/real_wavecert")

    devito = sub.add_parser("devito-study", help="actually execute Devito forward/adjoint/L-BFGS-B")
    devito.add_argument("--marmousi", required=True)
    devito.add_argument("--iterations", type=int, default=5)
    devito.add_argument("--output", default="results/devito_verified")

    openfwi = sub.add_parser(
        "openfwi-study", help="run WaveCert surrogate on an official OpenFWI velocity batch"
    )
    openfwi.add_argument("--velocity-file", required=True)
    openfwi.add_argument("--checkpoint", default="checkpoints/fno_subsurfacegen_v2.pt")
    openfwi.add_argument("--max-models", type=int, default=100)

    heuristics = sub.add_parser(
        "heuristic-baselines", help="compare random, periodic, forward-error and WaveCert fallback"
    )
    heuristics.add_argument("--cases", type=int, default=80)

    baselines = sub.add_parser(
        "baseline-study", help="repeat principal trajectory baselines across seeds"
    )
    baselines.add_argument("--cases", type=int, default=4)
    baselines.add_argument("--iterations", type=int, default=8)
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

        print(json.dumps(run_failure_study(n_cases=args.cases, seed=args.seed), indent=2))
        return 0
    if args.command == "certificate-study":
        from wavecert.experiments.certificate_study import run_certificate_study

        print(json.dumps(run_certificate_study(n_cases=args.cases, seed=args.seed), indent=2))
        return 0
    if args.command == "calibrate-certificate":
        from wavecert.experiments.certificate_calibration import run_certificate_calibration

        print(json.dumps(run_certificate_calibration(alpha=args.alpha), indent=2))
        return 0
    if args.command == "adaptive-study":
        from wavecert.experiments.adaptive_study import run_adaptive_study

        print(json.dumps(run_adaptive_study(n_cases=args.cases, seed=args.seed), indent=2))
        return 0
    if args.command == "end-to-end-study":
        from wavecert.experiments.end_to_end import run_end_to_end_study

        print(
            json.dumps(
                run_end_to_end_study(
                    n_cases=args.cases, iterations=args.iterations, seed=args.seed,
                    output_dir=args.output, step_acceptance=args.step_acceptance,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "evidence-study":
        from wavecert.experiments.evidence_study import EvidenceConfig, run_evidence_study

        config = EvidenceConfig(
            seeds=tuple(args.seeds), cases_per_seed=args.cases_per_seed,
            iterations=args.iterations, calibration_cases=args.calibration_cases,
            alpha=args.alpha, scenarios=tuple(args.scenarios),
            step_acceptance=args.step_acceptance, torch_threads=args.threads,
        )
        summary = run_evidence_study(checkpoint_path=args.checkpoint, output_dir=args.output, config=config)
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0
    if args.command == "verify-physics":
        from wavecert.physics.independent_verification import verify_physics_independently

        summary = verify_physics_independently(output_dir=args.output)
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0 if summary["passed"] else 1
    if args.command == "production-study":
        from wavecert.experiments.production_study import run_production_study

        print(
            json.dumps(
                run_production_study(iterations=args.iterations, marmousi_path=args.marmousi),
                indent=2,
            )
        )
        return 0
    if args.command == "ood-study":
        from wavecert.experiments.ood_study import run_ood_study

        print(
            json.dumps(
                run_ood_study(cases_per_setting=args.cases_per_setting, seed=args.seed), indent=2
            )
        )
        return 0
    if args.command == "train-real-fno":
        import torch

        from wavecert.experiments.real_geology_training import run_real_geology_training
        from wavecert.surrogates.fno import FNOConfig

        if args.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
        summary = run_real_geology_training(
            dataset_dir=args.dataset,
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            seed=args.seed,
            jacobian_weight=args.jacobian_weight,
            jacobian_seed=args.jacobian_seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            config=FNOConfig(
                modes_z=args.modes,
                modes_x=args.modes,
                width=args.width,
                depth=args.depth,
                padding=2,
            ),
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "real-wavecert-study":
        from wavecert.experiments.real_wavecert import run_real_wavecert_study

        print(
            json.dumps(
                run_real_wavecert_study(
                    dataset_dir=args.dataset,
                    checkpoint_path=args.checkpoint,
                    alpha=args.alpha,
                    output_dir=args.output,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "devito-study":
        from wavecert.experiments.devito_study import run_devito_study

        print(
            json.dumps(
                run_devito_study(
                    marmousi_path=args.marmousi,
                    optimizer_iterations=args.iterations,
                    output_dir=args.output,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "openfwi-study":
        from wavecert.experiments.openfwi_study import run_openfwi_study

        print(
            json.dumps(
                run_openfwi_study(
                    velocity_file=args.velocity_file,
                    checkpoint_path=args.checkpoint,
                    max_models=args.max_models,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "heuristic-baselines":
        from wavecert.experiments.heuristic_baselines import run_heuristic_baselines

        print(json.dumps(run_heuristic_baselines(n_cases=args.cases), indent=2))
        return 0
    if args.command == "baseline-study":
        from wavecert.experiments.baseline_study import run_baseline_study

        print(
            json.dumps(
                run_baseline_study(cases=args.cases, iterations=args.iterations),
                indent=2,
            )
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
