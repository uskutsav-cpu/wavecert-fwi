from __future__ import annotations

import argparse
import json

from wavecert.experiments.ablation import run_surrogate_quality_ablation
from wavecert.experiments.runner import run_demo


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wavecert", description="WaveCert-FWI research harness")
    sub = p.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the self-contained certified Helmholtz demo")
    demo.add_argument("--output", default="results/demo", help="output directory")
    demo.add_argument("--no-exact-validation", action="store_true")

    ab = sub.add_parser("ablation", help="sweep surrogate quality and fallback demand")
    ab.add_argument("--output", default="results/ablation", help="output directory")
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
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
