"""Separate fixed coarse-corrected POD freeze and five-model confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wavecert.experiments.coarse_pod_confirmation import (
    freeze_coarse_pod_comparator,
    run_five_model_confirmation,
    validate_coarse_addendum,
)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum", type=Path, default=root / "configs/coarse_corrected_pod_confirmation_addendum.json")
    parser.add_argument("--pod-addendum", type=Path, default=root / "configs/pod_confirmation_addendum.json")
    parser.add_argument("--plan", type=Path, default=root / "configs/gradient_confirmation_plan.json")
    commands = parser.add_subparsers(dest="stage", required=True)
    commands.add_parser("check-plan", help="Validate all prospective plan hashes; no outcomes")
    for name in ("freeze", "evaluate-five"):
        command = commands.add_parser(name)
        command.add_argument("--selection-freeze", type=Path, required=True)
        command.add_argument("--calibration-freeze", type=Path, required=True)
        command.add_argument("--pod-freeze", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "evaluate-five":
            command.add_argument("--coarse-freeze", type=Path, required=True)
            command.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "check-plan":
        result = validate_coarse_addendum(args.addendum, root)
        report = {"status": "validated without outcomes", "model_labels": result["model_labels"]}
    else:
        common = dict(addendum_path=args.addendum, repository_root=root, selection_dir=args.selection_freeze,
            calibration_dir=args.calibration_freeze, pod_dir=args.pod_freeze, output_dir=args.output)
        if args.stage == "freeze":
            result = freeze_coarse_pod_comparator(**common)
            report = {"stage": result["stage"], "operator_signature": result["operator_signature"]}
        else:
            result = run_five_model_confirmation(**common, pod_addendum_path=args.pod_addendum, plan_path=args.plan,
                coarse_dir=args.coarse_freeze, data_dir=args.data)
            report = {"status": result["status"], "n_blocks": result["n_blocks"]}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
