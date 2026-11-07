"""Prospective freeze, selected calibration, and fresh-index confirmation stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wavecert.experiments.gradient_confirmation import (
    freeze_selected_checkpoint,
    run_gradient_confirmation,
    run_selected_calibration,
    validate_confirmation_plan,
)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=root / "configs/gradient_confirmation_plan.json")
    commands = parser.add_subparsers(dest="stage", required=True)
    commands.add_parser("check-plan", help="Validate prospective folds/receipts without predictions")
    freeze = commands.add_parser("freeze-selection", help="Stage B; all training candidates must be complete")
    freeze.add_argument("--training", type=Path, required=True)
    freeze.add_argument("--training-source-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    calibration = commands.add_parser("calibrate-selected", help="Stage C; requires immutable Stage B first")
    calibration.add_argument("--selection-freeze", type=Path, required=True)
    calibration.add_argument("--dataset", type=Path, default=root / "data/reference/subsurfacegen_parent_split")
    calibration.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Stage D; requires both immutable freeze stages")
    evaluate.add_argument("--selection-freeze", type=Path, required=True)
    evaluate.add_argument("--calibration-freeze", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "check-plan":
        plan = validate_confirmation_plan(args.plan, root)
        print(json.dumps({"status": "validated without predictions", "families": [
            {"name": f["name"], "population": f["analysis_population_size"],
             "new_models": len(f["confirmation_selected_indices"])} for f in plan["families"]]}, indent=2))
    elif args.stage == "freeze-selection":
        result = freeze_selected_checkpoint(plan_path=args.plan, repository_root=root,
            training_dir=args.training, training_source_root=args.training_source_root, output_dir=args.output)
        print(json.dumps({"stage": result["stage"], "selected_candidate": result["selected_candidate"]}, indent=2))
    elif args.stage == "calibrate-selected":
        result = run_selected_calibration(freeze_dir=args.selection_freeze, repository_root=root,
            dataset_dir=args.dataset, output_dir=args.output)
        print(json.dumps({"stage": result["stage"]}, indent=2))
    else:
        result = run_gradient_confirmation(plan_path=args.plan, repository_root=root,
            selection_dir=args.selection_freeze, calibration_dir=args.calibration_freeze,
            data_dir=args.data, output_dir=args.output)
        print(json.dumps({"status": result["status"], "blocks": result["n_blocks"],
                          "model_checkpoint_evaluations": result["n_model_checkpoint_evaluations"]}, indent=2))


if __name__ == "__main__":
    main()
