"""Explicit classical POD comparator addendum, freeze, and joint confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wavecert.experiments.pod_confirmation import (
    freeze_pod_comparator,
    run_joint_pod_confirmation,
    validate_pod_addendum,
)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum", type=Path, default=root / "configs/pod_confirmation_addendum.json")
    parser.add_argument("--plan", type=Path, default=root / "configs/gradient_confirmation_plan.json")
    commands = parser.add_subparsers(dest="stage", required=True)
    check = commands.add_parser("audit", help="Read-only all-eight development/source/teacher audit; no predictions")
    check.add_argument("--teacher-study", type=Path, required=True)
    freeze = commands.add_parser("freeze", help="New immutable POD receipt after complete FNO Stage B")
    freeze.add_argument("--teacher-study", type=Path, required=True)
    freeze.add_argument("--selection-freeze", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate-joint", help="All four models; requires B, C and POD freezes")
    evaluate.add_argument("--selection-freeze", type=Path, required=True)
    evaluate.add_argument("--calibration-freeze", type=Path, required=True)
    evaluate.add_argument("--pod-freeze", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "audit":
        addendum, _, summary = validate_pod_addendum(args.addendum, root, teacher_study=args.teacher_study)
        result = {"status": "audited without predictions", "all_candidates": len(summary["candidates"]),
                  "selected_pod": addendum["selected_pod"]}
    elif args.stage == "freeze":
        result = freeze_pod_comparator(addendum_path=args.addendum, repository_root=root,
            teacher_study=args.teacher_study, selection_dir=args.selection_freeze, output_dir=args.output)
        result = {"stage": result["stage"], "selected_pod": result["selected_pod"]}
    else:
        result = run_joint_pod_confirmation(addendum_path=args.addendum, plan_path=args.plan, repository_root=root,
            selection_dir=args.selection_freeze, calibration_dir=args.calibration_freeze, pod_dir=args.pod_freeze,
            data_dir=args.data, output_dir=args.output)
        result = {"status": result["status"], "n_blocks": result["n_blocks"]}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
