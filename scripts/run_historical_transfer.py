"""Freeze/run fixed historical test reuse; no learning or pristine-test claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wavecert.experiments.historical_transfer import (
    freeze_historical_transfer,
    run_historical_transfer,
    validate_historical_protocol,
)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=root / "configs/historical_transfer_protocol.json")
    parser.add_argument("--dataset", type=Path, default=root / "data/reference/subsurfacegen_parent_split")
    commands = parser.add_subparsers(dest="stage", required=True)
    commands.add_parser("check-protocol", help="Metadata/manifests only; never open historical test NPZ arrays")
    freeze = commands.add_parser("freeze", help="Audit old guards in clean subprocess, then write new phase freeze")
    freeze.add_argument("--selection-freeze", type=Path, required=True)
    freeze.add_argument("--calibration-freeze", type=Path, required=True)
    freeze.add_argument("--pod-freeze", type=Path, required=True)
    freeze.add_argument("--coarse-freeze", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("evaluate", help="Requires protocol/source approval and immutable historical freeze")
    run.add_argument("--historical-freeze", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "check-protocol":
        protocol, _, _ = validate_historical_protocol(args.protocol, root, dataset_dir=args.dataset)
        report = {"status": "validated metadata only; no test arrays/outcomes opened", "roles": {
            role: {key: cfg[key] for key in ("n_parents", "n_slices", "n_blocks")} for role, cfg in protocol["dataset"]["roles"].items()}}
    elif args.stage == "freeze":
        result = freeze_historical_transfer(protocol_path=args.protocol, repository_root=root, dataset_dir=args.dataset,
            selection_dir=args.selection_freeze, calibration_dir=args.calibration_freeze, pod_dir=args.pod_freeze,
            coarse_dir=args.coarse_freeze, output_dir=args.output)
        report = {"stage": result["stage"], "models": [model["label"] for model in result["models"]]}
    else:
        result = run_historical_transfer(freeze_dir=args.historical_freeze, repository_root=root,
            dataset_dir=args.dataset, output_dir=args.output)
        report = {"status": result["status"], "n_slice_model_aggregates": result["n_slice_model_aggregates"], "scope": result["scope"]}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
