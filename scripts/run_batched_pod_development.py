"""Audit or run the fixed validation-only POD batching development extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wavecert.experiments.batched_pod_development import (
    PROTOCOL_SHA256,
    load_development_inputs,
    run_development,
)
from wavecert.provenance import source_digest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/batched_pod_development_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("results/batched_pod_development"))
    parser.add_argument("--compute-context", default="unspecified")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-inputs", action="store_true", help="No model construction, evaluation, timing or output writes")
    mode.add_argument("--run-development", action="store_true", help="Only after source review/root GO and other compute idle")
    args = parser.parse_args()
    protocol = args.protocol if args.protocol.is_absolute() else root / args.protocol
    if args.check_inputs:
        _, _, _, _, audit = load_development_inputs(protocol, root)
        report = dict(status="inputs verified; no evaluation/timing performed", protocol_sha256=PROTOCOL_SHA256,
                      source_sha256=source_digest(root), verified_file_sha256=audit["verified_file_sha256"],
                      unchanged_production_sources=audit["unchanged_production_sources"],
                      verified_execution_modules=audit["verified_execution_modules"],
                      n_validation_slices=8, n_validation_parents=4,
                      calibration_or_test_arrays_loaded=False, external_test_arrays_loaded=False)
    else:
        output = args.output if args.output.is_absolute() else root / args.output
        summary = run_development(protocol, output, root=root, compute_context=args.compute_context)
        report = dict(status=summary["status"], checks=summary["checks"],
                      development_recommendation=summary["development_recommendation"])
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
