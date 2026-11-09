"""Check inputs or reproduce the fixed exploratory coarse-adjoint validation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wavecert.experiments.coarse_adjoint_validation import (
    load_coarse_adjoint_inputs,
    run_coarse_adjoint_validation,
)
from wavecert.provenance import source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/coarse_adjoint_validation_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("results/coarse_adjoint_validation"))
    parser.add_argument("--check-inputs", action="store_true", help="No predictions, fine solves or output writes")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    if args.check_inputs:
        raw = path.read_bytes()
        _, _, audit = load_coarse_adjoint_inputs(json.loads(raw), root)
        print(json.dumps({"status": "inputs verified; no pilot evaluation performed",
            "protocol_sha256": hashlib.sha256(raw).hexdigest(), "source_sha256": source_digest(root),
            "verified_file_sha256": audit["verified_file_sha256"], "n_validation_slices": 8, "n_validation_parents": 4,
            "calibration_or_test_arrays_loaded": False, "external_test_arrays_loaded": False}, indent=2, allow_nan=False))
    else:
        output = args.output if args.output.is_absolute() else root / args.output
        summary = run_coarse_adjoint_validation(path, output, root=root)
        print(json.dumps({"status": summary["status"], "checks": summary["checks"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
