"""Audit inputs or run the fixed cold-query controls in an otherwise idle session."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wavecert.experiments.coarse_runtime_controls import load_runtime_inputs, run_runtime_controls
from wavecert.provenance import source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/coarse_runtime_controls.json"))
    parser.add_argument("--output", type=Path, default=Path("results/coarse_runtime_controls"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-inputs", action="store_true", help="No timing, predictions, fine solves or output writes")
    mode.add_argument("--run-timings", action="store_true", help="Execute after other compute is idle")
    parser.add_argument("--compute-context", default="unspecified", help="Record observed load/concurrent-work conditions")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = args.protocol if args.protocol.is_absolute() else root / args.protocol
    if args.check_inputs:
        raw = protocol.read_bytes()
        _, _, audit = load_runtime_inputs(json.loads(raw), root)
        print(json.dumps({"status": "inputs verified; no timings or evaluation executed",
            "protocol_sha256": hashlib.sha256(raw).hexdigest(), "source_sha256": source_digest(root),
            "verified_file_sha256": audit["verified_file_sha256"], "validation_protocol_sha256": audit["validation_protocol_sha256"]},
            indent=2, allow_nan=False))
        return
    output = args.output if args.output.is_absolute() else root / args.output
    result = run_runtime_controls(protocol, output, root=root, compute_context=args.compute_context)
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
