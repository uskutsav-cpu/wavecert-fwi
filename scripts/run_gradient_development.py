"""Run or inspect the frozen four-candidate train/validation-only comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wavecert.experiments.gradient_development import (
    load_development_roles,
    run_gradient_development,
    validate_protocol,
)
from wavecert.provenance import source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/gradient_development_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("results/gradient_development"))
    parser.add_argument("--check-inputs", action="store_true", help="Verify only the protocol and allowed inputs; no labels, training, or output writes")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    if args.check_inputs:
        protocol_bytes = protocol_path.read_bytes()
        protocol = json.loads(protocol_bytes)
        validate_protocol(protocol)
        roles, audit = load_development_roles(protocol, root)
        print(json.dumps({
            "status": "input-check-only; training not started",
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "source_sha256": source_digest(root),
            "roles": {role: {"raw_examples": len(value.arrays.models),
                              "parents": audit["parent_and_slice_audit"]["roles"][role]["unique_parent_models"]}
                      for role, value in roles.items()},
            "candidates": protocol["candidates"],
            "optimizer_updates_per_candidate": protocol["training"]["optimizer_updates"],
            "calibration_or_test_arrays_loaded": False,
            "external_test_arrays_loaded": False,
        }, indent=2, allow_nan=False))
        return
    output = args.output if args.output.is_absolute() else root / args.output
    result = run_gradient_development(protocol_path, output, root=root)
    print(json.dumps({"status": result["status"], "selection": result["selection"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
