"""Inspect or fit the frozen eight-candidate training-only POD control study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wavecert.experiments.pod_development import (
    load_teacher_inputs,
    run_pod_development,
    validate_pod_protocol,
)
from wavecert.provenance import source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/pod_development_protocol.json"))
    parser.add_argument("--teacher-study", type=Path, help="Read-only location override for the exact pinned teacher archives")
    parser.add_argument("--output", type=Path, default=Path("results/pod_development"))
    parser.add_argument("--check-inputs", action="store_true", help="Verify inputs only; no SVD, fitting, validation or output writes")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_pod_protocol(protocol)
    teacher = args.teacher_study or Path(protocol["teacher_study"])
    teacher = teacher if teacher.is_absolute() else root / teacher
    if args.check_inputs:
        _, _, audit = load_teacher_inputs(protocol, teacher)
        print(json.dumps({
            "status": "input-check only; no POD fit or validation performed",
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "source_sha256": source_digest(root),
            "roles": {key: {"parents": len(value["parent_ids"]), "slices": len(value["slice_ids"]),
                            "state_examples": value["state_examples"]} for key, value in audit["identities"].items()},
            "calibration_or_test_arrays_loaded": False, "external_test_arrays_loaded": False,
        }, indent=2, allow_nan=False))
        return
    output = args.output if args.output.is_absolute() else root / args.output
    summary = run_pod_development(protocol_path, teacher, output, root=root)
    print(json.dumps({"status": summary["status"], "selection": summary["selection"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
