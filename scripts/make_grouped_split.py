"""Create a new parent-isolated research manifest; retain original split provenance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from wavecert.data.splits import audit_split_manifest, grouped_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    with Path(args.input).open(newline="") as stream:
        rows = grouped_split_manifest(list(csv.DictReader(stream)), seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = audit_split_manifest(rows)
    audit["protocol"] = "new parent-isolated research re-split; source_split preserves original benchmark labels"
    audit["seed"] = args.seed
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({role: data["unique_parent_models"] for role, data in audit["roles"].items()}, indent=2))


if __name__ == "__main__":
    main()
