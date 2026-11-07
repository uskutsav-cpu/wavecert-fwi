"""Benchmark exact receiver-map controls without training or neural inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from wavecert.experiments.receiver_controls import measure_receiver_controls
from wavecert.provenance import environment_record, source_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/receiver_oracle_controls")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    provenance = environment_record(root)
    cases = [measure_receiver_controls(sources=s, receivers=r, seed=args.seed, repeats=args.repeats)
             for s, r in ((2, 12), (2, 6), (2, 2), (6, 12), (6, 6), (6, 2),
                          (12, 12), (12, 6), (12, 2))]
    end_hash = source_digest(root)
    unchanged = end_hash == provenance["source_sha256"]
    summary = {
        "schema_version": 1,
        "study_status": "complete" if unchanged else "stale-source-changed-during-run",
        "source_unchanged_during_run": unchanged,
        "source_sha256_at_end": end_hash,
        "scope": "Exact discrete reference controls; no neural model, optimization, or field-scale acceleration study.",
        "formulas_per_frequency": {
            "ordinary_exact_gradient_rhs": "2*s",
            "optimized_exact_gradient_rhs": "min(2*s, r_unique+s-covered_sources)",
            "exact_receiver_map_direction_rhs": "2*r_unique",
            "initial_norm_only_maps_rhs": "2*r_unique, plus two SVDs",
        },
        "limitations": [
            "One seeded model/direction per geometry, not independent geological replication.",
            "Only gradient controls construct a complete model-space gradient; the exact directional oracle supplies less information.",
            f"{args.repeats} cold repetitions and rotated order provide descriptive local timings, not statistical runtime or inversion acceleration evidence.",
            "Receiver maps occupy dense receiver-by-grid storage and require matrix multiplications; RHS counts do not capture all computation or memory.",
            "Ordinary floating-point sparse solves, not formally verified rounding enclosures.",
            "Transpose reciprocity is checked for this reference operator; no claim for Devito, field data, or another backend.",
            "Negative directional derivatives do not imply decrease for an unchecked finite or projected step.",
            "No cost for generating a useful proposal direction, neural inference, repairs, or line searches is included.",
        ],
        "all_checks_pass": all(case["all_checks_pass"] for case in cases),
        "cases": cases, "provenance": provenance,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    rows = []
    for case in cases:
        for method in case["methods"]:
            rows.append({
                "sources": case["sources"], "receivers": case["receivers"],
                "covered_sources": case["covered_sources"], "method": method["method"],
                "expected_rhs": method["expected_rhs"], "executed_rhs": method["executed_rhs"],
                "factorizations": method["costs"]["factorizations"],
                "svd_evaluations": method["costs"]["svd_evaluations"],
                "median_elapsed_seconds": method["median_elapsed_seconds"],
                "complete_gradient_constructed": method["complete_gradient_constructed"],
                "objective_absolute_error": method["maximum_objective_absolute_error"],
                "directional_absolute_error": method["maximum_directional_absolute_error"],
                "gradient_l2_error": method["maximum_gradient_l2_error"],
                "finite_difference_best_relative_error": case["finite_difference_best_relative_error"],
            })
    with (output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"cases": len(cases), "study_status": summary["study_status"],
                      "all_checks_pass": summary["all_checks_pass"],
                      "worst_best_finite_difference_error": max(c["finite_difference_best_relative_error"] for c in cases)},
                     indent=2, allow_nan=False))
    if not unchanged:
        raise RuntimeError("source changed during run; results are explicitly stale and must be rerun")


if __name__ == "__main__":
    main()
