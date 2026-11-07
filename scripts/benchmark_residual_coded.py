"""Verify exact receiver-coded controls; timings are disabled unless requested."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavecert.experiments.residual_coded_controls import measure_residual_coded_controls
from wavecert.provenance import environment_record, source_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/residual_coded_controls")
    parser.add_argument("--timing-repeats", type=int, default=0,
                        help="0 verifies operation counts without reporting timings; positive values time cold-solver repetitions")
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    if args.timing_repeats < 0:
        raise ValueError("timing-repeats must be non-negative")
    root, output = Path(__file__).resolve().parents[1], Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be new or empty; previous evidence is preserved")
    output.mkdir(parents=True, exist_ok=True)
    provenance = environment_record(root)
    status = {"status": "running", "completed_geometries": 0, "timing_repeats": args.timing_repeats,
        "provenance": provenance}
    status_path = output / "run_status.json"
    status_path.write_text(json.dumps(status, indent=2, allow_nan=False)+"\n")
    cases = []
    try:
        for sources, receivers in ((2, 12), (2, 6), (2, 2), (6, 12), (6, 6), (6, 2),
                                  (12, 12), (12, 6), (12, 2)):
            cases.append(measure_residual_coded_controls(sources=sources, receivers=receivers,
                seed=args.seed, timing_repeats=args.timing_repeats))
            status["completed_geometries"] = len(cases)
            status_path.write_text(json.dumps(status, indent=2, allow_nan=False)+"\n")
        unchanged = source_digest(root) == provenance["source_sha256"]
        summary = {"schema_version": 1, "study_status": "complete" if unchanged else "stale-source-changed",
            "source_unchanged_during_run": unchanged, "source_sha256_at_end": source_digest(root),
            "scope": "Exact discrete reference operation-count controls; no neural or inversion performance claim.",
            "timing_status": "not measured" if args.timing_repeats == 0 else "descriptive local cold-solver invocation timings",
            "timing_repeats": args.timing_repeats, "all_checks_pass": all(c["all_checks_pass"] for c in cases),
            "formulas_per_frequency": {"ordinary_gradient_rhs": "2*s", "symmetry_gradient_rhs": "r_unique+s-covered_sources, only if A.T==A",
                "coded_gradient_rhs": "2*r_unique", "best_gradient_rhs": "min(2*s,2*r_unique,r_unique+s-covered_sources when symmetry is valid)",
                "exact_directional_maps_rhs": "2*r_unique", "initial_norm_only_maps_rhs": "2*r_unique plus two SVDs"},
            "general_count_implication": "The chooser includes the exact 2r coded gradient, so it uses no more RHS than constructing both current receiver/directional maps for any admissible r/s. This is specific to these implementations and is not a universal lower bound, runtime inequality, or impossibility theorem for other certificates.",
            "limitations": [
                "One seeded model/direction reused across acquisition geometries, not independent geological replication.",
                "Full-gradient controls return more information than one exact directional derivative; norm-only setup returns neither a gradient nor a derivative.",
                "Linear state equation, fixed model-independent sources/common sampling, and the reference diagonal squared-slowness operator derivative are required.",
                "Coded receiver gradients do not require transpose symmetry; the separate r+s-c source-state reuse route does.",
                "Ordinary floating-point solves and SVDs are not formally verified roundoff enclosures.",
                "RHS counts exclude matrix products, memory traffic and neural inference; factorization and SVD counts are reported separately.",
                "Default run deliberately omits timings. Optional timings require a suitably idle machine and remain invocation measurements, not full-inversion acceleration evidence.",
                "No optimizer defaults, archived results, adaptive trajectories, or time-domain backend are changed by these controls.",
            ], "cases": cases, "provenance": provenance}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        rows = []
        for case in cases:
            for method in case["methods"]:
                rows.append({"sources": case["sources"], "receivers": case["receivers"],
                    "covered_sources": case["covered_sources"], "method": method["method"],
                    "expected_rhs": method["expected_rhs"], "executed_rhs": method["executed_rhs"],
                    "factorizations": method["costs"]["factorizations"], "svd_evaluations": method["costs"]["svd_evaluations"],
                    "full_gradient_constructed": method["full_gradient_constructed"],
                    "objective_absolute_error": method["objective_absolute_error"],
                    "directional_absolute_error": method["directional_absolute_error"],
                    "gradient_relative_l2_error": method["gradient_relative_l2_error"],
                    "median_elapsed_seconds": method["median_elapsed_seconds"],
                    "finite_difference_best_relative_error": case["finite_difference_best_relative_error"]})
        with (output / "cases.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        fig, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
        positions = np.arange(len(cases))
        for index, (method, label, color) in enumerate((("ordinary-gradient", "ordinary full gradient", "#0072B2"),
            ("symmetry-gradient", "symmetry full gradient", "#009E73"), ("coded-gradient", "coded full gradient", "#D55E00"),
            ("best-gradient", "lowest-RHS full gradient", "#CC79A7"))):
            axis.bar(positions+(index-1.5)*0.18,
                [next(m["executed_rhs"] for m in c["methods"] if m["method"] == method) for c in cases],
                width=0.18, color=color, label=label)
        axis.plot(positions, [next(m["executed_rhs"] for m in c["methods"] if m["method"] == "exact-directional-maps") for c in cases],
            color="#333333", linestyle="--", marker="o", label="directional / initial norm maps")
        axis.set_xticks(positions, labels=[f"s={c['sources']}\nr={c['receivers']}" for c in cases])
        axis.set_ylabel("Executed RHS across two frequencies")
        axis.set_xlabel("Source/receiver geometry on one 32×32 model")
        axis.set_title("An exact full-gradient option always matches or beats the map RHS count", loc="left")
        axis.legend(frameon=False, ncol=2, fontsize=8)
        fig.savefig(output / "rhs_comparison.png", dpi=180)
        plt.close(fig)
        status.update(status=summary["study_status"], source_unchanged_during_run=unchanged)
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False)+"\n")
        if not unchanged:
            raise RuntimeError("source changed during reference-control measurement")
        print(json.dumps({"study_status": summary["study_status"], "geometries": len(cases),
            "all_checks_pass": summary["all_checks_pass"], "timing_status": summary["timing_status"]}, indent=2))
    except Exception as error:
        status.update(status="failed" if not str(status["status"]).startswith("stale") else status["status"],
            error_type=type(error).__name__, error=str(error))
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False)+"\n")
        raise


if __name__ == "__main__":
    main()
