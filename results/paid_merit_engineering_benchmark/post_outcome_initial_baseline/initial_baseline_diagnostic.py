"""Additive post-outcome diagnostic; original benchmark and figures stay intact."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ["exact_lu_projected", "exact_lu_lbfgsb", "same_M_gmres_projected",
           "fixed_sweeps_uncertified", "fixed_sweeps_certified"]
LABELS = ["Exact LU\nprojected", "Exact LU\nL-BFGS-B", "Same-M\nGMRES",
          "Fixed sweeps\nuncertified", "Fixed sweeps\ncertified"]


def diagnose(audit_path, analysis, output):
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed" or audit["failures"]:
        raise ValueError("Passed independent model audit required")
    initial = {}
    for record in audit["records"]:
        values = record["saved_models"]["prepared_initial"]
        previous = initial.setdefault(record["case_index"], values)
        for key in ("model_sha256", "relative_physical_velocity_L2", "relative_squared_slowness_L2"):
            if previous[key] != values[key]:
                raise ValueError("Initial model/quality varies within case")
    with (analysis / "internal_checkpoints.csv").open(newline="") as stream:
        internal = list(csv.DictReader(stream))
    if len(internal) != 480 or set(initial) != set(range(8)):
        raise ValueError("Complete planned denominators required")
    result = []
    for case in range(8):
        for method in METHODS:
            group = [r for r in internal if int(r["case_index"]) == case and r["method"] == method
                     and float(r["budget_seconds"]) == 8]
            if len(group) != 3:
                raise ValueError("Missing planned repetitions")
            eligible = [r for r in group if r["internal_checkpoint_eligible"] == "True"]
            row = dict(case_index=case, slice_id=group[0]["slice_id"], method=method,
                       planned=3, eligible=len(eligible), internal_only=True)
            for metric, initial_key in (("velocity", "relative_physical_velocity_L2"),
                                        ("squared_slowness", "relative_squared_slowness_L2")):
                column = "relative_velocity_L2" if metric == "velocity" else "relative_squared_slowness_L2"
                start = initial[case][initial_key]
                end = statistics.median(float(r[column]) for r in eligible) if len(eligible) == 3 else None
                if not math.isfinite(start) or start < 0 or (end is not None and (not math.isfinite(end) or end < 0)):
                    raise ValueError("Invalid error")
                row[metric + "_initial_error"] = start
                row[metric + "_8s_median_error"] = end
                row[metric + "_error_over_initial"] = None if end is None or start == 0 else end / start
                row[metric + "_improved"] = None if end is None else end < start
            result.append(row)
    output.mkdir(exist_ok=False)
    with (output / "initial_baseline_comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    colors = ["#245a81", "#d17a2d", "#4b8465", "#9e5a81", "#66849d", "#aa9b3b", "#725b9c", "#68777e"]
    for ax, metric, label in zip(axes, ("velocity", "squared_slowness"), ("Physical velocity", "Squared slowness")):
        for case, color in enumerate(colors):
            group = [r for r in result if r["case_index"] == case]
            values = [r[metric + "_error_over_initial"] for r in group]
            ax.plot(range(5), [float("nan") if v is None else v for v in values], "o-",
                    color=color, linewidth=1, markersize=4, label=group[0]["slice_id"])
        ax.axhline(1, color="#222222", linestyle="--", linewidth=1.4, label="Starting-model error")
        ax.set_yscale("log", base=2)
        ax.set_xticks(range(5), LABELS, fontsize=9)
        ax.set_ylabel(f"{label} error / starting-model error")
        ax.set_title("Below 1: improved; above 1: worse", fontsize=11)
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle("Post-outcome diagnostic: 8-second INTERNAL states versus starting model", fontsize=13)
    fig.savefig(output / "initial_baseline_diagnostic.png", dpi=200)
    fig.savefig(output / "initial_baseline_diagnostic.svg")
    plt.close(fig)
    summary = dict(
        role="AI-authored additive POST-OUTCOME diagnostic, requested after inspecting 8s quality",
        selection="All eight cases and all five methods retained; no case or parameter tuning",
        original_study_and_figures_unchanged=True,
        audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method_summary=[dict(method=method, planned_cases=8,
            velocity_improved=sum(r["velocity_improved"] is True for r in result if r["method"] == method),
            velocity_worse=sum(r["velocity_improved"] is False and r["velocity_8s_median_error"] > r["velocity_initial_error"]
                               for r in result if r["method"] == method),
            unavailable=sum(r["velocity_improved"] is None for r in result if r["method"] == method))
            for method in METHODS],
        scope="Paid internal checkpoints, not completed-return evidence, independent geology, confirmation, causal explanation, or student interpretation",
        scientific_success_claim=False,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (output / "caption.txt").write_text(
        "Additive diagnostic created after outcome inspection, with all eight reused validation slices "
        "and all five methods retained. Each point is the median of three eligible paid internal "
        "8-second error measurements divided by that case's independently checked initial-model "
        "error. A ratio above one is worse than initialization. These are internal states, not "
        "fully returned results. Zero initial error would have no defined ratio; unavailable states "
        "remain missing. No IID uncertainty, independent-site, causal, performance-win or confirmation "
        "claim follows. The underlying frozen benchmark, original two figures and all outcomes remain "
        "unchanged. Data-misfit decrease and geological reconstruction are separate outcomes.\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.audit, args.analysis, args.output), indent=2))


if __name__ == "__main__":
    main()
