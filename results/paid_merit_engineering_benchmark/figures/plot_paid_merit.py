"""Plot post-terminal descriptive tables; never read a running benchmark."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ["exact_lu_projected", "exact_lu_lbfgsb", "same_M_gmres_projected",
           "fixed_sweeps_uncertified", "fixed_sweeps_certified"]
LABELS = ["Exact LU\nprojected", "Exact LU\nL-BFGS-B", "Same-M\nGMRES", "Fixed sweeps\nuncertified",
          "Fixed sweeps\ncertified"]


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot(analysis, output):
    summary = json.loads((analysis / "summary.json").read_text())
    if summary["planned_runs"] != 360:
        raise ValueError("Complete planned denominator required")
    targets = rows(analysis / "returned_targets.csv")
    internal = rows(analysis / "internal_checkpoints.csv")
    if len(targets) != 240 or len(internal) != 480:
        raise ValueError("Missing rows including failed/unavailable cases")
    output.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True, constrained_layout=True)
    categories = ["Eligible returned target", "Late target return", "Unreached / other", "Study ineligible"]
    colors = ["#257b99", "#d98f42", "#c9ced4", "#676f79"]
    for ax, mode, target in zip(axes, ("target_stop_0.5", "target_stop_0.25"), (0.5, 0.25)):
        grouped = []
        for method in METHODS:
            selected = [r for r in targets if r["method"] == method and r["measurement_mode"] == mode]
            if len(selected) != 24:
                raise ValueError("Full method denominator required")
            counts = Counter()
            for row in selected:
                category = (categories[3] if row["study_eligible"] != "True" else
                            categories[0] if row["target_eligible"] == "True" else
                            categories[1] if row["reason"] == "target_returned_after_budget" else categories[2])
                counts[category] += 1
            grouped.append(counts)
        bottom = [0] * 5
        for category, color in zip(categories, colors):
            values = [count[category] for count in grouped]
            ax.bar(range(5), values, bottom=bottom, color=color, label=category, width=.68)
            bottom = [a+b for a, b in zip(bottom, values)]
        for index, count in enumerate(grouped):
            ax.text(index, 24.5, f"{count[categories[0]]}/24", ha="center", fontsize=10)
        ax.set_xticks(range(5), LABELS, fontsize=9)
        ax.set_ylim(0, 27)
        ax.set_yticks([0, 6, 12, 18, 24])
        ax.set_title(f"Returned objective target: Φ / Φ₀ ≤ {target:g}")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Planned runs per method (8 slices × 3 repetitions)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle("Paid-merit engineering benchmark: actual return within 8 seconds", fontsize=14)
    fig.savefig(output / "returned_target_counts.png", dpi=200)
    fig.savefig(output / "returned_target_counts.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), constrained_layout=True)
    metrics = [("objective_ratio", "Objective Φ / Φ₀"),
               ("relative_velocity_L2", "Physical-velocity relative L2 error")]
    palette = ["#245a81", "#d17a2d", "#4b8465", "#9e5a81", "#66849d", "#aa9b3b", "#725b9c", "#68777e"]
    missing = []
    for ax, (metric, ylabel) in zip(axes, metrics):
        for case, color in enumerate(palette):
            values = []
            for method in METHODS:
                selected = [r for r in internal if int(r["case_index"]) == case and r["method"] == method
                            and float(r["budget_seconds"]) == 8]
                valid = [r for r in selected if r["internal_checkpoint_eligible"] == "True"]
                if len(selected) != 3:
                    raise ValueError("Three planned repetitions required")
                finite = [float(r[metric]) for r in valid if r[metric] != "" and math.isfinite(float(r[metric]))]
                values.append(statistics.median(finite) if len(valid) == len(finite) == 3 else float("nan"))
                if len(valid) != 3 or len(finite) != 3:
                    missing.append(dict(case_index=case, method=method, metric=metric,
                                        paid=len(valid), finite=len(finite), planned=3))
            identity = selected[0]["slice_id"]
            ax.plot(range(5), values, "o-", color=color, alpha=.85, linewidth=1.0,
                    markersize=4, label=identity)
        ax.set_xticks(range(5), LABELS, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.18)
        ax.set_title("Case median of 3 eligible repetitions; missing values stay missing", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("8-second INTERNAL paid states — not completed API-return evidence", fontsize=13)
    fig.savefig(output / "internal_8s_quality.png", dpi=200)
    fig.savefig(output / "internal_8s_quality.svg")
    plt.close(fig)
    (output / "missing_internal_repetitions.json").write_text(json.dumps(missing, indent=2)+"\n")
    caption = (
        "AI-authored software-development figures on eight reused validation slices; "
        "five methods, three repetitions, three measurement modes, 360 planned runs. "
        "Target bars retain every planned run and require terminal study eligibility, "
        "positive initial objective, complete objective-target stop and actual return within 8 s. "
        "Unreached targets have no observed target time; no censor time is imputed as success. "
        "Internal-quality panels use only completed paid checkpoints at or before 8 s and require "
        "all three paid repetitions and finite values for a metric's case median; legitimate "
        "null objective ratios (zero initial objective), gaps and missing counts remain recorded. "
        "These internal states are not completed-return evidence. Objective and model error are "
        "different quantities. Source loading, synthetic-observation preparation, setup, rejected "
        "trials, certification, fallback and bounded progress writes are in the run clock; fresh "
        "reference audits and final artifact I/O are separate. Slices/repetitions are correlated; "
        "SEAM IDs share ancestry. No IID "
        "interval, independent-site count, untouched confirmation, winner, student ownership, "
        "or scientific-goal completion is inferred. All methods use paid reference-objective decrease.\n"
    )
    (output / "caption.txt").write_text(caption)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis, args.output)


if __name__ == "__main__":
    main()
