"""Plot all frozen initial slope diagnostics; no new model/PDE evaluation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(run, output):
    summary = json.loads((run / "summary.json").read_text())
    if summary["result"] != "clear_software_diagnostic" or summary["issues"]:
        raise ValueError("Clear completed diagnostic required")
    cases = [json.loads((run / f"case_{i:02d}_slopes.json").read_text()) for i in range(8)]
    directions = ["actual_raw_proposal", "exact_normalized_negative_data_gradient"]
    columns = [("dataDD", "Data-loss directional derivative"),
               ("mSlope", "Half relative squared m-error derivative"),
               ("vSlope", "Half relative squared velocity-error derivative")]
    labels = ["Certified proposal", "Exact negative data gradient"]
    colors = ["#257b99", "#cf8d3d"]
    rows = []
    for case in cases:
        for name in directions:
            d = case["directions"][name]
            if not d["available"]:
                raise ValueError("Unavailable direction requires explicit missing-data handling")
            metrics = d["half_relative_squared_metric_derivatives"]
            rows.append(dict(case_index=case["case_index"], slice_id=case["identity"]["slice_id"],
                direction=name, dataDD=d["data_directional_derivative"]["value"],
                mSlope=metrics["half_relative_squared_m_error"]["analytic"],
                vSlope=metrics["half_relative_squared_velocity_error"]["analytic"],
                toward_truth_dataDD=case["data_gradient_dot_toward_truth_correction"]["value"],
                initial_box_feasible=case["initial_model_box_feasibility"]["literal_box_feasible"],
                truth_box_feasible=case["truth_model_box_feasibility"]["literal_box_feasible"],
                initial_exact_active_bounds=case["initial_model_box_feasibility"]["exact_lower_bound_cells"]
                    +case["initial_model_box_feasibility"]["exact_upper_bound_cells"],
                internal_initial_unprojected_only=True))
    output.mkdir(exist_ok=False)
    with (output / "all_initial_slopes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.3), sharey=True, constrained_layout=True)
    for ax, (column, label) in zip(axes, columns):
        for j, name in enumerate(directions):
            values = [next(r[column] for r in rows if r["case_index"] == i and r["direction"] == name) for i in range(8)]
            ax.barh([i+(j-.5)*.32 for i in range(8)], values, height=.29, color=colors[j], label=labels[j])
        ax.axvline(0, color="#252525", linewidth=1)
        ax.grid(axis="x", alpha=.17)
        ax.set_axisbelow(True)
        ax.set_xlabel(label)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        ax.set_title("Negative: decreases locally; positive: increases locally", fontsize=9)
    axes[0].set_yticks(range(8), [c["identity"]["slice_id"] for c in cases], fontsize=9)
    axes[0].invert_yaxis()
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="outside lower center", ncol=2, fontsize=10)
    fig.suptitle("Local unprojected slopes at all eight reused initial models", fontsize=14)
    fig.savefig(output / "initial_direction_slopes.png", dpi=200)
    fig.savefig(output / "initial_direction_slopes.svg")
    plt.close(fig)
    caption = (
        "AI-authored post-outcome software diagnostic. All eight original initial cases and both "
        "directions are shown; every direction is normalized in native squared-slowness coordinates. "
        "The three panels have different units and axis scales; compare signs within each metric, "
        "not magnitudes between metrics. Model-error derivatives concern half-relative-SQUARED errors, "
        "not the previously plotted relative L2 norms. The velocity derivative includes dv/dm=-.5m^-1.5. "
        "All true and initial models are feasible and no initial parameter is exactly at a bound. "
        "Five velocity-error slopes and three squared-slowness-error slopes are positive for each "
        "direction, while every data derivative is negative. Fixed two-scale centered differences "
        "check the metric calculus; numerical sign bands are not formal enclosures. These are raw "
        "unprojected local directions, not finite clipped steps or predictions of later trajectories. "
        "Initial sign cases differ from the late reconstruction-drift cases. No causal explanation, "
        "parameterization remedy, method/rank selection, new geological independence or student "
        "interpretation is established. Truth is used only for this offline diagnostic, never policy.\n"
    )
    (output / "caption.txt").write_text(caption)
    (output / "provenance.json").write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnostic_complete_receipt_sha256=hashlib.sha256((run / "complete_receipt.json").read_bytes()).hexdigest(),
        rows=len(rows), case_denominator=8, new_PDE_or_trajectory=False,
        scope="All fixed raw initial directions, post-outcome, numerical local statistics only",
    ), indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.run, args.output)


if __name__ == "__main__":
    main()
