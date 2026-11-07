"""Reproduce the receiver-map versus exact-gradient RHS break-even audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from wavecert.experiments.cost_break_even import measure_receiver_map_cost
from wavecert.provenance import environment_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/cost_break_even")
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    rows = [
        measure_receiver_map_cost(sources=sources, receivers=receivers, seed=args.seed)
        for receivers in (2, 6, 12) for sources in (2, 6, 12, 24)
    ]
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key not in {"certificate_costs", "exact_gradient_costs"}}
        flat.update({f"map_{key}": value for key, value in row["certificate_costs"].items()})
        flat.update({f"gradient_{key}": value for key, value in row["exact_gradient_costs"].items()})
        flat_rows.append(flat)
    with (output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    summary = {
        "schema_version": 1,
        "study_status": "engineering operation-count analysis; no timing or neural-performance experiment",
        "scope": "Current exact receiver-map directional certificate on one discrete Helmholtz model/direction per geometry.",
        "model_realizations": 1,
        "seed": args.seed,
        "formula": {
            "symbols": "s sources per frequency, r receivers, f frequencies",
            "first_map_certificate_rhs": "r*f for P A^-1 plus r*f for first directional kappa = 2*r*f",
            "complete_exact_gradient_rhs": "s*f forward plus s*f adjoint = 2*s*f",
            "ratio": "r/s",
            "necessary_condition_for_rhs_saving": "r < s, before counting inference, repairs or verification",
        },
        "limitations": [
            "RHS counts are not FLOPs, byte traffic, or wall-clock time; batching can change runtime per RHS.",
            "No runtime lower bound or universal impossibility claim for all certification algorithms is established.",
            "This lower-work comparison excludes neural state/JVP/VJP costs, residual application, later repair rounds, and finite-step verification.",
            "The analytic stability mode is a different algorithm and is not covered by this receiver-map count model.",
            "A ratio below one is not evidence of practical acceleration or reliable surrogate gradients.",
            "The cases vary geometry on one fixed random model; they are not independent geology trials.",
        ],
        "all_counter_checks_pass": all(row["observed_counts_match_formulas"] for row in rows),
        "cases": rows,
        "provenance": environment_record(root),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
    for receivers, color in ((2, "#0072B2"), (6, "#009E73"), (12, "#D55E00")):
        selected = [row for row in rows if row["receivers"] == receivers]
        ax.plot(
            [row["sources_per_frequency"] for row in selected],
            [row["certificate_to_gradient_rhs_ratio"] for row in selected],
            marker="o", color=color, label=f"{receivers} receivers",
        )
    ax.axhline(1, color="#555555", linestyle="--", linewidth=1.2)
    ax.axhspan(1, 8, color="#D55E00", alpha=0.07)
    ax.text(2.1, 1.12, "No RHS saving from initial maps alone", fontsize=9, color="#8F3C00",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks((2, 6, 12, 24), labels=("2", "6", "12", "24"))
    ax.set_yticks((0.125, 0.25, 0.5, 1, 2, 4, 8), labels=("1/8", "1/4", "1/2", "1", "2", "4", "8"))
    ax.set_ylim(0.06, 8)
    ax.set_xlabel("Sources per frequency")
    ax.set_ylabel("Initial certificate-map RHS / complete gradient RHS")
    ax.set_title("Receiver-map work before neural inference or repair", loc="left", weight="bold")
    ax.legend(title="32×32 grid, two frequencies", frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "rhs_break_even.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"cases": len(rows), "all_counter_checks_pass": summary["all_counter_checks_pass"],
                      "ratios": [row["certificate_to_gradient_rhs_ratio"] for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
