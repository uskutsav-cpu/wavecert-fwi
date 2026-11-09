"""Render the complete, audited prior48 endpoint comparison; no PDE calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.expected_report_sha256
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == "92433c9d7b9d2990799f3a81707f6d04f63784a1820d3f109804a83b484ccfa6"
    report = json.loads(raw)
    protocol = json.loads(args.protocol.read_bytes())
    rows, cases = report["rows"], report["summary"]["cases"]
    assert report["verified_full_study_credit"] is True and not report["issues"]
    assert len(rows) == 48 and len(cases) == 8 and all(r["eligible"] for r in rows)
    methods = [p["name"] for p in protocol["methods"]]
    args.output.mkdir(parents=True, exist_ok=True)
    flat = []
    for row in rows:
        initial = row["initial_quality"]["relative_physical_velocity_L2"]
        final = row["final_quality"]["relative_physical_velocity_L2"]
        flat.append(dict(
            run_index=row["run_index"], case_index=row["case_index"],
            slice_id=protocol["identities"][row["case_index"]]["slice_id"],
            repetition=row["repetition"], method=row["method"], weight=row["weight"],
            eligible=row["eligible"], full_return_seconds=row["wall"],
            initial_velocity_error=initial, final_velocity_error=final,
            final_error_over_initial=final / initial,
            raw_Phi_over_initial=row["common_raw_Phi_ratio"],
            **{"preparation_" + k: v for k, v in row["preparation_counts"].items()},
            **{"native_" + k: v for k, v in row["final_native_snapshot"]["physics"].items()},
            adapter_counts=json.dumps(row["adapter_counts"], sort_keys=True),
            metric_tail_counts=json.dumps(row["metric_tail_counts"], sort_keys=True),
            stored_postgraph_counts=json.dumps(row["stored_postgraph_counts"], sort_keys=True),
        ))
    with (args.output / "prior48_all_returns.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    compact = []
    for case in cases:
        index = case["case_index"]
        groups = [[r for r in flat if r["case_index"] == index and r["method"] == m] for m in methods]
        assert all(len(g) == 3 for g in groups)
        baseline, prior = [statistics.median(r["final_velocity_error"] for r in g) for g in groups]
        assert math.isclose(baseline, case["baseline_error"], rel_tol=1e-15)
        assert math.isclose(prior, case["candidate_error"], rel_tol=1e-15)
        compact.append(dict(
            **case, slice_id=protocol["identities"][index]["slice_id"],
            baseline_over_initial=baseline / case["initial_error"],
            prior_over_initial=prior / case["initial_error"],
            prior_over_baseline=prior / baseline,
            required_error_ceiling_over_initial=min(.9, .9 * baseline / case["initial_error"]),
            baseline_raw_Phi_ratio=statistics.median(r["raw_Phi_over_initial"] for r in groups[0]),
            baseline_median_seconds=statistics.median(r["full_return_seconds"] for r in groups[0]),
            prior_median_seconds=statistics.median(r["full_return_seconds"] for r in groups[1]),
        ))
    with (args.output / "prior48_case_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(compact[0]))
        writer.writeheader()
        writer.writerows(compact)
    summary = dict(
        audit_report_sha256=args.expected_report_sha256,
        audit_checks=report["checks"], audit_numeric_comparisons=report["numeric_comparisons"],
        audit_issues=len(report["issues"]), all48_eligible=True,
        actual_return_min_seconds=min(r["wall"] for r in rows),
        actual_return_max_seconds=max(r["wall"] for r in rows),
        successful_cases=sum(c["successful_case"] for c in cases), successful_cases_required=6,
        prior_beats_baseline_cases=sum(c["candidate_error"] < c["baseline_error"] for c in cases),
        prior_worse_than_initial_cases=sum(c["candidate_error"] > c["initial_error"] for c in cases),
        baseline_worse_than_initial_cases=sum(c["baseline_error"] > c["initial_error"] for c in cases),
        harm_gate_passes=sum(c["harm_gate_pass"] for c in cases),
        continuation_gate_pass=report["summary"]["proposed_development_continuation_gate_pass"],
        shared_allowance_is_not_identical_consumed_cost=True,
        old_platform_timings_not_pooled=True, reused_development_cases=True,
        independent_geological_sample_size_not_established=True, scientific_goals_complete=False,
    )
    (args.output / "prior48_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained")
    colors, names = ["#2263a0", "#c45917"], ["Exact data only", "Fractional update prior, λ=100"]
    for arm, method in enumerate(methods):
        offset = -.13 if arm == 0 else .13
        for index in range(8):
            group = sorted((r for r in flat if r["case_index"] == index and r["method"] == method), key=lambda r: r["repetition"])
            x = [index + offset + j for j in (-.025, 0, .025)]
            axes[0].scatter(x, [r["final_error_over_initial"] for r in group], s=18, color=colors[arm], alpha=.32)
            axes[1].scatter(x, [r["raw_Phi_over_initial"] for r in group], s=18, color=colors[arm], alpha=.32)
        for axis, key in [(axes[0], "final_error_over_initial"), (axes[1], "raw_Phi_over_initial")]:
            medians = [statistics.median(r[key] for r in flat if r["case_index"] == i and r["method"] == method) for i in range(8)]
            axis.scatter([i + offset for i in range(8)], medians, s=55, color=colors[arm], marker="o" if arm == 0 else "s", label=names[arm], zorder=4)
    axes[0].scatter(range(8), [r["required_error_ceiling_over_initial"] for r in compact], s=28, marker="D", color="#777777", label="Required error ceiling", zorder=3)
    axes[0].axhline(1, color="#333333", lw=1, ls="--")
    axes[0].set(title="Returned velocity error", ylabel="Error / initial error (lower is better)", yscale="log", ylim=(.74, 4.5))
    axes[0].set_yticks([.8, 1, 1.5, 2, 3, 4])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[1].axhline(.75, color="#777777", lw=1, ls=":", label="Required misfit ceiling: 0.75")
    axes[1].set(title="Returned data misfit", ylabel="Raw Φ / initial Φ (lower is better)", yscale="log", ylim=(1e-5, 1.3))
    for axis in axes:
        axis.set(xlim=(-.5, 7.5), xlabel="Reused development case")
        axis.set_xticks(range(8))
        axis.grid(axis="y", alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False, fontsize=9)
    fig.suptitle("Paired recovery study: 0 of 8 cases meet the fixed target", fontsize=15)
    for suffix in ("png", "svg"):
        fig.savefig(args.output / ("prior48_recovery_comparison." + suffix), dpi=180)
    plt.close(fig)
    caption = (
        "All48 eligible endpoints, eight reused cases × two exact arms × three timing repetitions. "
        "Large markers show case medians; small points retain all repetitions and are not confidence intervals. "
        "The velocity-error ceiling is min(0.90×initial error,0.90×paired baseline error). "
        "Successful cases also require raw Φ/Φ0≤0.75; continuation requires six successful cases and all eight harm gates. "
        "Observed full returns: 7.520710139–7.581680228s under the common8s cap; actual times and distinct work counters are retained in CSV. "
        "No native PDE was rerun by this rendering. Stored postreturn graph work is separate from returned-policy work. "
        "Repetitions/crops are not independent geology; SEAM labels share an ancestor. Classical prior, no certification/novelty/accepted-physics/student-defense claim.\n"
    )
    (args.output / "prior48_recovery_comparison.caption.txt").write_text(caption)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
