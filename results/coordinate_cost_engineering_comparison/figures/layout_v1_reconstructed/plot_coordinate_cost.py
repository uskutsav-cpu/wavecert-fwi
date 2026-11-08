"""Plot existing verified returned-state diagnostics; no fitting or PDE solves."""

from __future__ import annotations

import csv
import datetime
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RUN = ROOT / "work/coordinate_cost_run_02"
AUDIT = ROOT / "work/coordinate_cost_post_audit_02/report.json"
METHODS = ("native_m_lbfgsb", "physical_v_lbfgsb")
COLORS = {METHODS[0]: "#C76B08", METHODS[1]: "#008581"}
SOURCE = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    report = read_json(AUDIT)
    status_path = RUN / "status.json"
    status = read_json(status_path)
    assert report["result"] == "CLEAR_COORDINATE_ENGINEERING_POSTAUDIT"
    assert report["checks"] == 4523 and report["numeric_comparisons"] == 672
    assert not report["issues"] and report["new_PDE_solves"] == 0
    assert report["production_source_before"] == report["production_source_after"] == SOURCE
    assert status["status"] == "complete_engineering_comparison"
    assert len(report["rows"]) == len(status["runs"]) == 48
    assert all(row["verified_credit_eligible"] for row in report["rows"])
    by_index = {row["run_index"]: row for row in status["runs"]}
    input_paths = [AUDIT, status_path, RUN / "complete_receipt.json"]
    source_inputs = {}
    rows = []
    for checked in report["rows"]:
        recorded = by_index[checked["run_index"]]
        assert recorded["status"] == "complete" and recorded["final_study_eligible"]
        path = RUN / recorded["core_artifact"]
        assert sha(path) == recorded["core_sha256"]
        source_inputs[str(path.relative_to(ROOT))] = sha(path)
        trajectory = read_json(path)["trajectory"]
        initial_phi = trajectory["initial_objective"]
        final_phi = trajectory["final_objective"]
        assert math.isfinite(initial_phi) and initial_phi > 0 and math.isfinite(final_phi)
        ratio = final_phi / initial_phi
        assert 0 < ratio < 1
        quality = checked["scalar_errors"]
        rows.append(dict(
            run_index=checked["run_index"], case_index=checked["case_index"],
            slice_id=checked["slice_id"], method=checked["method"], repetition=checked["repetition"],
            verified_credit_eligible=True, actual_api_return_seconds=checked["actual_api_return_seconds"],
            initial_objective=initial_phi, returned_objective=final_phi, returned_objective_ratio=ratio,
            initial_physical_velocity_relative_L2=quality["common_initial"]["relative_physical_velocity_L2"],
            returned_physical_velocity_relative_L2=quality["returned_final"]["relative_physical_velocity_L2"],
            initial_squared_slowness_relative_L2=quality["common_initial"]["relative_squared_slowness_L2"],
            returned_squared_slowness_relative_L2=quality["returned_final"]["relative_squared_slowness_L2"],
            common_actual_native_m_sha256=checked["common_binding"]["actual_common_m_sha256"],
            returned_actual_native_m_sha256=checked["final_actual_native_m_sha256"],
            producer_core_artifact=recorded["core_artifact"], producer_core_sha256=recorded["core_sha256"],
        ))
        assert rows[-1]["actual_api_return_seconds"] <= 8
    rows.sort(key=lambda row: row["run_index"])
    write_csv(OUTPUT / "all_48_returned_states.csv", rows)
    groups = {}
    initials = {}
    summaries = []
    for case in range(8):
        case_rows = [row for row in rows if row["case_index"] == case]
        assert len(case_rows) == 6
        assert len({row["common_actual_native_m_sha256"] for row in case_rows}) == 1
        assert len({row["initial_physical_velocity_relative_L2"] for row in case_rows}) == 1
        initial = case_rows[0]["initial_physical_velocity_relative_L2"]
        initials[case] = initial
        for method in ("common_initial", *METHODS):
            selected = [row for row in case_rows if row["method"] == method]
            if method == "common_initial":
                error = [initial]
                phi = [1.]
            else:
                assert len(selected) == 3 and {row["repetition"] for row in selected} == {0, 1, 2}
                error = [row["returned_physical_velocity_relative_L2"] for row in selected]
                phi = [row["returned_objective_ratio"] for row in selected]
                groups[case, method] = (error, phi)
            summaries.append(dict(
                case_index=case, slice_id=case_rows[0]["slice_id"], method=method,
                optimizer_repetitions=len(selected),
                physical_velocity_relative_L2_min=min(error),
                physical_velocity_relative_L2_median=statistics.median(error),
                physical_velocity_relative_L2_max=max(error),
                returned_objective_ratio_min=min(phi),
                returned_objective_ratio_median=statistics.median(phi),
                returned_objective_ratio_max=max(phi),
                range_scope="one shared initial model" if method == "common_initial" else "all three repetitions; min-max, not a confidence interval",
            ))
    write_csv(OUTPUT / "all_8_case_summaries.csv", summaries)
    worse = {}
    for method in METHODS:
        worse[method] = [case for case in range(8) if statistics.median(groups[case, method][0]) > initials[case]]
    assert worse[METHODS[0]] == worse[METHODS[1]] == [0, 1, 2, 4, 5]
    assert all(statistics.median(groups[case, METHODS[1]][0]) < statistics.median(groups[case, METHODS[0]][0]) for case in worse[METHODS[0]])
    returned_ratio_counts = {
        method: {str(target): sum(row["returned_objective_ratio"] <= target for row in rows if row["method"] == method)
                 for target in (.5, .25)} for method in METHODS
    }
    assert returned_ratio_counts == {METHODS[0]: {"0.5": 24, "0.25": 24}, METHODS[1]: {"0.5": 21, "0.25": 18}}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": "#B2B8BF",
                         "axes.labelcolor": "#252B34", "xtick.color": "#424A55", "ytick.color": "#252B34",
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 9.0), sharey=True)
    fig.subplots_adjust(left=.215, right=.98, bottom=.19, top=.775, wspace=.18)
    fig.text(.215, .966, "Returned-state diagnostics under a fixed 8 s API cap", fontsize=17, weight="bold", color="#19212D")
    fig.text(.215, .927, "All 8 reused development slices · 48/48 eligible returns · both coordinate systems and the common initial model", fontsize=10.2, color="#46505D")
    handles = [Line2D([], [], color="#1B2027", marker="D", linestyle="none", markersize=6, label="Common initial model"),
               Line2D([], [], color=COLORS[METHODS[0]], marker="o", linestyle="-", markersize=6, label=r"L-BFGS-B in $m=v^{-2}$"),
               Line2D([], [], color=COLORS[METHODS[1]], marker="o", linestyle="-", markersize=6, label=r"L-BFGS-B in $v$")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.207, .891), ncol=3,
               frameon=False, fontsize=10.6, handlelength=1.6, columnspacing=1.7)
    fig.text(.215, .827, "Large circles: median of all 3 repetitions   |   Bars: min–max   |   Small marks: individual repetitions", fontsize=9.3, color="#58616D")
    offsets = {METHODS[0]: -.16, METHODS[1]: .16}
    for panel, ax in enumerate(axes):
        for case in range(8):
            if case % 2 == 0:
                ax.axhspan(case-.47, case+.47, color="#F4F6F8", zorder=0)
            baseline = 100*initials[case] if panel == 0 else 1.
            ax.scatter([baseline], [case], marker="D", s=28, c="#1B2027", edgecolors="white", linewidths=.45, zorder=5)
            for method in METHODS:
                values = groups[case, method][panel]
                values = [100*x for x in values] if panel == 0 else values
                median = statistics.median(values)
                y = case + offsets[method]
                ax.errorbar(median, y, xerr=[[median-min(values)], [max(values)-median]],
                            fmt="o", color=COLORS[method], markersize=6, capsize=3.1,
                            elinewidth=1.6, capthick=1.2, markeredgecolor="white", markeredgewidth=.5, zorder=4)
                ax.scatter(values, [y-.035, y, y+.035], marker="|", s=29, linewidths=1.05,
                           color=COLORS[method], alpha=.85, zorder=3)
        ax.set_xscale("log")
        ax.grid(axis="x", which="major", color="#DCE1E6", linewidth=.65)
        ax.set_axisbelow(True)
        ax.set_yticks(range(8))
        ax.tick_params(axis="y", length=0, pad=10)
        ax.tick_params(axis="x", labelsize=10.5)
        ax.set_ylim(7.58, -.58)
    labels = [f"{case} · " + next(row["slice_id"] for row in rows if row["case_index"] == case) for case in range(8)]
    axes[0].set_yticklabels(labels, fontsize=10.6)
    axes[0].set_title("(a) Physical-velocity reconstruction error", loc="left", fontsize=11.5, pad=14)
    axes[1].set_title("(b) Returned data misfit relative to initial", loc="left", fontsize=11.5, pad=14)
    all_errors = [100*row["returned_physical_velocity_relative_L2"] for row in rows] + [100*x for x in initials.values()]
    axes[0].set_xlim(min(all_errors)*.72, max(all_errors)*1.26)
    axes[0].xaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.), numticks=12))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[0].set_xlabel(r"$100\,\|v-v^*\|_2/\|v^*\|_2$  (%, log scale)", labelpad=12)
    min_ratio = min(row["returned_objective_ratio"] for row in rows)
    axes[1].set_xlim(min_ratio*.5, 1.45)
    axes[1].xaxis.set_major_locator(LogLocator(base=10, numticks=8))
    axes[1].xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    axes[1].set_xlabel(r"$\Phi_{\rm return}/\Phi_0$  (log scale)", labelpad=12)
    fig.text(.215, .100, "Both coordinate systems exceed the initial reconstruction error in cases 0, 1, 2, 4 and 5.", fontsize=10.2, weight="bold", color="#353D48")
    fig.text(.215, .064, "Engineering development only · shared SEAM ancestry · min–max ranges are not confidence intervals", fontsize=9.4, color="#58616D")
    fig.savefig(OUTPUT / "coordinate_cost_returned_diagnostics.png", dpi=220, facecolor="white")
    fig.savefig(OUTPUT / "coordinate_cost_returned_diagnostics.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)

    old_path = ROOT / "work/coordinate_cost_run_01/status.json"
    old = read_json(old_path)
    old_counts = {name: sum(row["status"] == name for row in old["runs"]) for name in sorted({row["status"] for row in old["runs"]})}
    assert old_counts == {"failed": 14, "interrupted": 1, "unattempted": 33}
    caption = (
        "Returned-state diagnostics for all eight reused development slices, two paid-reference L-BFGS-B coordinate systems and three repetitions: all 48 source-bound returned states pass the joint post-audit eligibility gate under the same fixed 8 s API cap (7.5 s engine budget plus a fixed 0.5 s output reserve, not a guarantee of return). Methods may return before the cap; actual consumed times are retained in the 48-row CSV. Preparation, input/source checks, synthetic-observation construction, optimizer work, engine finishing work and final native-model conversion/hash/provenance/cost snapshot are inside the API clock. Artifact compression and fresh audits are separately measured after return.\n\n"
        "Panel (a) shows relative physical-velocity L2 error as percent of the truth norm. Panel (b) shows returned Phi/Phi0 for the same runs. Small colored marks show every repetition; large circles are the median of all three; bars span their minimum and maximum (not confidence intervals). Some ranges are narrower than the marker. Black diamonds identify the single common canonical initial model; its panel-(b) ratio is one. No internal checkpoint receives reconstruction-quality credit.\n\n"
        "Both coordinate systems have larger physical-velocity error than the common initial model in the same five cases (0, 1, 2, 4, 5), and smaller error in cases 3, 6, 7. The v-coordinate median error is smaller than the m-coordinate median in those five cases, while remaining larger than the initial error. Native m returns meet the half and quarter misfit ratios in 24/24 runs each; v-coordinate returns do so in 21/24 and 18/24. These are returned-final diagnostics, not stopping targets or first-hit times; no target thresholds are drawn.\n\n"
        "This is the current finite 49-by-49 Dirichlet Helmholtz inverse problem with same-operator noiseless observations. The eight slices are already inspected development inputs from four provider IDs; the two SEAM IDs share one original SEAM ancestor. Slices and computational repetitions do not establish independent sites. Coordinate-dependent gtol is not an equal physical-stationarity criterion. No neural/certificate comparison, recovery guarantee, continuum/field validation, new-method claim or research-goal completion follows. The earlier invalid v1 attempt (14 failed, one interrupted, 33 unattempted) is preserved separately and is not pooled with v2. This figure and caption are AI-authored engineering documentation, not a student paper or student-authored scientific conclusion.\n"
    )
    (OUTPUT / "coordinate_cost_returned_diagnostics.caption.txt").write_text(caption)
    for path in input_paths + [old_path]:
        source_inputs[str(path.relative_to(ROOT))] = sha(path)
    assert all(sha(ROOT/path) == checksum for path, checksum in source_inputs.items())
    provenance = dict(
        role="AI-authored publication of existing verified engineering data; no fitting, propagation, threshold selection or omission",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), production_source_sha256=SOURCE,
        audit_result=report["result"], audit_checks=4523, audit_numeric_comparisons=672, audit_issues=0,
        planned_denominator=48, shown_verified_returns=48, case_denominator=8, repetitions_per_method_case=3,
        raw_rows_csv="all_48_returned_states.csv", case_summaries_csv="all_8_case_summaries.csv",
        common_initial_is_single_model_not_independent_replicates=True,
        summary_rule="all3 median with min-max; no confidence interval or successful-only selection",
        larger_error_than_initial_cases=worse, returned_ratio_counts=returned_ratio_counts,
        invalid_prior_attempt_counts=old_counts, no_new_PDE_solves=True, no_new_timings=True,
        plotting_versions=dict(python=__import__("sys").version, numpy=np.__version__, matplotlib=matplotlib.__version__),
        input_file_sha256=source_inputs, input_files_unchanged_after_plotting=True,
        visual_QA="pending separate image inspection",
        artifact_sha256={path.name: sha(path) for path in OUTPUT.iterdir() if path.is_file() and path.name != "provenance.json"},
    )
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(dict(output=str(OUTPUT), shown_rows=48, shown_cases=8, larger_error_cases=worse,
                          returned_ratio_counts=returned_ratio_counts, figure_sha256=sha(OUTPUT / "coordinate_cost_returned_diagnostics.png"))))


if __name__ == "__main__":
    main()
