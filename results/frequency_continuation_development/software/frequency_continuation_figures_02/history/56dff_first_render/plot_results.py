#!/usr/bin/env python3
"""Post-terminal descriptive plots. Default CLI reads no study outcomes."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

METHODS = ("joint8", "low4_then8", "restart8", "original4")
LABELS = ("Joint 8", "Low 4 → 8", "Restart 8", "Original 4")
PROTOCOL_SHA = "5c2838eaa1142ae7497717d7a4dfc46c76df4fb5ff8b3e016d5479eb4b51af1e"
DRIVER_SHA = "a5f4a7bd38350ea1958160d0a5d24fb29b03e986b2601117c1bc7fd399394558"
SOURCE_SHA = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
SCOPE = (
    "Eight reused development slices; nominal SEAM identities share ancestry, "
    "and repetitions are timing repetitions, not independent geology. "
    "Original 4 uses fewer observations. Descriptive engineering comparison; "
    "no physical acceptance, novelty, causal recovery or competition claim."
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def number(value):
    return (float(value) if isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) else None)


def median(values):
    values = [x for x in values if x is not None]
    return statistics.median(values) if values else None


def gz_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def bound_artifact(root, row, kind, manifest):
    name = row.get(kind + "_artifact")
    if name is None:
        return None
    path = root / name
    if path.parent != root or not path.is_file():
        raise ValueError("invalid or missing artifact: " + str(name))
    digest = sha(path)
    if digest != row.get(kind + "_sha256") or digest != manifest.get(name):
        raise ValueError("artifact hash mismatch: " + name)
    return gz_json(path)


def stage_details(trace):
    """Retain actual statuses and native limits; never infer nonbinding caps."""
    details = []
    for stage in trace.get("stages", []):
        result = stage.get("engine_result", {})
        native = result.get("native_solver_result") or {}
        # Native SciPy metadata is named solver_result by the frozen engine.
        native = result.get("solver_result") or native
        details.append(dict(index=stage.get("index"), status=stage.get("status"),
            engine_result_available=bool(result),
            maximum_steps=stage.get("maximum_steps"),
            callback_invocations=stage.get("native_callback_invocations"),
            callback_point_records=stage.get("native_callback_point_records"),
            native_solver_result=native, engine_message=result.get("message"),
            scipy_maxfun_default=result.get("scipy_maxfun_default"),
            scipy_maxfun_passed_explicitly=result.get("scipy_maxfun_passed_explicitly")))
    return details


def cap_flag(details):
    """Diagnostic native message/count classification, with raw messages in CSV."""
    flags = []
    for stage in details:
        if not stage["engine_result_available"]:
            continue
        native = stage["native_solver_result"]
        message = str(native.get("message", "")).upper()
        nit, cap = number(native.get("nit")), number(stage["maximum_steps"])
        count_hit = nit is not None and cap is not None and nit >= cap
        flags.append(bool(count_hit or "ITERATIONS REACHED LIMIT" in message
                          or "EVALUATIONS EXCEEDS LIMIT" in message
                          or "F AND G EVALUATIONS" in message))
    return any(flags) if flags else None


def collect(root, receipt, protocol):
    manifest = receipt["artifact_sha256"]
    schedule = json.loads((root / "schedule.json").read_text())
    planned = {(c, r, m) for c in range(8) for r in range(3) for m in METHODS}
    keys = [(r["case_index"], r["repetition"], r["method"]) for r in schedule]
    if len(schedule) != 96 or len(set(keys)) != 96 or set(keys) != planned:
        raise ValueError("saved schedule is not the complete 8 × 4 × 3 plan")
    actual = receipt.get("runs", [])
    if len({r["run_index"] for r in actual}) != len(actual):
        raise ValueError("duplicate observed run_index")
    by_index = {r["run_index"]: r for r in actual}
    rows, issues = [], []
    for saved in schedule:
        observed = by_index.get(saved["run_index"])
        row = dict(run_index=saved["run_index"], case_index=saved["case_index"],
            repetition=saved["repetition"], method=saved["method"],
            slice_id=saved["identity"]["slice_id"], parent_id=saved["identity"]["parent_id"],
            status="missing" if observed is None else observed.get("status"),
            observed=observed is not None, final_study_eligible=False, integrity_valid=False,
            runtime_eligible=False, actual_api_return_seconds=None, wall_until_failure_seconds=None,
            velocity_initial=None, velocity_final_raw=None, velocity_final_eligible=None,
            common8_initial_objective=None, common8_final_objective_raw=None,
            common8_ratio_raw=None, common8_ratio_eligible=None,
            common8_baseline_ratio=None, fit_unavailable_reason=None,
            cap_or_function_limit_diagnostic=None,
            stage_details="[]", exclusion_reason="missing scheduled row", error=None)
        if observed is None:
            issues.append(dict(run_index=row["run_index"], error="missing scheduled row"))
            rows.append(row)
            continue
        try:
            for key in ("case_index", "repetition", "method", "identity"):
                if observed.get(key) != saved[key]:
                    raise ValueError("schedule binding mismatch: " + key)
            timing = observed.get("timing", {})
            row.update(integrity_valid=observed.get("integrity_valid") is True,
                runtime_eligible=timing.get("runtime_eligible") is True,
                actual_api_return_seconds=number(timing.get("actual_api_return_seconds")),
                wall_until_failure_seconds=number(observed.get("wall_until_failure_or_interrupt_seconds")),
                error=observed.get("error"))
            bound_trace = bound_artifact(root, observed, "trajectory", manifest)
            bound_audit = bound_artifact(root, observed, "audit", manifest)
            trace, audit = bound_trace or {}, bound_audit or {}
            details = stage_details(trace)
            row["stage_details"] = json.dumps(details, separators=(",", ":"), allow_nan=False)
            row["cap_or_function_limit_diagnostic"] = cap_flag(details)
            quality = audit.get("own_quality", {})
            row["velocity_initial"] = number(quality.get("initial", {}).get("relative_physical_velocity_L2"))
            row["velocity_final_raw"] = number(quality.get("final", {}).get("relative_physical_velocity_L2"))
            initial = trace.get("common_initial_full8") or {}
            final = trace.get("common_final_full8") or {}
            row["common8_initial_objective"] = number(initial.get("objective"))
            row["common8_final_objective_raw"] = number(final.get("objective"))
            row["common8_ratio_raw"] = number(trace.get("common_full8_ratio"))
            if row["common8_initial_objective"] is not None and row["common8_initial_objective"] > 0:
                row["common8_baseline_ratio"] = 1.0
            wall = row["actual_api_return_seconds"]
            eligible = (observed.get("final_study_eligible") is True and row["integrity_valid"]
                and row["runtime_eligible"] and observed.get("status") == "complete"
                and observed.get("primary_quality") is not None and wall is not None
                and 0 <= wall <= protocol["controller_settings"]["api_return_cap_seconds"])
            if observed.get("final_study_eligible") is True and not eligible:
                raise ValueError("producer eligibility inconsistent with returned/integrity/time fields")
            if eligible and (bound_trace is None or bound_audit is None or row["velocity_final_raw"] is None
                    or observed["primary_quality"] != quality.get("final")):
                raise ValueError("eligible primary quality is missing or differs from bound audit")
            if eligible:
                a, b = row["common8_initial_objective"], row["common8_final_objective_raw"]
                if a is None or b is None or a < 0 or b < 0:
                    raise ValueError("eligible paid common8 initial/final objectives missing or invalid")
                if a > 0 and row["common8_ratio_raw"] is None:
                    raise ValueError("positive-initial eligible common8 ratio missing or nonfinite")
                if a == 0 and row["common8_ratio_raw"] is not None:
                    raise ValueError("zero-initial common8 objective incorrectly assigned a ratio")
                if a == 0:
                    row["fit_unavailable_reason"] = "zero_initial_objective; no ratio or fit credit"
            row["final_study_eligible"] = eligible
            row["velocity_final_eligible"] = row["velocity_final_raw"] if eligible else None
            row["common8_ratio_eligible"] = row["common8_ratio_raw"] if eligible else None
            row["exclusion_reason"] = (None if eligible else "failed_missing_incomplete_late_or_integrity_invalid")
        except Exception as exc:
            row.update(final_study_eligible=False, velocity_final_eligible=None,
                common8_ratio_eligible=None, exclusion_reason="postprocessor_binding_or_parse_failure")
            issues.append(dict(run_index=row["run_index"], error=f"{type(exc).__name__}: {exc}"))
        rows.append(row)
    unknown = sorted(set(by_index) - {r["run_index"] for r in schedule})
    if unknown:
        issues.append(dict(extra_unplanned_run_indices=unknown))
    return rows, issues


def summaries(rows):
    groups = []
    for case in range(8):
        for method in METHODS:
            group = [r for r in rows if r["case_index"] == case and r["method"] == method]
            valid = [r for r in group if r["final_study_eligible"]]
            groups.append(dict(case_index=case, method=method, slice_id=group[0]["slice_id"],
                planned=3, observed=sum(r["observed"] for r in group), eligible=len(valid),
                all_three_eligible=len(group) == len(valid) == 3,
                median_scope="conditional on eligible returns; denominator reported; no failure imputation",
                initial_velocity_median_available=median([r["velocity_initial"] for r in group]),
                initial_velocity_available=sum(r["velocity_initial"] is not None for r in group),
                final_velocity_median_conditional=median([r["velocity_final_eligible"] for r in valid]),
                final_velocity_available_eligible=sum(r["velocity_final_eligible"] is not None for r in valid),
                common8_ratio_median_conditional=median([r["common8_ratio_eligible"] for r in valid]),
                common8_ratio_available_eligible=sum(r["common8_ratio_eligible"] is not None for r in valid),
                all_three_fit_available=sum(r["common8_ratio_eligible"] is not None for r in valid) == 3,
                actual_return_median_all_available=median([r["actual_api_return_seconds"] for r in group]),
                cap_diagnostic_available_rows=sum(r["cap_or_function_limit_diagnostic"] is not None for r in group),
                cap_or_function_limit_rows=sum(r["cap_or_function_limit_diagnostic"] is True for r in group)))
    return groups


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(output, rows, groups, protocol):
    # Deliberately deferred until explicit post-terminal execution.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ("#2171b5", "#238b45", "#88419d", "#d95f0e")
    jitter = (-.12, 0, .12)
    metrics = (("velocity", "velocity_initial", "velocity_final_eligible", "Physical velocity relative L2 error"),
               ("common8_fit", "common8_baseline_ratio", "common8_ratio_eligible", "Common-8 Φfinal / Φinitial"),
               ("actual_return", None, "actual_api_return_seconds", "Actual complete API return (s)"))
    for name, baseline, endpoint, ylabel in metrics:
        fig, axes = plt.subplots(4, 2, figsize=(12, 14), squeeze=False)
        for case, ax in enumerate(axes.flat):
            for position, (method, color) in enumerate(zip(METHODS, colors, strict=True)):
                group = sorted([r for r in rows if r["case_index"] == case and r["method"] == method],
                               key=lambda r: r["repetition"])
                summary = next(s for s in groups if s["case_index"] == case and s["method"] == method)
                for row in group:
                    x = position + jitter[row["repetition"]]
                    if baseline is not None and row[baseline] is not None:
                        ax.scatter(x, row[baseline], marker="D", facecolors="none", edgecolors="0.45", s=25)
                    y = row[endpoint]
                    if y is not None:
                        ax.scatter(x, y, color=color if row["final_study_eligible"] else "0.4",
                                   marker="o" if row["final_study_eligible"] else "x", s=32)
                    if y is None:
                        ax.plot(x, .025, marker="x", color="0.45", transform=ax.get_xaxis_transform())
                values = [r[endpoint] for r in group if r[endpoint] is not None]
                if values:
                    ax.plot([position-.22, position+.22], [median(values)]*2, color=color, lw=2.4)
                ax.text(position, -.14,
                    f"{len(values)}/3 metric; {summary['eligible']}/3 eligible\n"
                    f"{summary['cap_or_function_limit_rows']} flags/{summary['cap_diagnostic_available_rows']} known",
                    ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=8)
            if name == "actual_return":
                for value, style in ((8., "--"), (7.5, ":"), (3.75, ":")):
                    ax.axhline(value, ls=style, color="0.4", lw=.8)
            ax.set_title(f"Case {case}: {group[0]['slice_id']}", fontsize=10)
            ax.set_xticks(range(4), LABELS, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(axis="y", alpha=.18)
        legend = [Line2D([], [], marker="o", ls="", color="0.2", label="Eligible returned state: 3 rep slots"),
                  Line2D([], [], color="0.2", lw=2.4, label="Median of available plotted values; k/3 shown"),
                  Line2D([], [], marker="x", ls="", color="0.45", label="Unavailable/ineligible (bottom if no value)")]
        if baseline is not None:
            baseline_label = ("Initial model: offline diagnostic, each available rep" if name == "velocity"
                              else "Paid common-8 initial ratio 1, each available rep")
            legend.insert(0, Line2D([], [], marker="D", ls="", markerfacecolor="none", color="0.45", label=baseline_label))
        fig.legend(handles=legend, loc="upper center", ncol=2, fontsize=9)
        title = ("Returned recovery diagnostic" if name == "velocity" else "Returned common-8 data-fit diagnostic"
                 if name == "common8_fit" else "Actual consumed return time; 8s cap, 7.5/3.75s absolute engine ceilings")
        fig.suptitle(title, y=.995, fontsize=13)
        fig.text(.5, .012, "Reused 8 development slices; repetitions do not add geology. Original 4 uses fewer data.\n"
                 "Quality medians condition on eligible completed returns; no 8s failure/event imputation.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(top=.93, bottom=.065, hspace=.57, wspace=.26)
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=160)
        plt.close(fig)


def execute(args):
    root, output = args.input.resolve(), args.output.resolve()
    if args.expected_source_sha256 != sha(Path(__file__)):
        raise ValueError("explicit postprocessor source SHA mismatch")
    if output.exists() or not output.parent.is_dir() or output == root or root in output.parents:
        raise ValueError("fresh output outside producer directory required")
    output.mkdir()
    write_json(output / "status.json", dict(status="started", criteria="fixed before outcome reads", scope=SCOPE))
    try:
        receipt_path = root / "complete_receipt.json"
        if sha(receipt_path) != args.expected_complete_sha256:
            raise ValueError("root-confirmed terminal receipt SHA mismatch")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "complete_engineering_attempts":
            raise ValueError("terminal complete receipt required before scalar outcomes")
        if sha(root / "protocol.json") != PROTOCOL_SHA or sha(root / "driver_source.py") != DRIVER_SHA:
            raise ValueError("frozen producer source/protocol mismatch")
        before, after = receipt["source_before"], receipt["source_after"]
        if before != after or before["production"]["repository_source_sha256"] != SOURCE_SHA:
            raise ValueError("producer source before/after binding failed")
        manifest = receipt["artifact_sha256"]
        for name, expected in manifest.items():
            artifact = root / name
            if artifact.parent != root or not artifact.is_file() or sha(artifact) != expected:
                raise ValueError("producer manifest mismatch: " + name)
        protocol = json.loads((root / "protocol.json").read_text())
        rows, issues = collect(root, receipt, protocol)
        groups = summaries(rows)
        write_csv(output / "all96_runs.csv", rows)
        write_csv(output / "all32_case_method_medians.csv", groups)
        write_json(output / "scalar_report.json", dict(planned_runs=96, rows=len(rows), groups=len(groups),
            eligible=sum(r["final_study_eligible"] for r in rows), issues=issues,
            conditional_medians=True, complete_receipt_sha256=sha(receipt_path),
            source_sha256=sha(Path(__file__)), scope=SCOPE,
            target_scope="No first-hit/time-to-target inference from final returned states",
            timing_scope="All available actual return times; no cap/event imputation for missing returns",
            cap_scope="Raw per-stage native statuses/messages/counts retained; message flags diagnostic only"))
        plots(output, rows, groups, protocol)
        write_json(output / "complete_receipt.json", dict(status="complete" if not issues else "complete_with_issues",
            issues=issues, artifact_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
        return 0 if not issues else 2
    except BaseException as exc:
        write_json(output / "failure.json", dict(error=f"{type(exc).__name__}: {exc}",
            scope="All already-written postprocessing artifacts preserved"))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--root-terminal-go", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-complete-sha256")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(dict(status="source_only_default_no_outcome_reads", runs=96, cases=8,
            methods=METHODS, repetitions=3, required_terminal_GO=True, scope=SCOPE)))
        return 0
    if not args.root_terminal_go or any(getattr(args, key) is None for key in
        ("input", "output", "expected_source_sha256", "expected_complete_sha256")):
        parser.error("explicit root terminal GO, paths and both frozen hashes required")
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())
