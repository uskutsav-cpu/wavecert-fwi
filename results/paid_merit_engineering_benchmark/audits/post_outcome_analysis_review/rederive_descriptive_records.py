"""Independent descriptive arithmetic from retained audit/status records; no PDE."""
from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

METHODS = ["exact_lu_projected", "exact_lu_lbfgsb", "same_M_gmres_projected",
           "fixed_sweeps_uncertified", "fixed_sweeps_certified"]
MODES = ["anytime", "target_stop_0.5", "target_stop_0.25"]
LABELS = dict(zip(METHODS, ["Exact LU, projected descent", "Exact LU, L-BFGS-B",
                          "Same-M GMRES, projected descent", "Fixed sweeps, uncertified",
                          "Fixed sweeps, certified"], strict=True))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = Path(__file__).resolve().parents[2]
    repo = base / "wavecert-fwi"
    archive = repo / "results" / "paid_merit_engineering_benchmark"
    outer = base / "paid_merit_initial_baseline_diagnostic_01"
    copied = archive / "post_outcome_initial_baseline"
    paths = {
        "objective_audit": archive / "audits" / "objective_report_01.json",
        "receipt_audit": archive / "audits" / "receipt_report_01.json",
        "raw_status": archive / "run_01" / "status.json",
        "analysis_summary": archive / "analysis" / "summary.json",
        "baseline_csv": copied / "initial_baseline_comparison.csv",
        "baseline_summary": copied / "summary.json",
        "baseline_caption": copied / "caption.txt",
        "baseline_source": copied / "initial_baseline_diagnostic.py",
        "main_doc": repo / "docs" / "paid_merit_engineering_benchmark.md",
        "artifact_readme": archive / "README.md",
    }
    before = {k: sha(v) for k, v in paths.items()}
    own_before = sha(Path(__file__))
    audit = json.loads(paths["objective_audit"].read_bytes())
    receipt_audit = json.loads(paths["receipt_audit"].read_bytes())
    status = json.loads(paths["raw_status"].read_bytes())
    analysis = json.loads(paths["analysis_summary"].read_bytes())
    baseline = json.loads(paths["baseline_summary"].read_bytes())
    with paths["baseline_csv"].open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    checks = []

    def check(name, condition):
        checks.append({"id": name, "passed": bool(condition)})

    close_count = 0
    max_abs_difference = 0.0

    def close(name, actual, expected):
        nonlocal close_count, max_abs_difference
        close_count += 1
        max_abs_difference = max(max_abs_difference, abs(actual - expected))
        check(name, math.isfinite(actual) and math.isfinite(expected)
              and math.isclose(actual, expected, rel_tol=5e-11, abs_tol=2e-13))

    check("objective_audit_passed", audit["status"] == "passed" and not audit["failures"])
    check("objective_audit_record_denominators", len(audit["records"]) == audit["runs_with_saved_models_checked"] == 360
          and audit["saved_model_slots_checked"] == 2160 and audit["distinct_case_model_sha_from_saved_slots"] == 708
          and audit["numeric_comparisons"] == 12928)
    check("receipt_audit_passed", receipt_audit["result"] == "CLEAR_RECEIPT_CHECKS" and not receipt_audit["issues"]
          and receipt_audit["checks"] == 232675)
    planned = set(itertools.product(range(8), METHODS, range(3), MODES))
    raw_keys = [(r["case_index"], r["method"], r["repetition"], r["measurement_mode"]) for r in status["runs"]]
    audit_keys = [(r["case_index"], r["method"], r["repetition"], r["measurement_mode"]) for r in audit["records"]]
    check("raw_exact_360_cartesian_unique", len(raw_keys) == len(set(raw_keys)) == 360 and set(raw_keys) == planned)
    check("audit_exact_360_cartesian_unique", len(audit_keys) == len(set(audit_keys)) == 360 and set(audit_keys) == planned)
    raw_by_key = dict(zip(raw_keys, status["runs"], strict=True))
    audit_by_index = {r["run_index"]: r for r in audit["records"]}
    mode_method_counts = Counter((r["measurement_mode"], r["method"]) for r in status["runs"])
    check("all_fifteen_method_modes_have_24_runs", set(mode_method_counts.values()) == {24}
          and len(mode_method_counts) == 15)
    check("all_360_final_study_eligible_complete", all(r["status"] == "complete"
          and r["final_study_eligible"] is True for r in status["runs"]))
    initial = {}
    for r in audit["records"]:
        p = r["saved_models"]["prepared_initial"]
        v = (p["model_sha256"], p["relative_physical_velocity_L2"], p["relative_squared_slowness_L2"], p["objective"])
        initial.setdefault(r["case_index"], v)
        check("initial_invariance_run_" + str(r["run_index"]), v == initial[r["case_index"]])
    check("csv_exact_all_forty_case_methods", len(csv_rows) == 40 and
          {(int(r["case_index"]), r["method"]) for r in csv_rows} == set(itertools.product(range(8), METHODS)))
    csv_map = {(int(r["case_index"]), r["method"]): r for r in csv_rows}
    reproduced = []
    for case, method in itertools.product(range(8), METHODS):
        group = [r for r in audit["records"] if r["case_index"] == case and r["method"] == method
                 and r["measurement_mode"] == "anytime"]
        key = str(case) + "_" + method
        check(key + "_exact_three_unique_repetitions", len(group) == 3 and {r["repetition"] for r in group} == {0, 1, 2})
        eligible = [r for r in group if r["saved_models"]["internal_budget_3"]["verified_credit_eligible"] is True
                    and r["terminal_final_study_eligible"] is True and r["row_status"] == "complete"]
        check(key + "_all_three_paid_internal_checkpoints", len(eligible) == 3)
        row = {"case_index": case, "method": method, "eligible": len(eligible), "internal_only": True}
        saved_csv = csv_map[(case, method)]
        check(key + "_csv_denominators", saved_csv["planned"] == saved_csv["eligible"] == "3"
              and saved_csv["internal_only"] == "True")
        raw_identities = [raw_by_key[(case, method, rep, "anytime")]["identity"]["slice_id"] for rep in range(3)]
        check(key + "_slice_identity", len(set(raw_identities)) == 1 and saved_csv["slice_id"] == raw_identities[0])
        row["slice_id"] = raw_identities[0]
        for metric, field, initial_index in [("velocity", "relative_physical_velocity_L2", 1),
                                             ("squared_slowness", "relative_squared_slowness_L2", 2)]:
            start = initial[case][initial_index]
            values = [r["saved_models"]["internal_budget_3"][field] for r in eligible]
            end = statistics.median(values) if len(values) == 3 else None
            ratio = end / start if end is not None and start > 0 else None
            row[metric] = {"initial_error": start, "repetition_errors": values, "median_8s_error": end,
                           "error_over_initial": ratio, "improved": end < start if end is not None else None}
            if end is not None and ratio is not None:
                for suffix, value in [("initial_error", start), ("8s_median_error", end), ("error_over_initial", ratio)]:
                    close(key + "_" + metric + "_" + suffix, value, float(saved_csv[metric + "_" + suffix]))
                check(key + "_" + metric + "_improved_flag", str(end < start) == saved_csv[metric + "_improved"])
        row["objective_8s_median_over_initial"] = statistics.median(
            r["saved_models"]["internal_budget_3"]["objective"] for r in eligible) / initial[case][3]
        reproduced.append(row)
    method_summary = []
    for method in METHODS:
        rows = [r for r in reproduced if r["method"] == method]
        result = {"method": method, "planned_cases": 8}
        for metric in ["velocity", "squared_slowness"]:
            result[metric + "_improved"] = sum(r[metric]["improved"] is True for r in rows)
            result[metric + "_worse"] = sum(r[metric]["median_8s_error"] > r[metric]["initial_error"] for r in rows)
            result[metric + "_unavailable"] = sum(r[metric]["improved"] is None for r in rows)
        saved = next(r for r in baseline["method_summary"] if r["method"] == method)
        check(method + "_baseline_summary_counts", all(result[k] == saved[k] for k in
              ["planned_cases", "velocity_improved", "velocity_worse"]) and saved["unavailable"] == 0)
        method_summary.append(result)
    by_case_method = {(r["case_index"], r["method"]): r for r in reproduced}
    certificate_better_cases = [case for case in range(8) if by_case_method[(case, "fixed_sweeps_certified")]["velocity"]["median_8s_error"]
                                < by_case_method[(case, "exact_lu_lbfgsb")]["velocity"]["median_8s_error"]]
    all_arms_worse_in_those_cases = all(by_case_method[(case, method)]["velocity"]["error_over_initial"] > 1
                                      for case in certificate_better_cases for method in METHODS)
    check("doc_certificate_better_5_but_all_five_arms_worse_initial", len(certificate_better_cases) == 5
          and all_arms_worse_in_those_cases)

    target_events, late_reached = [], []
    for row in status["runs"]:
        if row["measurement_mode"] == "anytime":
            check("anytime_has_no_return_target_credit_" + str(row["run_index"]), row["end_to_end_target"]["eligible"] is False)
            continue
        core_path = archive / "run_01" / row["core_artifact"]
        check("raw_core_hash_" + str(row["run_index"]), sha(core_path) == row["core_sha256"])
        core = json.loads(gzip.decompress(core_path.read_bytes()))
        trajectory = core["trajectory"]
        wall = row["timing"]["actual_start_to_control_return_seconds"]
        phi0, phi = trajectory["initial_objective"], trajectory["final_objective"]
        reached = math.isfinite(phi0) and phi0 > 0 and math.isfinite(phi) and phi <= row["target_ratio"] * phi0
        eligible = bool(reached and trajectory["status"] == "objective_target" and trajectory["costs_complete"] is True
                        and math.isfinite(wall) and 0 <= wall <= 8
                        and row["status"] == "complete" and row["final_study_eligible"] is True)
        stored = row["end_to_end_target"]
        check("rederived_return_eligibility_" + str(row["run_index"]), eligible == stored["eligible"]
              == audit_by_index[row["run_index"]]["verified_returned_target_credit_eligible"])
        check("real_return_or_null_event_time_" + str(row["run_index"]), stored["elapsed_seconds"] == wall
              if eligible else stored["elapsed_seconds"] is None)
        if reached and trajectory["status"] == "objective_target" and wall > 8:
            late_reached.append(row["run_index"])
        target_events.append({"run_index": row["run_index"], "mode": row["measurement_mode"], "method": row["method"],
                              "eligible": eligible, "actual_return_seconds": wall, "event_seconds": wall if eligible else None})
    target_summary = []
    for mode, method in itertools.product(MODES[1:], METHODS):
        group = [r for r in target_events if r["mode"] == mode and r["method"] == method]
        times = [r["event_seconds"] for r in group if r["eligible"]]
        row = {"mode": mode, "method": method, "planned": len(group), "eligible": len(times),
               "censored_or_ineligible": len(group) - len(times),
               "conditional_median_return_seconds": statistics.median(times) if times else None}
        saved = next(r for r in analysis["target_summary"] if r["mode"] == mode and r["method"] == method)
        check(mode + method + "_analysis_counts", saved["planned"] == len(group) == 24
              and saved["eligible_targets"] == len(times) and saved["ineligible_or_censored"] == 24 - len(times))
        if times:
            close(mode + method + "_analysis_conditional_median", row["conditional_median_return_seconds"],
                  saved["median_seconds_among_eligible_ONLY"])
        target_summary.append(row)
    doc = paths["main_doc"].read_text()
    table = {cells[0]: [int(cells[1]), int(cells[2])] for line in doc.splitlines()
             if line.startswith("| ") and (cells := [x.strip() for x in line.split("|")[1:-1]])[0] in LABELS.values()}
    for method in METHODS:
        counts = [next(r["eligible"] for r in target_summary if r["mode"] == mode and r["method"] == method) for mode in MODES[1:]]
        check(method + "_doc_returned_table", table.get(LABELS[method]) == counts)
    for method, display in [("exact_lu_lbfgsb", "0.690"), ("exact_lu_projected", "0.914"),
                            ("same_M_gmres_projected", "0.627"), ("fixed_sweeps_uncertified", "1.688"),
                            ("fixed_sweeps_certified", "2.271")]:
        half = next(r for r in target_summary if r["mode"] == "target_stop_0.5" and r["method"] == method)
        check(method + "_doc_rounded_conditional_median", f'{half["conditional_median_return_seconds"]:.3f}' == display and display + " s" in doc)
    counts = dict(Counter(r["engine_status"] for r in status["runs"]))
    check("doc_terminal_engine_counts", counts == {"budget_exhausted": 190, "solver_terminated": 5, "objective_target": 165})
    check("doc_164_eligible_240_targets_one_late", len(target_events) == 240
          and sum(r["eligible"] for r in target_events) == 164 and len(late_reached) == 1)
    for name in ["initial_baseline_comparison.csv", "summary.json", "caption.txt"]:
        check("outer_and_archived_" + name, (outer / name).read_bytes() == (copied / name).read_bytes())
    check("postoutcome_source_byte_exact", (base / "new-phase-staging" / "initial_baseline_diagnostic.py").read_bytes()
          == paths["baseline_source"].read_bytes())
    check("baseline_diagnostic_source_and_audit_pins", baseline["audit_sha256"] == before["objective_audit"]
          and baseline["source_sha256"] == before["baseline_source"])
    check("postoutcome_explicit_in_doc_caption_summary", "after inspecting the outcomes" in doc
          and "post-outcome diagnostic" in doc and "POST-OUTCOME" in baseline["role"]
          and "after outcome inspection" in paths["baseline_caption"].read_text())
    after = {k: sha(v) for k, v in paths.items()}
    check("review_inputs_unchanged", before == after)
    check("review_source_unchanged", own_before == sha(Path(__file__)))
    report = {"role": "AI independent post-outcome descriptive record review, not human review or student scientific interpretation",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": own_before, "input_sha256_before": before, "input_sha256_after": after,
              "arithmetic_scope": "All 40 case-method medians rederived directly from independent objective-audit anytime internal_budget_3 and prepared_initial records. Target events rederived from raw status/core trajectory and actual return times. No optimizer/model/PDE reruns.",
              "numeric_comparison_tolerance": {"relative": 5e-11, "absolute": 2e-13,
                                               "basis": "Retained independent saved-model quality-audit engineering tolerance; no statistical inference."},
              "checks": checks, "numeric_comparisons": close_count, "max_numeric_absolute_difference": max_abs_difference,
              "mode_method_counts": [{"mode": k[0], "method": k[1], "planned": v} for k, v in sorted(mode_method_counts.items())],
              "all_forty_case_method_rows": reproduced, "method_reconstruction_summary": method_summary,
              "certificate_lower_velocity_error_than_lbfgsb_cases": certificate_better_cases,
              "all_five_arms_worse_than_initial_on_those_cases": all_arms_worse_in_those_cases,
              "returned_target_summary": target_summary, "late_objective_target_runs": late_reached,
              "engine_status_counts": counts,
              "scope_review": "Counts/medians do not support unique certificate safety, unconditional speed, reliable recovery, independent geology, causal explanation, new confirmation or student ownership. Initial comparison is explicitly post-outcome. Dimensional bound is a rank upper bound, not a diagnosis of observed drift.",
              "failed_checks": [r for r in checks if not r["passed"]]}
    report["status"] = "PASS_DESCRIPTIVE_RECORD_REVIEW" if not report["failed_checks"] else "FAIL_DESCRIPTIVE_RECORD_REVIEW"
    output = Path(__file__).with_name("descriptive_review_01.json")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(checks), "numeric_comparisons": close_count,
                      "failures": len(report["failed_checks"]), "report_sha256": sha(output),
                      "method_summary": method_summary, "cert_better_cases": certificate_better_cases,
                      "target_summary": target_summary}))
    return 0 if not report["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
