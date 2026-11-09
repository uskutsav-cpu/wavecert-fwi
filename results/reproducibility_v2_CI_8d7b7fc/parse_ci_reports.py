"""Parse preserved decoded CI logs; no solver or project imports."""

import hashlib
import json
import re
from pathlib import Path

EXPECTED = {
    "bp_numpy_tangent_reproduction",
    "coordinate_offline_import_integration",
    "frequency_continuation_archival_manufactured_reproduction",
    "seven_faithful_NumPy_traces",
    "eight_initial_raw_direction_witnesses",
    "recovery_metric_initial_slopes",
    "analytic_homogeneous_wave_reference",
}
JOBS = {128913003176: "test (3.11)", 128913003345: "test (3.12)", 128913003377: "test (3.10)", 128913003456: "learned-and-data"}
SOURCE = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
INPUTS = "2d878a87d947e8b4f1e9e69a5e487ab772e3de1156511a714a0a3b012fad3b35"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reports(raw):
    decoder = json.JSONDecoder()
    result = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        for match in re.finditer(r"\{", line):
            try:
                value, end = decoder.raw_decode(line[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("check") in EXPECTED:
                text = line[match.start():match.start() + end]
                result.append({"line_number": line_number, "raw_JSON_sha256": sha(text.encode()), "report": value})
                break
    return result


def validate(rows):
    issues = []

    def require(label, condition):
        if not condition:
            issues.append(label)

    require("exact seven-report denominator", len(rows) == 7 and {r["report"]["check"] for r in rows} == EXPECTED)
    if issues:
        return issues
    values = {r["report"]["check"]: r["report"] for r in rows}
    bp = values["bp_numpy_tangent_reproduction"]
    require("BP denominator", bp["traces"] == 3 and bp["informative_directions"] == 6 and [r["case_id"] for r in bp["cases"]] == ["upper_left", "middle_center", "lower_right"])
    require("BP fixed canonical inputs and truthful flags", bp["canonical_input_manifest_sha256"] == INPUTS and bp["native_derived_numeric_inputs_used"] is True and bp["native_solution_outputs_used_during_own_solves"] is False and bp["independent_numeric_input_generation"] is False and bp["all_own_solves_before_native_solution_outputs"] is True)
    require("BP unchanged thresholds/no physical claim", bp["trace_tolerance"] == 1e-9 and bp["directional_tolerance"] == 1e-3 and bp["physical_acceptance"] is None)
    for case in bp["cases"]:
        require(case["case_id"] + " fixed trace/source", 0 <= case["trace_relative_L2"] <= 1e-9 and 0 <= case["source_relative_L2"] <= 1e-14)
        require(case["case_id"] + " six informative derivatives", set(case["informative_direction_relative_errors"]) == {"interior", "edge"} and all(0 <= value <= 1e-3 for value in case["informative_direction_relative_errors"].values()))
    require("BP reported maxima", bp["maximum_trace_relative_L2"] == max(r["trace_relative_L2"] for r in bp["cases"]) and bp["maximum_directional_relative_error"] == max(value for r in bp["cases"] for value in r["informative_direction_relative_errors"].values()))
    f = values["frequency_continuation_archival_manufactured_reproduction"]
    require("108 explicit actual denominator", f["status"] == "pass_all108" and f["expected_total"] == f["actual_observed"] == f["actual_passed"] == 108)
    require("108 production/source/scope", f["production_files"] == 161 and f["production_sha256"] == SOURCE and f["scientific_claim"] is False)
    expected = [("bounded_engine", 17), ("controller", 24), ("driver", 15), ("launcher", 15), ("scalar_auditor_v2", 37)]
    require("108 all five actual child processes", len(f["suites"]) == 5)
    for suite, (name, count) in zip(f["suites"], expected, strict=True):
        require("108 " + name, suite["name"] == name and suite["actual_exit_code"] == 0 and suite["status"] == "passed" and suite["expected_passed"] == suite["observed"] == suite["passed"] == count and suite["failed"] == suite["errors"] == suite["skipped"] == 0)
    traces = values["seven_faithful_NumPy_traces"]
    require("seven faithful traces", len(traces["cases"]) == 7 and all(r["agreement"] is True for r in traces["cases"]) and traces["all_own_traces_computed_before_targets"] is True and traces["maximum_trace_relative_L2"] <= traces["trace_tolerance"] == 1e-9)
    witnesses = values["eight_initial_raw_direction_witnesses"]
    require("eight stored direction witnesses", witnesses["directions"] == 8 and len(witnesses["cases"]) == 8 and witnesses["checks"] == 1289 and witnesses["false_certifications"] == witnesses["literal_bound_misses"] == 0)
    recovery = values["recovery_metric_initial_slopes"]
    require("recovery stored software diagnostics", recovery["checks"] == 4483 and recovery["truth_feasible"] == recovery["initial_feasible"] == len(recovery["cases"]) == 8)
    analytic = values["analytic_homogeneous_wave_reference"]
    require("analytic diagnostic without acceptance claim", analytic["internal_numeric_gates"] is True and analytic["physical_quality_pass"] is None and len(analytic["cases"]) == 7)
    coordinate = values["coordinate_offline_import_integration"]
    require("coordinate synthetic API", coordinate["checks"] == 9 and coordinate["numeric_comparisons"] == 6 and coordinate["synthetic_shape"] == [24, 24] and coordinate["refined_shape"] == [49, 49])
    return issues


def write_exact_or_new(path, value):
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError("refuse to overwrite changed extraction: " + str(path))
    else:
        path.write_text(text)


def main():
    folder = Path(__file__).resolve().parent
    output = []
    for job, name in JOBS.items():
        path = folder / f"{job}.log"
        if not path.exists():
            continue
        raw = path.read_text()
        rows = reports(raw)
        issues = validate(rows)
        summaries = re.findall(r"(?m)^.*?Z\s+(\d+ passed[^\n]*\bin [^\n]+)$", raw)
        found_errors = [line for line in raw.splitlines() if "##[error]" in line or re.search(r"\bFAILED\s+tests/", line)]
        record = {"job_id": job, "job_name": name, "log_sha256": sha(path.read_bytes()), "log_bytes": path.stat().st_size, "reports_count": len(rows), "reports": rows, "report_validation_issues": issues, "pytest_summary_lines": summaries, "error_lines": found_errors}
        write_exact_or_new(folder / f"{job}.reports.json", record)
        values = {row["report"]["check"]: row["report"] for row in rows}
        bp = values.get("bp_numpy_tangent_reproduction", {})
        f = values.get("frequency_continuation_archival_manufactured_reproduction", {})
        output.append({"job_id": job, "name": name, "reports": len(rows), "report_validation_issues": issues, "pytest": summaries, "error_lines": len(found_errors), "python": bp.get("python"), "numpy": bp.get("numpy"), "BP_max_trace_relative_L2": bp.get("maximum_trace_relative_L2"), "BP_max_DD_relative_error": bp.get("maximum_directional_relative_error"), "actual_frequency_controls_passed": f.get("actual_passed")})
    print(json.dumps({"parsed_jobs": output}, indent=2))
    return 2 if any(r["report_validation_issues"] or r["error_lines"] for r in output) else 0


if __name__ == "__main__":
    raise SystemExit(main())
