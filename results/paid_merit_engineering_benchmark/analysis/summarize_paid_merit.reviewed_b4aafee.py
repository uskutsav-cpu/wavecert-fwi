"""Post-terminal descriptive summary; internal checkpoints are not returned results.

AI-authored engineering analysis. No IID, confirmation, competition, or student
ownership claim. This program never executes an optimizer or selects a method.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def read_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def median(values):
    return statistics.median(values) if values else None


def nonnegative_metric(value, *, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid numerical quality metric")
    if not math.isfinite(value) or value < 0:
        raise ValueError("Nonfinite/negative numerical quality metric")
    return value


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(run, output):
    status = read_json(run / "status.json")
    receipt = read_json(run / "complete_receipt.json")
    if status["status"] != "complete_engineering_benchmark":
        raise ValueError("Refuse partial/interrupted benchmark analysis")
    if receipt["status"] != status["status"] or receipt["runs"] != status["runs"]:
        raise ValueError("Final receipt/status disagree")
    for name, expected in receipt["artifacts"].items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Artifact differs: " + name)
    runs = status["runs"]
    protocol = read_json(run / "historical_candidate_protocol.json")
    saved_schedule = read_json(run / "schedule.json")
    modes = ("anytime", "target_stop_0.5", "target_stop_0.25")
    if len(runs) != 360 or Counter(r["measurement_mode"] for r in runs) != {
        "anytime": 120, "target_stop_0.5": 120, "target_stop_0.25": 120
    }:
        raise ValueError("Incomplete planned schedule")
    expected_keys = {(mode, method, case, repetition)
        for mode in modes for method in protocol["methods"]
        for case in range(8) for repetition in range(3)}
    observed_keys = [(r["measurement_mode"], r["method"], r["case_index"], r["repetition"])
                     for r in runs]
    if len(set(observed_keys)) != 360 or set(observed_keys) != expected_keys:
        raise ValueError("Missing/duplicate Cartesian schedule entry")
    if len(saved_schedule) != 360 or [r["run_index"] for r in runs] != list(range(360)):
        raise ValueError("Run indices differ")
    stable_keys = ("run_index", "case_index", "repetition", "method", "measurement_mode",
                   "target_ratio", "original_schedule_index", "order_position", "identity")
    for row, planned in zip(runs, saved_schedule):
        if any(row[k] != planned[k] for k in stable_keys):
            raise ValueError("Run differs from saved schedule")
        if row["identity"] != protocol["identities"][row["case_index"]]:
            raise ValueError("Case identity differs from frozen protocol")
    maximum_budget = float(protocol["trajectory"]["budget_seconds"])
    output.mkdir(exist_ok=False)
    targets, internal, manifest_rows = [], [], []
    for row in runs:
        valid = row["status"] == "complete" and row.get("final_study_eligible") is True
        common = {k: row[k] for k in (
            "run_index", "repetition", "case_index", "method", "measurement_mode"
        )}
        common.update(provider_id=row["identity"]["parent_id"],
                      slice_id=row["identity"]["slice_id"],
                      terminal_status=row["status"], study_eligible=valid)
        manifest_rows.append(common)
        if row["measurement_mode"] != "anytime":
            event = row.get("end_to_end_target", {})
            eligible = valid and event.get("eligible") is True
            quality = None
            if valid:
                audit = read_json(run / row["audit_artifact"])
                quality = audit["credited_reconstruction_quality"]["fully_returned_target_state"]
            actual_return = row.get("timing", {}).get("actual_start_to_control_return_seconds")
            if eligible:
                seconds = event.get("elapsed_seconds")
                if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
                    raise ValueError("Invalid eligible target time")
                if not 0 <= seconds == actual_return <= maximum_budget or quality is None:
                    raise ValueError("Eligible target time/quality contradicts paid return gate")
                for key in ("relative_physical_velocity_L2", "relative_squared_slowness_L2"):
                    nonnegative_metric(quality[key])
            reason = event.get("reason", row.get("error", "unknown"))
            if not valid:
                reason = "study_ineligible:" + row.get("error", row["status"])
            targets.append(dict(common, target_ratio=row["target_ratio"],
                target_eligible=eligible,
                target_seconds=event.get("elapsed_seconds") if eligible else None,
                actual_return_seconds=actual_return, reason=reason,
                runtime_reason=event.get("reason"),
                engine_status=row.get("engine_status"),
                relative_velocity_L2=quality["relative_physical_velocity_L2"] if eligible else None,
                relative_squared_slowness_L2=quality["relative_squared_slowness_L2"] if eligible else None))
        elif valid:
            audit = read_json(run / row["audit_artifact"])
            endpoints = audit["budget_endpoint_diagnostics"]
            qualities = audit["credited_reconstruction_quality"]["internal_budget_states"]
            if len(endpoints) != 4 or len(qualities) != 4:
                raise ValueError("Endpoint/quality lengths differ")
            for endpoint, quality in zip(endpoints, qualities):
                if endpoint["budget_seconds"] != quality["budget_seconds"]:
                    raise ValueError("Endpoint/quality budgets differ")
                paid = endpoint["completed_paid_checkpoint"] is True
                if paid != quality["eligible_paid_checkpoint"]:
                    raise ValueError("Unpaid quality credit")
                q = quality["quality"]
                if paid:
                    nonnegative_metric(endpoint["offline_objective_ratio"], nullable=True)
                    for key in ("relative_physical_velocity_L2", "relative_squared_slowness_L2"):
                        nonnegative_metric(q[key])
                internal.append(dict(common, budget_seconds=endpoint["budget_seconds"],
                    internal_checkpoint_eligible=paid,
                    checkpoint_seconds=endpoint["checkpoint_elapsed_seconds"],
                    objective_ratio=endpoint["offline_objective_ratio"] if paid else None,
                    relative_velocity_L2=q["relative_physical_velocity_L2"] if paid else None,
                    relative_squared_slowness_L2=q["relative_squared_slowness_L2"] if paid else None,
                    returned_API_evidence=False))
        else:
            for budget in (1, 2, 4, 8):
                internal.append(dict(common, budget_seconds=budget,
                    internal_checkpoint_eligible=False, checkpoint_seconds=None,
                    objective_ratio=None, relative_velocity_L2=None,
                    relative_squared_slowness_L2=None, returned_API_evidence=False))
    methods = sorted({r["method"] for r in runs})
    target_summary = []
    case_summary = []
    for mode in ("target_stop_0.5", "target_stop_0.25"):
        for method in methods:
            rows = [r for r in targets if r["measurement_mode"] == mode and r["method"] == method]
            successful = [r for r in rows if r["target_eligible"]]
            target_summary.append(dict(mode=mode, method=method, planned=len(rows),
                eligible_targets=len(successful), ineligible_or_censored=len(rows)-len(successful),
                median_seconds_among_eligible_ONLY=median([r["target_seconds"] for r in successful]),
                failure_reasons=dict(Counter(r["reason"] for r in rows if not r["target_eligible"]))))
            for case in range(8):
                group = [r for r in rows if r["case_index"] == case]
                observed = [r["target_seconds"] for r in group if r["target_eligible"]]
                case_summary.append(dict(mode=mode, method=method, case_index=case,
                    provider_id=group[0]["provider_id"], slice_id=group[0]["slice_id"],
                    planned=len(group), eligible_targets=len(observed),
                    all_three_eligible=len(group) == len(observed) == 3,
                    median_observed_seconds=median(observed),
                    median_all_three_seconds=median(observed) if len(group) == len(observed) == 3 else None))
    summary = dict(
        role="AI-authored descriptive SOFTWARE analysis; all eight cases reused development",
        planned_runs=360, terminal_counts=dict(Counter(r["status"] for r in runs)),
        target_summary=target_summary,
        internal_availability=[dict(method=method, budget_seconds=budget,
            planned=sum(r["method"] == method and r["budget_seconds"] == budget for r in internal),
            eligible=sum(r["method"] == method and r["budget_seconds"] == budget
                         and r["internal_checkpoint_eligible"] for r in internal))
            for budget in (1, 2, 4, 8) for method in methods],
        internal_diagnostics_only=True,
        target_censoring="Missing targets have null times; 8 seconds is never substituted as an observed event",
        medians_scope="Eligible-only medians are conditional; inspect every failure denominator and case row",
        dependence="Three repetitions and slices do not create independent geological sites; SEAM IDs share ancestry",
        statistical_inference="No IID confidence interval, p-value, unconditional speedup, or winner promotion",
        scientific_goal_complete=False,
    )
    write_csv(output / "all_runs.csv", manifest_rows)
    write_csv(output / "returned_targets.csv", targets)
    write_csv(output / "internal_checkpoints.csv", internal)
    write_csv(output / "case_target_summary.csv", case_summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
