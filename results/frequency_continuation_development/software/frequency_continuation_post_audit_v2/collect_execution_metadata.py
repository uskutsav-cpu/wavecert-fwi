"""Post-outcome stored-metadata collector; no arrays, policy or PDE execution."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

OUTER = Path(__file__).resolve().parents[3]
RUN = OUTER/"work/frequency_continuation_run_02"
REPORT = OUTER/"work/frequency_continuation_post_audit_run_02_v2/report.json"
OUTPUT = Path(__file__).with_name("ACTUAL_METADATA_SUMMARY_01.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUTPUT.exists()
    assert sha(REPORT) == "af3dd02c6b2e3ca6d7cf79e001a80c4af226ecdc0122aff45fc7eaf54106eb2b"
    report = json.loads(REPORT.read_bytes())
    assert report["verified_study_credit_enabled"] is True and report["issues"] == []
    complete = json.loads((RUN/"complete_receipt.json").read_bytes())
    stage_statuses, solver_messages = Counter(), Counter()
    callback_invocations, callback_points = [], []
    callback_cap_hits = 0
    cp_omitted = step_omitted = global_library = retained_library = 0
    for row in complete["runs"]:
        path = RUN/row["trajectory_artifact"]
        assert sha(path) == report["run_files_after"][path.name] == row["trajectory_sha256"]
        trace = json.loads(gzip.decompress(path.read_bytes()))
        for stage in trace["stages"]:
            stage_statuses[stage["status"]] += 1
            engine = stage.get("engine_result")
            if engine is None:
                continue
            invocations = engine["native_callback_invocations"]
            callback_invocations.append(invocations)
            callback_points.append(engine["native_callback_point_records"])
            callback_cap_hits += invocations >= stage["maximum_steps"]
            if engine.get("solver_result") is not None:
                solver_messages[engine["solver_result"]["message"]] += 1
            cp_omitted += engine["checkpoint_trace"]["omitted_count"]
            step_omitted += engine["step_trace"]["omitted_count"]
            global_library += engine["all_library_evaluation_trace"]["total_count"]
            retained_library += sum(len(step["library_evaluations"]) for step in engine["steps"])
    result = dict(report_sha256=sha(REPORT), producer_rows=len(complete["runs"]),
        verified_rows=sum(r["verified_credit_eligible"] for r in report["records"]),
        all3_verified_groups=sum(g["verified_repetitions"] == 3 for g in report["summary"]["groups"]),
        per_arm=report["summary"]["arms"], stage_status_counts=dict(stage_statuses),
        recorded_engine_stages=len(callback_invocations), callback_safety_cap_hits=callback_cap_hits,
        callback_invocations_range=[min(callback_invocations), max(callback_invocations)],
        callback_point_records_range=[min(callback_points), max(callback_points)],
        recorded_native_solver_messages=dict(solver_messages),
        actual_return_range_seconds=[min(r["actual_api_return_seconds"] for r in report["records"]),
            max(r["actual_api_return_seconds"] for r in report["records"])],
        omitted_middle_checkpoint_records=cp_omitted, omitted_middle_step_records=step_omitted,
        all_library_record_commitment_count=global_library, retained_child_library_samples=retained_library,
        scalar_middle_replay_claim=False, numerical_acceptance_changed=False,
        PDE_or_optimizer_calls=0, physical_acceptance=None,
        scope="Stored verified metadata only; all96 rows retained, omitted scalar middles remain commitments, correlated inputs/shared-machine timings, no winner or student design claim")
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
