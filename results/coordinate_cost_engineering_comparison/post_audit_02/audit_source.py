"""Independent scalar collector; never execute before root confirms terminal."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def atomic_report(path, value):
    temporary = path.with_name(path.name + ".tmp")
    try:
        write_new(temporary, value)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw)


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


class Checks:
    def __init__(self):
        self.count = self.numeric_count = 0
        self.issues = []

    def check(self, condition, label, detail=None):
        self.count += 1
        if not condition:
            self.issues.append(dict(check=label, detail=detail))
        return bool(condition)

    def near(self, actual, expected, label, rtol, atol):
        self.numeric_count += 1
        return self.check(finite(actual) and finite(expected)
                          and abs(actual - expected) <= atol + rtol * abs(expected),
                          label, dict(actual=actual if finite(actual) else repr(actual),
                                      expected=expected if finite(expected) else repr(expected), rtol=rtol, atol=atol))


def expected_schedule(criteria):
    rng, rows = random.Random(criteria["seed"]), []
    for repetition in range(criteria["repetitions"]):
        for case in range(criteria["cases"]):
            methods = list(criteria["methods"])
            rng.shuffle(methods)
            for position, method in enumerate(methods):
                rows.append(dict(run_index=len(rows), case_index=case, repetition=repetition,
                    order_position=position, method=method, slice_id=criteria["slice_ids"][case],
                    optimizer_coordinates="physical_velocity" if method == "physical_v_lbfgsb" else "squared_slowness"))
    return rows


def source_manifest(repo):
    files = [repo / "pyproject.toml"]
    files += sorted((repo / "src").rglob("*.py"))
    files += sorted(p for p in (repo / "configs").rglob("*") if p.is_file())
    files += sorted(p for p in (repo / "scripts").rglob("*") if p.suffix in {".py", ".sh"})
    files += sorted((repo / ".github/workflows").glob("*.yml"))
    files += sorted(repo.glob("requirements*.txt"))
    manifest = {p.relative_to(repo).as_posix(): sha(p) for p in sorted(set(files)) if p.is_file()}
    digest = hashlib.sha256()
    for name, checksum in manifest.items():
        digest.update(name.encode() + b"\0" + checksum.encode() + b"\0")
    return dict(digest=digest.hexdigest(), files=manifest)


def artifact(run, name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("unsafe artifact name")
    path = (run / name).resolve()
    if path.parent != run or not path.is_file():
        raise ValueError("missing or escaping artifact: " + name)
    return path


def array_sha(values, np):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def counts(value, prefix=""):
    if not isinstance(value, dict):
        raise ValueError("counter branch must be a dictionary")
    result = {}
    for key, item in value.items():
        name = prefix + "/" + str(key)
        if isinstance(item, dict):
            result.update(counts(item, name))
        elif type(item) is int and item >= 0:
            result[name] = item
        else:
            raise ValueError("non-count value at " + name)
    return result


def runtime_gate(trajectory, wall, cap):
    paid = bool(trajectory.get("checkpoints") and finite(trajectory.get("initial_objective"))
                and finite(trajectory.get("final_objective")))
    if trajectory.get("costs_complete") is not True or trajectory.get("status") == "failed":
        return False, "failed_or_incomplete_method"
    if not paid:
        return False, "no_paid_final_state"
    if not finite(wall) or wall < 0 or wall > cap:
        return False, "late_or_invalid_actual_api_return"
    return True, "fully_returned_paid_state"


def reconstruct_truth(raw, np):
    """Own scalar half-cell bilinear velocity interpolation; no helper import."""
    if raw.shape != (24, 24) or not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError("positive24x24 coarse model required")
    velocity = [[1. / math.sqrt(float(raw[z, x])) for x in range(24)] for z in range(24)]
    values = []
    for z in range(49):
        coord_z = .5 * (z + 1)
        iz, fz = math.floor(coord_z), coord_z - math.floor(coord_z)
        z0, z1 = max(0, min(23, iz - 1)), max(0, min(23, iz))
        for x in range(49):
            coord_x = .5 * (x + 1)
            ix, fx = math.floor(coord_x), coord_x - math.floor(coord_x)
            x0, x1 = max(0, min(23, ix - 1)), max(0, min(23, ix))
            v = math.fsum(((1-fz)*(1-fx)*velocity[z0][x0], (1-fz)*fx*velocity[z0][x1],
                           fz*(1-fx)*velocity[z1][x0], fz*fx*velocity[z1][x1]))
            values.append(1. / (v*v))
    return np.asarray(values, dtype="<f8")


def scalar_errors(model, truth):
    m = [float(x) for x in model]
    t = [float(x) for x in truth]
    v = [1. / math.sqrt(x) for x in m]
    vt = [1. / math.sqrt(x) for x in t]
    return dict(relative_squared_slowness_L2=math.sqrt(math.fsum((a-b)*(a-b) for a, b in zip(m, t, strict=True))
                                                     / math.fsum(x*x for x in t)),
        relative_physical_velocity_L2=math.sqrt(math.fsum((a-b)*(a-b) for a, b in zip(v, vt, strict=True))
                                               / math.fsum(x*x for x in vt)))


def median3(values):
    if len(values) != 3 or not all(finite(x) for x in values):
        raise ValueError("three finite repetitions required")
    return sorted(values)[1]


def recorded_status_counts(rows):
    labels = sorted({row.get("recorded_status", "unavailable") for row in rows})
    return {label:sum(row.get("recorded_status", "unavailable") == label for row in rows) for label in labels}


def check_timing(timing, checks, label, criteria):
    tol = criteria["numeric_agreement"]["timing_nesting_atol_seconds"]
    actual = timing["actual_api_return_seconds"]
    eng_return = timing["engine_call_return_seconds"]
    eng = timing["engine_reported_seconds_including_preparation"]
    prefix = timing["pre_engine_seconds"]
    checks.check(all(finite(x) and x >= 0 for x in (actual, eng_return, eng, prefix))
                 and prefix <= eng+tol and eng <= eng_return+tol and eng_return <= actual+tol,
                 label + "/timing_nesting")
    atol = criteria["numeric_agreement"]["derived_timing_atol_seconds"]
    for key, expected in [("engine_finish_tail_seconds", max(0., eng_return-eng)),
                          ("api_output_tail_seconds", max(0., actual-eng_return)),
                          ("actual_api_overshoot_seconds", max(0., actual-criteria["api_return_cap_seconds"]))]:
        checks.near(timing[key], expected, label+"/"+key, 0., atol)
    for key in ["input_loading_hash_seconds", "preparation_seconds", "coordinate_preparation_seconds",
                "timed_progress_write_seconds", "final_native_conversion_hash_seconds"]:
        checks.check(finite(timing[key]) and 0 <= timing[key] <= actual+tol, label+"/subset/"+key)
    checks.check(timing["preparation_and_progress_are_subsets_never_added_twice"] is True,
                 label + "/no_prefix_double_charge")
    checks.check(timing["engine_budget_seconds"] == 7.5 and timing["fixed_api_cap_seconds"] == 8.
                 and timing["reserve_seconds"] == .5, label + "/fixed_caps")
    return actual


def inspect_core(row, run, protocol, criteria, truth, raw, checks, np):
    label = "run_" + str(row["run_index"])
    core_path = artifact(run, row["core_artifact"])
    checks.check(sha(core_path) == row["core_sha256"], label+"/core_hash")
    core = load_json(core_path)
    trajectory = core["trajectory"]
    coordinates = row["optimizer_coordinates"]
    for key in ["run_index", "case_index", "repetition", "method", "optimizer_coordinates", "identity"]:
        checks.check(core["schedule"][key] == row[key], label+"/core_schedule/"+key)
    checks.check(core["optimizer_coordinates"] == coordinates, label+"/coordinate_tag")
    model_path = artifact(run, row["models_artifact"])
    checks.check(row["models_sha256"] == core["models_sha256"] == sha(model_path), label+"/models_hash")
    checks.check(core["models_artifact"] == row["models_artifact"], label+"/models_name")
    names = ["initial_requested_m", "initial_common_m", "initial_velocity", "initial_parameter", "final_parameter", "final_native_m"]
    with np.load(model_path, allow_pickle=False) as archive:
        checks.check(set(archive.files) == set(names), label+"/model_slots")
        models = {key: archive[key].copy() for key in names}
    for key, model in models.items():
        if not checks.check(model.shape == (2401,) and model.dtype.kind == "f" and model.dtype.itemsize == 8
                            and np.isfinite(model).all() and (model > 0).all(), label+"/finite_positive/"+key):
            raise ValueError("invalid saved model vector")
    provenance = core["provenance"]
    checks.check(provenance == row["output_provenance"], label+"/row_provenance")
    checks.check(provenance["optimizer_coordinates"] == coordinates, label+"/provenance_coordinate_tag")
    checks.check(core["timing"] == row["timing"], label+"/row_timing")
    if row["status"] == "complete":
        checks.check({name:record["sha256"] for name,record in row["input_hashes_after"].items()} == protocol["inputs_sha256"], label+"/post_return_input_pins")
    binding = provenance["initial_common_binding"]
    checks.check(binding == row["common_binding"], label+"/common_binding")
    checks.check(binding["actual_common_m_sha256"] == array_sha(models["initial_common_m"], np), label+"/common_m_hash")
    checks.check(binding["initial_velocity_sha256"] == array_sha(models["initial_velocity"], np), label+"/initial_v_hash")
    checks.check(binding["requested_m_sha256"] == array_sha(models["initial_requested_m"], np), label+"/requested_m_hash")
    checks.check(binding["raw_model_sha256"] == array_sha(raw, np), label+"/raw_coarse_hash")
    checks.check(np.array_equal(np.power(models["initial_velocity"], -2.), models["initial_common_m"]), label+"/common_actual_mapping")
    checks.check(np.array_equal(1./np.sqrt(models["initial_requested_m"]), models["initial_velocity"]), label+"/requested_to_v_mapping")
    common_record = provenance["common_start_record"]
    checks.check(common_record["requested_native_model_sha256"] == binding["requested_m_sha256"]
                 and common_record["actual_native_model_sha256"] == binding["actual_common_m_sha256"]
                 and common_record["velocity_model_sha256"] == binding["initial_velocity_sha256"], label+"/roundtrip_hashes")
    checks.check(common_record["native_roundtrip_bitwise_equal"] == bool(np.array_equal(models["initial_requested_m"], models["initial_common_m"])), label+"/roundtrip_flag")
    checks.near(common_record["native_roundtrip_max_abs_difference"],
                max(abs(float(a)-float(b)) for a,b in zip(models["initial_requested_m"], models["initial_common_m"], strict=True)),
                label+"/roundtrip_difference", 0., 0.)
    is_v = coordinates == "physical_velocity"
    initial_expected = models["initial_velocity"] if is_v else models["initial_common_m"]
    final_expected = np.power(models["final_parameter"], -2.) if is_v else models["final_parameter"]
    checks.check(np.array_equal(initial_expected, models["initial_parameter"]), label+"/initial_coordinate_vector")
    checks.check(np.array_equal(final_expected, models["final_native_m"]), label+"/final_native_mapping")
    checks.check(provenance["final_parameter_sha256"] == array_sha(models["final_parameter"], np)
                 and provenance["final_actual_native_m_sha256"] == array_sha(models["final_native_m"], np), label+"/final_hashes")
    endpoint = provenance["endpoint_conversion"]
    checks.check(endpoint["actual_native_model_sha256"] == array_sha(models["final_native_m"], np), label+"/endpoint_native_hash")
    lo,hi = protocol["squared_slowness_bounds"]
    vlo = np.full(2401, 1. / np.sqrt(hi), dtype="<f8")
    vhi = np.full(2401, 1. / np.sqrt(lo), dtype="<f8")
    checks.check(provenance["bound_hashes"] == dict(velocity_lower=array_sha(vlo,np),velocity_upper=array_sha(vhi,np)), label+"/actual_velocity_bound_hashes")
    lower,upper = (vlo,vhi) if is_v else (lo,hi)
    if not checks.check(np.all(models["initial_parameter"] >= lower) and np.all(models["initial_parameter"] <= upper)
                         and np.all(models["final_parameter"] >= lower) and np.all(models["final_parameter"] <= upper), label+"/declared_coordinate_bounds"):
        raise ValueError("saved parameter violates its declared coordinate bounds")
    if is_v:
        contract = provenance["policy_contract"]
        checks.check(endpoint["parameterization"] == "physical_velocity"
                     and endpoint["velocity_model_sha256"] == array_sha(models["final_parameter"],np)
                     and endpoint["problem_fingerprint_sha256"] == contract["problem_fingerprint_sha256"], label+"/velocity_endpoint_fingerprint")
        checks.check(contract["parameterization"]["name"] == "physical_velocity"
                     and contract["parameterization"]["velocity_units"] == "km/s"
                     and contract["parameterization"]["adapter_source_sha256"] == criteria["velocity_oracle_sha256"], label+"/velocity_contract_units_source")
    else:
        checks.check(endpoint["parameterization"] == "squared_slowness", label+"/native_endpoint_tag")
    checks.check(trajectory["final_model"]["sha256"] == array_sha(models["final_parameter"], np), label+"/trajectory_final_parameter_hash")
    first = trajectory.get("checkpoints", [])[:1]
    initial_validation = provenance["paid_initial_binding_validation"]
    checks.check(initial_validation["paid_checkpoint_available"] == bool(first), label+"/paid_initial_availability")
    if first:
        checks.check(initial_validation["passes"] is True and first[0]["model_sha256"] == array_sha(initial_expected, np)
                     and first[0]["model"]["sha256"] == array_sha(initial_expected, np)
                     and initial_validation["actual_native_initial_sha256"] == binding["actual_common_m_sha256"], label+"/paid_initial_binding")
    wall = check_timing(row["timing"], checks, label, criteria)
    expected_runtime, reason = runtime_gate(trajectory, wall, criteria["api_return_cap_seconds"])
    gate = row["timing"]["runtime_gate"]
    checks.check(gate["runtime_eligible"] is expected_runtime and gate["reason"] == reason, label+"/runtime_gate")
    checks.check(gate["observed_event_seconds"] == (wall if expected_runtime else None), label+"/runtime_event")
    checks.check(core["engine_costs"] == trajectory["total_online_costs"], label+"/engine_counter_binding")
    engine_counts, api_counts = counts(core["engine_costs"]), counts(core["costs_at_api_return"])
    checks.check(all(api_counts.get(key, -1) >= value for key,value in engine_counts.items()), label+"/cost_monotonicity")
    for branch in ["physics", "oracle"]:
        checks.check(core["engine_costs"]["oracle"][branch] == core["costs_at_api_return"]["oracle"][branch], label+"/no_output_PDE/"+branch)
    if is_v:
        before = core["engine_costs"]["oracle"].get("velocity_coordinates", {}).get("endpoint_conversions", 0)
        after = core["costs_at_api_return"]["oracle"]["velocity_coordinates"].get("endpoint_conversions", 0)
        checks.check(after-before == 1, label+"/one_counted_output_conversion")
    else:
        checks.check(core["engine_costs"] == core["costs_at_api_return"], label+"/native_policy_counters_unchanged_in_output")
    prep = counts(core["preparation_physics_costs"])
    checks.check(prep.get("/factorizations") == 4 and prep.get("/forward_rhs") == 24, label+"/paid_preparation_physics")
    scalar = {key:scalar_errors(models[name], truth) for key,name in [("common_initial", "initial_common_m"), ("returned_final", "final_native_m")]}
    audit_ok = False
    if row.get("audit_artifact"):
        audit_path = artifact(run, row["audit_artifact"])
        checks.check(sha(audit_path) == row["audit_sha256"], label+"/embedded_audit_hash")
        audit = load_json(audit_path)
        audit_ok = not audit["failures"]
        checks.check(audit_ok, label+"/embedded_audit_no_failures")
        checks.check(finite(audit["maximum_relative_linear_residual"])
                     and audit["maximum_relative_linear_residual"] <= criteria["numeric_agreement"]["linear_residual_maximum"], label+"/embedded_residual")
        for slot in scalar:
            for metric,value in scalar[slot].items():
                for source in [audit["own_native_model_errors"][slot], audit["independent"][slot]]:
                    checks.near(value, source[metric], label+"/scalar/"+slot+"/"+metric,
                                criteria["numeric_agreement"]["quality_rtol"], criteria["numeric_agreement"]["quality_atol"])
    eligible = row["status"] == "complete" and row.get("final_study_eligible") is True and expected_runtime and audit_ok
    if row["status"] == "complete":
        checks.check(row.get("final_study_eligible") is True and audit_ok, label+"/terminal_integrity")
        checks.check((row.get("primary_quality") is not None) == eligible, label+"/primary_null_gate")
    else:
        checks.check(row.get("primary_quality") is None and row.get("final_study_eligible") is False, label+"/failed_primary_null")
    if row.get("primary_quality") is not None:
        for metric,value in scalar["returned_final"].items():
            checks.near(value, row["primary_quality"][metric], label+"/primary/"+metric,
                        criteria["numeric_agreement"]["quality_rtol"], criteria["numeric_agreement"]["quality_atol"])
    targets = []
    phi0,phi = trajectory.get("initial_objective"),trajectory.get("final_objective")
    for ratio in criteria["returned_ratios"]:
        reached = finite(phi0) and phi0 > 0 and finite(phi) and phi <= ratio*phi0
        event = wall if eligible and reached else None
        recorded = next((r for r in row.get("returned_targets", []) if r["ratio"] == ratio), None)
        if row["status"] == "complete":
            checks.check(recorded is not None and recorded["reached"] is bool(reached)
                         and recorded["event_seconds"] == event, label+"/returned_ratio/"+str(ratio))
        elif recorded is not None:
            checks.check(recorded["event_seconds"] is None, label+"/failed_ratio_null/"+str(ratio))
        targets.append(dict(ratio=ratio, reached_at_return=bool(reached), verified_event_seconds=event))
    native_bound_counts = dict(below=sum(float(x)<lo for x in models["final_native_m"]), above=sum(float(x)>hi for x in models["final_native_m"]))
    return dict(candidate_joint_eligible=bool(eligible), runtime_eligible=expected_runtime, runtime_reason=reason,
        engine_status=trajectory["status"], actual_api_return_seconds=wall, scalar_errors=scalar, returned_targets=targets,
        common_binding=binding, final_parameter_sha256=array_sha(models["final_parameter"], np),
        final_actual_native_m_sha256=array_sha(models["final_native_m"], np), literal_final_native_bound_counts=native_bound_counts,
        bound_scope="literal native counts retained; VP float64 inverse boxes are not outward-rounded identical native boxes",
        source_truth_scope="own scalar half-cell bilinear truth; roundoff need not share producer truth bytes",
        intermediate_raw_replay_available=False)


def summaries(rows, criteria, globally_clear):
    groups=[]
    for case in range(8):
        for method in criteria["methods"]:
            selected=[r for r in rows if r["case_index"]==case and r["method"]==method]
            available=[r for r in selected if r.get("verified_credit_eligible")]
            all3=globally_clear and len(selected)==len(available)==3
            median=None
            if all3:
                median={key:median3([r["scalar_errors"]["returned_final"][key] for r in available])
                        for key in ["relative_squared_slowness_L2","relative_physical_velocity_L2"]}
            groups.append(dict(case_index=case,slice_id=criteria["slice_ids"][case],method=method,
                planned_repetitions=3,verified_available=len(available),all3_required=True,median_returned_quality=median))
    pairs=[]
    for repetition in range(3):
        for case in range(8):
            selected=[r for r in rows if r["case_index"]==case and r["repetition"]==repetition]
            pairs.append(dict(case_index=case,repetition=repetition,planned_methods=2,
                verified_available=sum(bool(r.get("verified_credit_eligible")) for r in selected),
                both_available=globally_clear and len(selected)==2 and all(r.get("verified_credit_eligible") for r in selected)))
    return dict(case_method_groups=groups,paired_rows=pairs,case_denominator=8,paired_denominator=24,
        row_denominator=48,case_method_denominator=16,scope="complete denominators and all3 medians; no winner/p-values/first-hit/independent-geology inference")


def execute(args, criteria):
    import numpy as np

    run,repo,out=args.run.resolve(),args.repo.resolve(),args.output.resolve()
    if out.exists() or out.is_relative_to(run) or out.is_relative_to(repo) or out.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("fresh OUTER audit output required")
    out.mkdir()
    shutil.copyfile(Path(__file__),out/"audit_source.py")
    shutil.copyfile(args.criteria,out/"criteria.json")
    checks=Checks()
    plan=expected_schedule(criteria)
    report=dict(status="running",result="PENDING",planned_rows=48,rows=[{**r,"audit_status":"unattempted","verified_credit_eligible":False} for r in plan],
        issues=[],created_utc=datetime.now(timezone.utc).isoformat(),new_PDE_solves=0,physical_quality_pass=None,
        role=criteria["role"],source_sha256=sha(Path(__file__)),criteria_sha256=sha(args.criteria),
        terminal_observation_sha256=sha(args.terminal_observation),terminal_scope="Operator/root confirmation plus completed producer status; not independent process/liveness measurement.")
    atomic_report(out/"report.json",report)
    before=None
    input_before={}
    run_before={}
    try:
        before=source_manifest(repo)
        checks.check(before["digest"]==criteria["production_source_sha256"],"production_source_before")
        pins={"driver_source.py":criteria["producer_sha256"],"protocol.json":criteria["protocol_sha256"],
            "velocity_oracle.py":criteria["velocity_oracle_sha256"],"independent_saved_models.py":criteria["independent_saved_models_sha256"],
            "original_protocol.json":criteria["original_protocol_sha256"]}
        for name,h in pins.items():checks.check(sha(artifact(run,name))==h,"run_source_pin/"+name)
        protocol=load_json(run/"protocol.json")
        status=load_json(run/"status.json")
        complete_path=run/"complete_receipt.json"
        complete=load_json(complete_path) if complete_path.exists() else None
        checks.check(status["status"]=="complete_engineering_comparison" and complete is not None,"complete_benchmark_required_for_credit")
        if complete is not None:
            checks.check({k:v for k,v in complete.items() if k!="artifact_sha256"}==status,"complete_status_binding")
            for name,h in complete["artifact_sha256"].items():checks.check(sha(artifact(run,name))==h,"complete_manifest/"+name)
            checks.check({p.name for p in run.iterdir() if p.is_file()}==set(complete["artifact_sha256"])|{"complete_receipt.json"},"complete_manifest_exact_file_set")
        run_before={p.name:sha(p) for p in run.iterdir() if p.is_file()}
        schedule=load_json(run/"schedule.json")
        checks.check(len(schedule)==len(status["runs"])==48,"full48_schedule")
        for expected,scheduled,row in zip(plan,schedule,status["runs"],strict=True):
            report["rows"][expected["run_index"]].update(
                recorded_status=row["status"], recorded_error=row.get("error"),
                recorded_terminal_integrity_flag=row.get("final_study_eligible"),
                recorded_primary_quality_is_null=row.get("primary_quality") is None,
                recorded_phase_at_failure=row.get("phase_at_failure"),
                recorded_elapsed_until_failure_or_interrupt_seconds=row.get("elapsed_until_failure_or_interrupt_seconds"))
            for key in ["run_index","case_index","repetition","order_position","method","optimizer_coordinates"]:
                checks.check(expected[key]==scheduled[key]==row[key],"schedule/"+str(expected["run_index"])+"/"+key)
            checks.check(expected["slice_id"]==scheduled["identity"]["slice_id"]==row["identity"]["slice_id"],"slice_identity/"+str(expected["run_index"]))
            checks.check(scheduled["status"]=="unattempted","planned_schedule_unattempted/"+str(expected["run_index"]))
            for name_key, hash_key in [("core_artifact","core_sha256"),("models_artifact","models_sha256"),
                                       ("audit_artifact","audit_sha256"),("failure_artifact","failure_sha256")]:
                if row.get(name_key):
                    checks.check(sha(artifact(run,row[name_key]))==row[hash_key],"row_artifact/"+str(row["run_index"])+"/"+name_key)
        report["recorded_status_counts"] = recorded_status_counts(report["rows"])
        if status["status"] == "complete_engineering_comparison":
            checks.check(status.get("runs_attempted")==48 and all(row["status"] in {"complete","failed"} for row in status["runs"]),"all48_actually_attempted")
        checks.check(status["source_before"]==status.get("source_after"),"producer_source_before_after")
        checks.check(status["source_before"]["driver_sha256"]==criteria["producer_sha256"]
                     and status["source_before"]["protocol_sha256"]==criteria["protocol_sha256"]
                     and status["source_before"]["production"]["repository_source_sha256"]==criteria["production_source_sha256"],"producer_recorded_source_pins")
        for name,h in protocol["inputs_sha256"].items():
            path=(repo/name).resolve()
            if not path.is_relative_to(repo):raise ValueError("input escaped repository")
            input_before[name]=sha(path)
            checks.check(input_before[name]==h,"pinned_input/"+name)
        if checks.issues:
            raise ValueError("source/schedule/artifact prerequisites failed; no array evaluation")
        input_prefix="data/reference/subsurfacegen_parent_split/"
        metadata=list(csv.DictReader((repo/(input_prefix+"validation_metadata.csv")).open()))
        with np.load(repo/(input_prefix+"validation.npz"),allow_pickle=False) as archive:
            raw_models=archive["models"].copy()
        checks.check(raw_models.shape==(192,24,24) and raw_models.dtype==np.float32,"raw_model_schema")
        raw_by_case={}
        truth_by_case={}
        for case,identity in enumerate(protocol["identities"]):
            index=identity["array_index"]
            selected=[i for i,row in enumerate(metadata) if row["slice_id"]==identity["slice_id"]]
            checks.check(len(selected)==24 and selected[0]==index and all(np.array_equal(raw_models[i],raw_models[index]) for i in selected),"raw_case_binding/"+str(case))
            raw_by_case[case]=raw_models[index].astype(np.float64)
            truth_by_case[case]=reconstruct_truth(raw_by_case[case],np)
        report["independent_truth_sha256"]={str(k):array_sha(v,np) for k,v in truth_by_case.items()}
        common={}
        for expected,row,result in zip(plan,status["runs"],report["rows"],strict=True):
            result.update(recorded_status=row["status"],recorded_terminal_integrity_flag=row.get("final_study_eligible"),
                          verified_credit_eligible=False,candidate_joint_eligible=False)
            try:
                if row.get("core_artifact"):
                    detail=inspect_core(row,run,protocol,criteria,truth_by_case[row["case_index"]],raw_by_case[row["case_index"]],checks,np)
                    result.update(detail,audit_status="saved_core_checked")
                    binding=detail["common_binding"]
                    checks.check(binding==common.setdefault(row["case_index"],binding),"all_methods_reps_common_binding/"+str(row["run_index"]))
                else:
                    checks.check(row["status"]!="complete" and row.get("primary_quality") is None and row.get("final_study_eligible") is not True,"no_core_no_credit/"+str(row["run_index"]))
                    result.update(audit_status="no_returned_core",actual_api_return_seconds=None,scalar_errors=None,returned_targets=[dict(ratio=r,verified_event_seconds=None) for r in criteria["returned_ratios"]])
                if row.get("failure_artifact"):
                    path=artifact(run,row["failure_artifact"])
                    checks.check(sha(path)==row["failure_sha256"],"failure_hash/"+str(row["run_index"]))
                    failure=load_json(path)
                    checks.check(failure["final_study_eligible"] is False,"failure_ineligible/"+str(row["run_index"]))
                    result["failure_counter_scope"]={key:failure.get(key) for key in ["in_flight_engine_counters_unknown","latest_policy_counter_snapshot_unknown","observed_counters_complete_for_successful_api","counter_scope"]}
                    for key in ["latest_observed_policy_cost_snapshot","earlier_engine_cost_snapshot"]:
                        if failure.get(key) is not None:counts(failure[key])
                    if failure.get("failure_models_artifact"):
                        checks.check(sha(artifact(run,failure["failure_models_artifact"]))==failure["failure_models_sha256"],"failure_models_hash/"+str(row["run_index"]))
                result["recorded_elapsed_until_failure_or_interrupt_seconds"]=row.get("elapsed_until_failure_or_interrupt_seconds")
            except Exception as exc:
                checks.check(False,"row_exception/"+str(row["run_index"]),str(exc))
                result.update(audit_status="audit_exception",error=f"{type(exc).__name__}: {exc}",candidate_joint_eligible=False)
            report.update(checks=checks.count,numeric_comparisons=checks.numeric_count,issues=checks.issues)
            atomic_report(out/"report.json",report)
        checks.check(source_manifest(repo)==before,"production_source_after")
        checks.check({name:sha(repo/name) for name in input_before}==input_before,"inputs_after")
        checks.check({p.name:sha(p) for p in run.iterdir() if p.is_file()}==run_before,"all_run_artifacts_after")
        clear=not checks.issues
        for row in report["rows"]:
            row["verified_credit_eligible"]=bool(clear and row.get("candidate_joint_eligible"))
            if not row["verified_credit_eligible"]:
                for target in row.get("returned_targets",[]):target["verified_event_seconds"]=None
        report.update(status="complete",result="CLEAR_COORDINATE_ENGINEERING_POSTAUDIT" if clear else "FAILED_COORDINATE_ENGINEERING_POSTAUDIT",
            checks=checks.count,numeric_comparisons=checks.numeric_count,issues=checks.issues,all48_rows_retained=True,
            summary=summaries(report["rows"],criteria,clear),production_source_before=before["digest"],production_source_after=source_manifest(repo)["digest"],
            recorded_status_counts=recorded_status_counts(report["rows"]),
            limitations=criteria["limits"],finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        checks.check(False,"audit_execution_exception",f"{type(exc).__name__}: {exc}")
        for row in report["rows"]:
            row["verified_credit_eligible"]=False
            for target in row.get("returned_targets",[]):target["verified_event_seconds"]=None
        report.update(status="failed_or_interrupted",result="FAILED_COORDINATE_ENGINEERING_POSTAUDIT",checks=checks.count,
            numeric_comparisons=checks.numeric_count,issues=checks.issues,summary=summaries(report["rows"],criteria,False),all48_rows_retained=True,
            recorded_status_counts=recorded_status_counts(report["rows"]),
            error=f"{type(exc).__name__}: {exc}",finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_report(out/"report.json",report)
        raise
    atomic_report(out/"report.json",report)
    print(json.dumps(dict(result=report["result"],checks=checks.count,numeric_comparisons=checks.numeric_count,
                         issues=len(checks.issues),rows=len(report["rows"]),report=str(out/"report.json"))))
    return 0 if not checks.issues else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criteria",type=Path,default=Path(__file__).with_name("criteria.json"))
    parser.add_argument("--execute-after-terminal-confirmation",action="store_true")
    parser.add_argument("--run",type=Path)
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--terminal-observation",type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-criteria-sha256")
    args=parser.parse_args()
    criteria=load_json(args.criteria)
    if not args.execute_after_terminal_confirmation:
        print(json.dumps(dict(status="metadata_only_no_project_reads",planned_rows=48,source_sha256=sha(Path(__file__)),criteria_sha256=sha(args.criteria))))
        return 0
    if not all([args.run,args.repo,args.output,args.terminal_observation,args.expected_source_sha256,args.expected_criteria_sha256]):
        parser.error("explicit run/repo/new output/terminal observation and both source hashes required")
    if sha(Path(__file__))!=args.expected_source_sha256 or sha(args.criteria)!=args.expected_criteria_sha256:
        raise ValueError("audit source/criteria freeze mismatch")
    if not args.terminal_observation.is_file():raise ValueError("root terminal-observation artifact required")
    return execute(args,criteria)


if __name__=="__main__":
    raise SystemExit(main())
