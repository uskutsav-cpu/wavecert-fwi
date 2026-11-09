"""Prospective48-run software comparison; default CLI loads no project arrays."""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace


class StaleFreeze(RuntimeError):
    pass


class AnchoredClock:
    def __init__(self, start, clock=perf_counter):
        self.start, self.clock, self.calls, self.last = float(start), clock, 0, float(start)

    def __call__(self):
        value = self.start if self.calls == 0 else float(self.clock())
        self.calls += 1
        if not math.isfinite(value) or value < self.last:
            raise ValueError("clock must be finite and monotone")
        self.last = value
        return value


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def exclusive(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    try:
        exclusive(temporary, json_bytes(value))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def new_gzip(path, value):
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(json_bytes(value))


def load_protocol(path):
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if (protocol["schema_version"] != 1 or protocol["methods"] != ["native_m_lbfgsb", "physical_v_lbfgsb"]
            or protocol["schedule"] != dict(cases=8, repetitions=3, total_runs=48, seed=2026110801,
                shuffle="stdlib random.Random(seed).shuffle(two methods) per case inside repetition; paired adjacent methods; fixed case order")
            or protocol["trajectory"]["budget_seconds"] != 7.5 or protocol["timing"]["api_return_cap_seconds"] != 8
            or protocol["student_final_research_design"] is not False or protocol["competition_evidence"] is not False
            or protocol["project_execution_authorized"] is not False):
        raise ValueError("fixed prospective engineering scope changed")
    for filename, expected in protocol["local_dependency_sha256"].items():
        if sha((path.parent / filename).read_bytes()) != expected:
            raise StaleFreeze("local frozen dependency changed: " + filename)
    original_raw = (path.parent / "original_protocol.json").read_bytes()
    original = json.loads(original_raw)
    for key in ["identities", "inputs_sha256", "physics", "geometry", "squared_slowness_bounds", "lbfgsb", "threads"]:
        if protocol[key] != original[key]:
            raise StaleFreeze("original fixed setting changed: " + key)
    if protocol["trajectory"] != {**original["trajectory"], "budget_seconds": 7.5}:
        raise StaleFreeze("only the frozen engine budget changes")
    return protocol, raw, original, original_raw


def make_schedule(protocol):
    rng, rows = random.Random(protocol["schedule"]["seed"]), []
    for repetition in range(3):
        for case in range(8):
            order = list(protocol["methods"])
            rng.shuffle(order)
            for position, method in enumerate(order):
                rows.append(dict(run_index=len(rows), repetition=repetition, case_index=case, order_position=position,
                    method=method, identity=dict(protocol["identities"][case]), status="unattempted",
                    optimizer_coordinates="squared_slowness" if method == "native_m_lbfgsb" else "physical_velocity"))
    return rows


def import_file(name, path):
    if name in sys.modules:
        raise StaleFreeze("foreign/already loaded explicit module: " + name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("explicit module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise StaleFreeze("explicit module origin mismatch")
    return module


def load_runtime(root, protocol, bundle):
    directory = root / protocol["original_candidate_relative_directory"]
    for name, expected in protocol["candidate_files_sha256"].items():
        if sha((directory / name).read_bytes()) != expected:
            raise StaleFreeze("original candidate changed: " + name)
    sys.path.insert(0, str(root / "src"))
    name = "_coordinate_cost_original_candidate"
    if name in sys.modules:
        raise StaleFreeze("candidate module name already occupied")
    spec = importlib.util.spec_from_file_location(name, directory / "__init__.py", submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    helpers = importlib.import_module(name + ".verification")
    if Path(helpers.__file__).resolve() != (directory / "verification.py").resolve():
        raise StaleFreeze("foreign candidate helper")
    helpers._read_protocol(directory / "protocol.json")
    engine = importlib.import_module("wavecert.optimization.paid_descent")
    physics = importlib.import_module("wavecert.physics.helmholtz")
    native = importlib.import_module("wavecert.physics.exact_merit")
    vp = import_file("_coordinate_cost_frozen_velocity", bundle / "velocity_oracle.py")
    audit = import_file("_coordinate_cost_independent_saved_models", bundle / "independent_saved_models.py")
    return SimpleNamespace(helpers=helpers, engine=engine, physics=physics, native=native, vp=vp, audit=audit,
                           np=importlib.import_module("numpy"))


def static_receipt(root, protocol_path, protocol_raw, original_raw, protocol, runtime):
    if protocol_path.read_bytes() != protocol_raw:
        raise StaleFreeze("coordinate protocol changed")
    source = runtime.helpers.source_receipt(root, original_raw)
    if source["repository_source_sha256"] != protocol["production_source_sha256"]:
        raise StaleFreeze("production source changed")
    paths = {name: sha((protocol_path.parent / name).read_bytes()) for name in protocol["local_dependency_sha256"]}
    if paths != protocol["local_dependency_sha256"]:
        raise StaleFreeze("local dependency changed")
    directory = root / protocol["original_candidate_relative_directory"]
    candidate = {name: sha((directory / name).read_bytes()) for name in protocol["candidate_files_sha256"]}
    if candidate != protocol["candidate_files_sha256"]:
        raise StaleFreeze("candidate source changed")
    return dict(production=source, local_dependencies=paths, candidate=candidate,
                driver_sha256=sha(Path(__file__).read_bytes()), protocol_sha256=sha(protocol_raw))


def hash_inputs(root, protocol):
    values = {}
    for name, expected in protocol["inputs_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise StaleFreeze("input path escapes repository")
        raw = path.read_bytes()
        if sha(raw) != expected:
            raise StaleFreeze("original development input changed: " + name)
        values[name] = dict(sha256=expected, bytes=len(raw))
    return values


def progress(output, row, context, clock=perf_counter):
    start = clock()
    keys = ["run_index", "case_index", "repetition", "order_position", "method", "optimizer_coordinates", "status", "phase",
            "started_utc", "global_started_perf_counter", "common_start_record", "common_binding", "error", "finished_utc"]
    value = {key: row[key] for key in keys if key in row}
    value.update(scope="bounded current-run metadata only; no growing cohort or trajectory arrays",
                 preparation_physics_costs=context.get("preparation_physics_costs"))
    atomic_json(output / row["progress_artifact"], value)
    name = "timed_progress_write_seconds" if context.get("inside_timed_interval") else "outside_progress_write_seconds"
    context[name] = context.get(name, 0.) + clock() - start


def validate_paid_initial(result, initial_parameter, binding, policy, is_velocity, runtime):
    if not result.get("checkpoints"):
        return dict(paid_checkpoint_available=False, passes=False, reason="no paid initial checkpoint")
    first = result["checkpoints"][0]
    parameter_sha = runtime.vp.model_sha256(first["model"])
    mapped = runtime.vp.velocity_to_slowness(first["model"]) if is_velocity else first["model"]
    native_sha = runtime.vp.model_sha256(mapped)
    native_receipt_record = policy.initial_model_record if is_velocity else None
    expected_parameter_sha = runtime.vp.model_sha256(initial_parameter)
    passes = bool(parameter_sha == first.get("model_sha256") == expected_parameter_sha
        and runtime.np.array_equal(first["model"], initial_parameter)
        and native_sha == binding["actual_common_m_sha256"]
        and (not is_velocity or native_receipt_record is not None
             and native_receipt_record["actual_native_model_sha256"] == native_sha
             and native_receipt_record["velocity_model_sha256"] == parameter_sha))
    return dict(paid_checkpoint_available=True, passes=passes, first_parameter_sha256=parameter_sha,
                expected_parameter_sha256=expected_parameter_sha, actual_native_initial_sha256=native_sha,
                expected_native_common_sha256=binding["actual_common_m_sha256"],
                native_initial_receipt_record=native_receipt_record)


def execute_timed(root, protocol, original, row, runtime, context, phase, check_static, *, expected_common=None, clock=perf_counter):
    """The caller timestamps the actual return after this complete payload exists."""
    start = float(clock())
    context.update(global_started=start, inside_timed_interval=True, preparation_physics_costs=None)
    phase("source_checks", dict(global_started_perf_counter=start))
    check_static()
    phase("input_loading_hash_verification", {})
    begin = clock()
    try:
        cases = runtime.helpers.load_raw_inputs(root, original)
    except ValueError as exc:
        raise StaleFreeze("input/schema freeze failed") from exc
    context["input_loading_seconds"] = clock() - begin
    raw_case = cases[row["case_index"]]
    context["raw_case"] = raw_case
    phase("refinement_initialization_observations", {})
    begin = clock()
    prepared = runtime.helpers.prepare_case(raw_case, original)
    context.update(prepared=prepared, preparation_physics_costs=prepared["preparation_costs"],
                   preparation_seconds=clock() - begin)
    del cases
    phase("common_coordinate_start", {})
    begin = clock()
    v0, common_m, conversion = runtime.vp.start_from_slowness(prepared["initial"])
    if not runtime.np.array_equal(runtime.vp.velocity_to_slowness(v0), common_m):
        raise ValueError("VP common initial state is not bitwise canonical")
    lo, hi = original["squared_slowness_bounds"]
    vlo, vhi = runtime.vp.slowness_bounds_to_velocity(lo, hi, size=len(common_m))
    observation_sha = hashlib.sha256()
    observation_sha.update(json_bytes(list(prepared["receivers"])))
    for block in prepared["blocks"]:
        observation_sha.update(json_bytes([int(block.source_index), float(block.frequency_hz)]))
        observation_sha.update(runtime.np.ascontiguousarray(block.observed, dtype="<c16").tobytes())
    binding = dict(actual_common_m_sha256=runtime.vp.model_sha256(common_m),
        initial_velocity_sha256=runtime.vp.model_sha256(v0), requested_m_sha256=runtime.vp.model_sha256(prepared["initial"]),
        observations_sha256=observation_sha.hexdigest(), truth_sha256=prepared["truth_sha256"], raw_model_sha256=prepared["raw_model_sha256"])
    if expected_common is not None and binding != expected_common:
        raise StaleFreeze("paired case common actual model/data differ bitwise")
    context.update(initial_velocity=v0, common_initial_m=common_m, common_start_record=conversion,
                   common_binding=binding, coordinate_preparation_seconds=clock() - begin)
    phase("policy_call", dict(common_start_record=conversion, common_binding=binding))
    is_velocity = row["method"] == "physical_v_lbfgsb"
    initial_parameter = v0 if is_velocity else common_m
    context["initial_parameter"] = initial_parameter

    def oracle_factory():
        physics = runtime.physics.Helmholtz2D(**{**original["physics"], "shape": tuple(original["physics"]["shape"])})
        native = runtime.native.ExactMeritOracle(physics, prepared["blocks"], prepared["receivers"])
        policy = runtime.vp.VelocityMeritOracle(native, velocity_units="km/s") if is_velocity else native
        context.update(policy=policy, policy_contract=policy.contract)
        return policy

    anchored = AnchoredClock(start, clock)
    context["pre_engine_seconds"] = clock() - start
    result = runtime.engine.run_exact_lbfgsb(oracle_factory=oracle_factory, m0=initial_parameter,
        lower_bound=vlo if is_velocity else lo, upper_bound=vhi if is_velocity else hi,
        options=runtime.engine.TrajectoryOptions(**protocol["trajectory"]), clock=anchored, **protocol["lbfgsb"])
    context.update(result=result, engine_returned=clock(), anchored_clock_calls=anchored.calls)
    phase("final_native_conversion_hash_provenance", {})
    conversion_start = clock()
    initial_validation = validate_paid_initial(result, initial_parameter, binding, context["policy"], is_velocity, runtime)
    context["paid_initial_binding_validation"] = initial_validation
    if initial_validation["paid_checkpoint_available"] and not initial_validation["passes"]:
        raise ValueError("first paid parameter/native model differs from promised common start")
    parameter = runtime.np.ascontiguousarray(result["final_model"], dtype="<f8").copy()
    if is_velocity:
        final_native, endpoint = context["policy"].to_native_model(parameter)
    else:
        final_native = parameter.copy()
        endpoint = dict(actual_native_model_sha256=runtime.vp.model_sha256(final_native), parameterization="squared_slowness")
    if (not runtime.np.isfinite(parameter).all() or not runtime.np.isfinite(final_native).all()
            or (final_native <= 0).any()):
        raise ValueError("returned parameter/native snapshot invalid")
    provenance = dict(optimizer_coordinates=row["optimizer_coordinates"],
        final_parameter_sha256=runtime.vp.model_sha256(parameter), final_actual_native_m_sha256=runtime.vp.model_sha256(final_native),
        endpoint_conversion=endpoint, initial_common_binding=binding, common_start_record=conversion,
        paid_initial_binding_validation=initial_validation,
        initial_parameter_sha256=runtime.vp.model_sha256(initial_parameter),
        bound_hashes=dict(velocity_lower=runtime.vp.model_sha256(vlo), velocity_upper=runtime.vp.model_sha256(vhi)),
        policy_contract=context["policy_contract"], native_units="m in(s/km)^2; v inkm/s; no unit rescaling",
        final_native_is_separate_snapshot=True)
    serialized_provenance = json_bytes(provenance)  # Complete output metadata construction/validation is paid.
    context.update(final_parameter=parameter, final_native_m=final_native, output_provenance=provenance,
        final_conversion_seconds=clock() - conversion_start,
        api_cost_snapshot=dict(oracle=context["policy"].cost_snapshot(), proposal={}),
        output_coordinate_work=dict(parameter_snapshot_entries=int(parameter.size), native_snapshot_entries=int(final_native.size),
                                    initial_binding_validation_entries=int(initial_parameter.size),
                                    provenance_utf8_bytes=len(serialized_provenance)))
    return dict(result=result, final_parameter=parameter, final_native_m=final_native,
                output_provenance=provenance, costs_at_api_return=context["api_cost_snapshot"])


def return_gate(result, wall, cap):
    paid = bool(result.get("checkpoints") and result.get("initial_objective") is not None
                and result.get("final_objective") is not None
                and math.isfinite(result["initial_objective"]) and math.isfinite(result["final_objective"]))
    reason = "fully_returned_paid_state"
    if not result.get("costs_complete") or result["status"] == "failed":
        reason = "failed_or_incomplete_method"
    elif not paid:
        reason = "no_paid_final_state"
    elif not math.isfinite(wall) or wall < 0 or wall > cap:
        reason = "late_or_invalid_actual_api_return"
    return dict(runtime_eligible=reason == "fully_returned_paid_state", reason=reason, actual_api_wall_seconds=wall,
                observed_event_seconds=wall if reason == "fully_returned_paid_state" else None,
                primary_quality_requires_terminal_study_eligibility=True)


def timing_record(context, returned, protocol):
    result, start = context["result"], context["global_started"]
    wall = returned - start
    engine = float(result["total_online_seconds"])
    engine_return_wall = context["engine_returned"] - start
    if not (0 <= context["pre_engine_seconds"] <= engine + 1e-8 <= engine_return_wall + 2e-8 <= wall + 3e-8):
        raise ValueError("global/engine/output timing nesting invalid")
    return dict(actual_api_return_seconds=wall, engine_call_return_seconds=engine_return_wall,
        engine_reported_seconds_including_preparation=engine,
        engine_finish_tail_seconds=max(0., engine_return_wall - engine), api_output_tail_seconds=max(0., wall - engine_return_wall),
        final_native_conversion_hash_seconds=context["final_conversion_seconds"], pre_engine_seconds=context["pre_engine_seconds"],
        input_loading_hash_seconds=context["input_loading_seconds"], preparation_seconds=context["preparation_seconds"],
        coordinate_preparation_seconds=context["coordinate_preparation_seconds"],
        timed_progress_write_seconds=context.get("timed_progress_write_seconds", 0.),
        preparation_and_progress_are_subsets_never_added_twice=True,
        engine_budget_seconds=7.5, fixed_api_cap_seconds=8., reserve_seconds=.5,
        actual_api_overshoot_seconds=max(0., wall - 8.), engine_budget_overshoot_seconds=result["budget_overshoot_seconds"],
        anchored_clock_calls=context["anchored_clock_calls"],
        runtime_gate=return_gate(result, wall, protocol["timing"]["api_return_cap_seconds"]))


def save_core(output, row, context, runtime):
    prefix = f"run_{row['run_index']:03d}"
    models = dict(initial_requested_m=context["prepared"]["initial"], initial_common_m=context["common_initial_m"],
                  initial_velocity=context["initial_velocity"], initial_parameter=context["initial_parameter"],
                  final_parameter=context["final_parameter"], final_native_m=context["final_native_m"])
    path = output / (prefix + "_models.npz")
    with path.open("xb") as stream:
        runtime.np.savez_compressed(stream, **models)
    core = dict(schedule={key: row[key] for key in ["run_index", "case_index", "repetition", "method", "optimizer_coordinates", "identity"]},
        trajectory=runtime.helpers.compact_trajectory(context["result"]), optimizer_coordinates=row["optimizer_coordinates"],
        actual_native_model_key="final_native_m", provenance=context["output_provenance"], timing=row["timing"],
        engine_costs=context["result"]["total_online_costs"], costs_at_api_return=context["api_cost_snapshot"],
        driver_output_coordinate_work=context["output_coordinate_work"], preparation_physics_costs=context["preparation_physics_costs"],
        models_artifact=path.name, models_sha256=sha(path.read_bytes()),
        scope="final/initial vectors retained; intermediate trajectory vectors compact hashes only; no internal quality credit")
    core_path = output / (prefix + "_core.json.gz")
    new_gzip(core_path, core)
    return dict(core_artifact=core_path.name, core_sha256=sha(core_path.read_bytes()),
                models_artifact=path.name, models_sha256=sha(path.read_bytes()))


def own_errors(model, truth, np):
    return dict(relative_squared_slowness_L2=float(np.linalg.norm(model - truth) / np.linalg.norm(truth)),
        relative_physical_velocity_L2=float(np.linalg.norm(1 / np.sqrt(model) - 1 / np.sqrt(truth)) / np.linalg.norm(1 / np.sqrt(truth))))


def offline_audit(row, context, original, runtime, protocol):
    """Fresh independently assembled audit after return, never a policy input."""
    checks = runtime.audit.Checks()
    case = runtime.audit.IndependentCase(context["raw_case"]["raw_model"], original, checks, row["case_index"])
    common = case.evaluate(context["common_initial_m"])
    final = case.evaluate(context["final_native_m"])
    own = {name: own_errors(model, context["prepared"]["truth"], runtime.np) for name, model in
           [("common_initial", context["common_initial_m"]), ("returned_final", context["final_native_m"]) ]}
    settings = protocol["offline_audit"]
    for label, independent, recorded in [("common_initial", common, context["result"]["initial_objective"]),
                                          ("returned_final", final, context["result"]["final_objective"])]:
        if recorded is not None:
            checks.near(independent["objective"], recorded, rtol=settings["objective_rtol"],
                atol=settings["objective_atol_initial_scale"] * max(common["objective"], settings["objective_atol_initial_floor"]),
                context=label + "/objective", family="objective")
        for key in ["relative_squared_slowness_L2", "relative_physical_velocity_L2"]:
            checks.near(independent[key], own[label][key], rtol=settings["quality_rtol"], atol=settings["quality_atol"],
                        context=label + "/" + key, family="quality")
    return dict(own_native_model_errors=own, independent=dict(common_initial=common, returned_final=final),
        checks=checks.count, numeric_comparisons=checks.numeric_count, failures=checks.failures,
        factorizations=case.factorizations, forward_rhs=case.forward_rhs,
        maximum_relative_linear_residual=case.maximum_relative_linear_residual,
        scope="fresh post-return scalar audit; ordinary FP, shared NumPy/SciPy/SuperLU; raw diagnostics are not primary credit")


def preserve_failure(output, row, context, runtime):
    value = dict(row=row, in_flight_engine_counters_unknown="result" not in context,
                 in_flight_trajectory_not_recoverable="result" not in context,
                 preparation_physics_costs=context.get("preparation_physics_costs"), final_study_eligible=False)
    value["earlier_engine_cost_snapshot"] = None if "result" not in context else context["result"].get("total_online_costs")
    value["paid_initial_binding_validation"] = context.get("paid_initial_binding_validation")
    value["latest_policy_counter_snapshot_unknown"] = True
    value["observed_counters_complete_for_successful_api"] = False
    value["counter_scope"] = "Best-effort observed cumulative counters; in-flight work may be incomplete; no successful API-return credit."
    if "policy" in context:
        try:
            value["latest_observed_policy_cost_snapshot"] = context["policy"].cost_snapshot()
            value["latest_policy_counter_snapshot_unknown"] = False
        except Exception as exc:
            value["counter_snapshot_error"] = f"{type(exc).__name__}: {exc}"
    if "result" in context:
        value["returned_trajectory"] = runtime.helpers.compact_trajectory(context["result"])
        if "models_artifact" not in row:
            arrays = {key: context[key] for key in ["initial_parameter", "initial_velocity", "common_initial_m", "final_parameter", "final_native_m"] if key in context}
            arrays["returned_parameter"] = context["result"]["final_model"]
            path = output / f"run_{row['run_index']:03d}_failure_models.npz"
            with path.open("xb") as stream:
                runtime.np.savez_compressed(stream, **arrays)
            value.update(failure_models_artifact=path.name, failure_models_sha256=sha(path.read_bytes()))
    path = output / f"run_{row['run_index']:03d}_failure.json.gz"
    new_gzip(path, value)
    row.update(failure_artifact=path.name, failure_sha256=sha(path.read_bytes()))


def run(root, output, protocol_path, expected_protocol, expected_driver):
    bootstrap = perf_counter()
    root, output, protocol_path = root.resolve(), output.resolve(), protocol_path.resolve()
    protocol, raw, original, original_raw = load_protocol(protocol_path)
    if sha(raw) != expected_protocol or sha(Path(__file__).read_bytes()) != expected_driver:
        raise StaleFreeze("explicit driver/protocol hash gate failed")
    if output.exists() or output.is_relative_to(root) or output.is_relative_to(protocol_path.parent):
        raise FileExistsError("new output outside repository and frozen bundle required")
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < protocol["disk"]["initial_free_bytes"]:
        raise OSError("initial disk reserve insufficient")
    for name, value in protocol["threads"].items():
        if os.environ.get(name) != value:
            raise ValueError("single-thread environment required: " + name)
    runtime = load_runtime(root, protocol, protocol_path.parent)
    freeze = static_receipt(root, protocol_path, raw, original_raw, protocol, runtime)
    schedule = make_schedule(protocol)
    output.mkdir()
    for name, data in {"driver_source.py": Path(__file__).read_bytes(), "protocol.json": raw,
                       "original_protocol.json": original_raw, "schedule.json": json_bytes(schedule)}.items():
        exclusive(output / name, data)
    for name in ["velocity_oracle.py", "independent_saved_models.py"]:
        exclusive(output / name, (protocol_path.parent / name).read_bytes())
    status = dict(status="running", runs=schedule, source_before=freeze, started_utc=utc(),
        all48_scheduled_before_input_reads=True, student_final_design=False, research_success_claim=False,
        shared_bootstrap_seconds=perf_counter() - bootstrap,
        environment=dict(python=sys.version, platform=platform.platform(), threads=protocol["threads"]))
    atomic_json(output / "status.json", status)
    common_by_case = {}

    def check_static():
        if static_receipt(root, protocol_path, raw, original_raw, protocol, runtime) != freeze:
            raise StaleFreeze("executing source/protocol changed")

    try:
        for row in schedule:
            context = {}
            row.update(status="attempted", phase="before_preparation", started_utc=utc(), final_study_eligible=False,
                       primary_quality=None, progress_artifact=f"run_{row['run_index']:03d}_progress.json")
            atomic_json(output / "status.json", status)
            progress(output, row, context)
            def phase(name, details, *, row=row, context=context):
                context["phase"] = name
                row.update(phase=name, **details)
                progress(output, row, context)
            try:
                if shutil.disk_usage(output).free < protocol["disk"]["per_run_reserve_bytes"]:
                    raise OSError("per-run disk reserve insufficient")
                execute_timed(root, protocol, original, row, runtime, context, phase, check_static,
                              expected_common=common_by_case.get(row["case_index"]))
                returned = perf_counter()  # Caller observes complete API return, including its final payload.
                context["inside_timed_interval"] = False
                row.update(engine_status=context["result"]["status"], timing=timing_record(context, returned, protocol),
                           output_provenance=context["output_provenance"], common_binding=context["common_binding"])
                common_by_case.setdefault(row["case_index"], context["common_binding"])
                artifact_start = perf_counter()
                row.update(save_core(output, row, context, runtime))
                row["core_artifact_seconds"] = perf_counter() - artifact_start
                phase("post_return_integrity", {})
                integrity_start = perf_counter()
                check_static()
                row["input_hashes_after"] = hash_inputs(root, protocol)
                row["post_return_integrity_seconds"] = perf_counter() - integrity_start
                phase("offline_independent_audit", {})
                audit_start = perf_counter()
                audit = offline_audit(row, context, original, runtime, protocol)
                row["offline_audit_seconds"] = perf_counter() - audit_start
                audit_path = output / f"run_{row['run_index']:03d}_audit.json.gz"
                audit_save_start = perf_counter()
                new_gzip(audit_path, audit)
                row.update(audit_artifact=audit_path.name, audit_sha256=sha(audit_path.read_bytes()),
                           audit_artifact_seconds=perf_counter() - audit_save_start)
                if audit["failures"]:
                    raise ValueError("fresh independent audit failed; all failures retained")
                check_static()
                row.update(status="complete", phase="finished", final_study_eligible=True,
                    primary_quality=audit["own_native_model_errors"]["returned_final"]
                        if row["timing"]["runtime_gate"]["runtime_eligible"] else None)
                phi0, phi = context["result"]["initial_objective"], context["result"]["final_objective"]
                row["returned_targets"] = [dict(ratio=ratio, reached=bool(phi0 is not None and phi0 > 0 and phi is not None and phi <= ratio * phi0),
                    event_seconds=row["timing"]["actual_api_return_seconds"] if row["primary_quality"] is not None
                        and phi0 is not None and phi0 > 0 and phi is not None and phi <= ratio * phi0 else None)
                    for ratio in protocol["reported_returned_objective_ratios"]]
            except BaseException as exc:
                context["inside_timed_interval"] = False
                row.update(status="failed" if isinstance(exc, Exception) else "interrupted", error=f"{type(exc).__name__}: {exc}",
                    phase_at_failure=context.get("phase", row["phase"]), final_study_eligible=False, primary_quality=None,
                    preparation_physics_costs=context.get("preparation_physics_costs"),
                    in_flight_engine_counters_unknown="result" not in context)
                if "global_started" in context:
                    row["elapsed_until_failure_or_interrupt_seconds"] = perf_counter() - context["global_started"]
                preserve_failure(output, row, context, runtime)
                progress(output, row, context)
                atomic_json(output / "status.json", status)
                if not isinstance(exc, Exception) or isinstance(exc, (StaleFreeze, OSError)):
                    raise
            row["finished_utc"] = utc()
            progress(output, row, context)
            atomic_json(output / "status.json", status)
            print(json.dumps(dict(run_index=row["run_index"], method=row["method"], status=row["status"],
                actual_api_return=None if "timing" not in row else row["timing"]["actual_api_return_seconds"],
                primary_quality_eligible=row["primary_quality"] is not None)), flush=True)
        check_static()
        status.update(status="complete_engineering_comparison", source_after=freeze, finished_utc=utc(),
            input_hashes_after=hash_inputs(root, protocol), runs_attempted=48, research_goal_complete=False)
        atomic_json(output / "status.json", status)
        receipt = dict(status, artifact_sha256={p.name: sha(p.read_bytes()) for p in output.iterdir() if p.is_file()})
        atomic_json(output / "complete_receipt.json", receipt)
        return receipt
    except BaseException as exc:
        status.update(status="failed_or_interrupted", finished_utc=utc(), error=f"{type(exc).__name__}: {exc}")
        atomic_json(output / "status.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--execute-coordinate-engineering-comparison", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--expected-driver-sha256")
    args = parser.parse_args(argv)
    protocol, raw, _, _ = load_protocol(args.protocol)
    if not args.execute_coordinate_engineering_comparison:
        print(json.dumps(dict(status="metadata_only_no_project_arrays", planned_runs=len(make_schedule(protocol)),
                              protocol_sha256=sha(raw), driver_sha256=sha(Path(__file__).read_bytes()))))
        return 0
    if not all([args.repo, args.output, args.expected_protocol_sha256, args.expected_driver_sha256]):
        parser.error("explicit run requires repository/new output and both reviewed hashes")
    receipt = run(args.repo, args.output, args.protocol, args.expected_protocol_sha256, args.expected_driver_sha256)
    print(json.dumps(dict(status=receipt["status"], attempted=receipt["runs_attempted"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
