"""Held classical prior comparison; default inspects local metadata only.

No project study is authorized without a separately reviewed exact root GO.
All48 reused development slots are fixed before any project input read.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import gzip
import hashlib
import importlib
import importlib.util
import inspect
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

PRODUCTION = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
PROPOSAL = "48c26556658f5736b3191af03fa8eb858fdec162acdd35f2a9ad5d0ecfb786f5"
ADAPTER = "28f8bc9e6e71fd8c28165a8eddcb56e605cd7323286578180fabdf351e75a637"
METHODS = [("normalized_exact_data_only", 0.0), ("normalized_exact_fractional_update_prior", 100.0)]
FREQUENCIES = [.25, .5, 1., 2., 2.5, 3., 3.5, 4.]


class StaleFreeze(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def exclusive(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def atomic(path, value):
    temp = path.with_name(path.name + ".tmp")
    try:
        exclusive(temp, encoded(value))
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def artifact(path, raw, limit, reserve=2*1024**2):
    used = sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
    if used + len(raw) + reserve > limit:
        raise OSError("retained output/final receipt reservation exhausted")
    exclusive(path, raw)


def count_tree(value):
    if not isinstance(value, dict):
        raise TypeError("counter tree must be a dictionary")
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[key] = count_tree(item)
        elif isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
        else:
            raise ValueError("invalid nonnegative integer work count")
    return result


def schedule(protocol):
    rng, rows = random.Random(protocol["schedule_seed"]), []
    for repetition in range(3):
        for case in range(8):
            arms = list(METHODS)
            rng.shuffle(arms)
            for position, (method, weight) in enumerate(arms):
                rows.append({"run_index": len(rows), "case_index": case, "identity": protocol["identities"][case],
                    "repetition": repetition, "order_position": position, "method": method, "weight": weight,
                    "status": "unattempted", "final_study_eligible": False, "primary_quality": None})
    return rows


def load_protocol(path):
    p = json.loads(path.read_bytes())
    original = json.loads(path.with_name("original_protocol.json").read_bytes())
    proposal = json.loads(path.with_name("original_prior_proposal.json").read_bytes())
    if (p["production_source_sha256"] != PRODUCTION or p["production_file_count"] != 161
            or p["original_prior_proposal_sha256"] != PROPOSAL
            or p["local_dependency_sha256"]["composed_exact_adapter.py"] != ADAPTER
            or p["frequencies_hz"] != FREQUENCIES or p["observation_blocks"] != 48
            or p["engine_deadline_seconds"] != 7.5 or p["complete_return_cap_seconds"] != 8.
            or p["maximum_steps"] != 4096 or p["schedule_seed"] != 2026110901 or p["total_runs"] != 48
            or p["prior_smoothing_length_km"] != .1 or p["student_final_design"] is not False
            or p["methods"] != [{"name": name, "lambda": weight} for name, weight in METHODS]
            or p["continuation_gates"] != proposal["fixed_proposed_development_continuation_gates"]):
        raise StaleFreeze("fixed proposed development scope changed")
    for name in ("identities", "inputs_sha256", "physics", "squared_slowness_bounds"):
        if p[name] != original[name]:
            raise StaleFreeze("original input/physics selection changed: " + name)
    actual_geometry = dict(original["geometry"], frequencies_hz=FREQUENCIES, blocks_per_model=48)
    if p["original_geometry_context"] != original["geometry"] or p["geometry"] != actual_geometry:
        raise StaleFreeze("legacy context or actual all8/48 observation geometry changed")
    if p["lbfgsb"] != {"maximum_corrections": 10, "function_tolerance": 1e-12,
                          "gradient_tolerance": 1e-10, "maximum_line_search_trials": 12}:
        raise StaleFreeze("fixed native solver settings changed")
    for name, expected in p["local_dependency_sha256"].items():
        candidate = (path.parent / name).resolve()
        if not candidate.is_relative_to(path.parent.resolve()) or sha(candidate) != expected:
            raise StaleFreeze("local dependency changed: " + name)
    saved = json.loads(path.with_name("FROZEN48_SCHEDULE.json").read_bytes())
    if saved != schedule(p) or sha(path.with_name("FROZEN48_SCHEDULE.json")) != p["schedule_sha256"]:
        raise StaleFreeze("all48 pre-input slot order changed")
    return p, original


def validate_go(go, p, driver_sha, protocol_sha):
    required = {"authorization": "GO_CLASSICAL_PRIOR48_DEVELOPMENT", "driver_sha256": driver_sha,
        "protocol_sha256": protocol_sha, "schedule_sha256": p["schedule_sha256"],
        "production_source_sha256": PRODUCTION, "independent_source_review_clear": True}
    if any(go.get(key) != value for key, value in required.items()):
        raise StaleFreeze("separate root source/protocol/schedule GO required")


def import_file(name, path):
    if name in sys.modules:
        raise StaleFreeze("explicit import name occupied")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime(repo, path, p):
    sys.path.insert(0, str(repo / "src"))
    directory = repo / p["original_candidate_relative_directory"]
    for name, expected in p["candidate_files_sha256"].items():
        if sha(directory / name) != expected:
            raise StaleFreeze("raw preparation helper changed")
    name = "_prior48_original_preparation"
    if name in sys.modules:
        raise StaleFreeze("preparation package import name occupied")
    spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
        submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    helpers = importlib.import_module(name + ".verification")
    helpers._read_protocol(directory / "protocol.json")
    adapter = import_file("_prior48_reviewed_composition", path.parent / "composed_exact_adapter.py")
    dependencies = adapter.load_frozen_dependencies()
    exact = importlib.import_module("wavecert.physics.exact_merit")
    physics = importlib.import_module("wavecert.physics.helmholtz")
    for module, relative in ((exact, "exact_merit.py"), (physics, "helmholtz.py")):
        intended = repo / "src/wavecert/physics" / relative
        if Path(module.__file__).resolve() != intended or Path(inspect.getsourcefile(module.ExactMeritOracle if module is exact else module.Helmholtz2D)).resolve() != intended:
            raise StaleFreeze("foreign installed native module")
    if dependencies.engine.minimize is not importlib.import_module("scipy.optimize").minimize:
        raise StaleFreeze("native SciPy optimizer was replaced")
    audit = import_file("_prior48_raw_graph_only", path.parent / "independent_saved_models.py")
    for alias, target in (("np", "numpy"), ("scipy", "scipy"), ("sp", "scipy.sparse"),
                          ("spla", "scipy.sparse.linalg"), ("ndi", "scipy.ndimage")):
        setattr(audit, alias, importlib.import_module(target))
    return SimpleNamespace(helpers=helpers, adapter=adapter, dependencies=dependencies,
        exact=exact, physics=physics, audit=audit, np=importlib.import_module("numpy"))


def static_receipt(repo, path, p, runtime):
    actual = runtime.helpers.source_receipt(repo, path.with_name("original_protocol.json").read_bytes())
    if actual["repository_source_sha256"] != PRODUCTION:
        raise StaleFreeze("production source9f changed")
    local = {name: sha(path.parent / name) for name in p["local_dependency_sha256"]}
    if local != p["local_dependency_sha256"]:
        raise StaleFreeze("frozen local software changed")
    return {"production": actual, "local": local, "driver_sha256": sha(Path(__file__)), "protocol_sha256": sha(path)}


def prepare_problem(repo, original, p, row, runtime, context, expected_binding):
    # Called under the true run clock. No old controller/trajectory is used.
    raw = runtime.helpers.load_raw_inputs(repo, original)[row["case_index"]]
    context["raw_case"] = raw
    preparation = copy.deepcopy(original)
    preparation["geometry"]["frequencies_hz"] = list(FREQUENCIES)
    preparation["geometry"]["blocks_per_model"] = 48
    prepared = runtime.helpers.prepare_case(raw, preparation)
    context["prepared"] = prepared
    context["preparation_costs"] = count_tree(prepared["preparation_costs"])
    if (context["preparation_costs"]["factorizations"] != 8
            or context["preparation_costs"]["forward_rhs"] != 48):
        raise ValueError("fresh all8 data generation denominator differs")
    digest = hashlib.sha256(encoded(list(prepared["receivers"])))
    for block in prepared["blocks"]:
        digest.update(encoded([int(block.source_index), float(block.frequency_hz), str(block.label)]))
        digest.update(runtime.np.ascontiguousarray(block.observed, dtype="<c16").tobytes())
    binding = {"initial_native_m_sha256": runtime.adapter.model_sha(prepared["initial"]),
        "raw_model_sha256": prepared["raw_model_sha256"], "truth_sha256": prepared["truth_sha256"],
        "all8_observations_sha256": digest.hexdigest()}
    if expected_binding is not None and binding != expected_binding:
        raise StaleFreeze("repeated direct initial/observation bytes differ")
    context["common_binding"] = binding
    # This is the complete policy input boundary: no raw/true model or metric.
    return SimpleNamespace(initial=prepared["initial"], blocks=prepared["blocks"],
        receivers=prepared["receivers"], physics=copy.deepcopy(p["physics"]))


def native_owner(problem, runtime):
    config = dict(problem.physics, shape=tuple(problem.physics["shape"]))
    physics = runtime.physics.Helmholtz2D(**config, factorization_cache_size=0)
    owner = runtime.exact.ExactMeritOracle(physics, problem.blocks, problem.receivers)
    if type(owner) is not runtime.exact.ExactMeritOracle:
        raise ValueError("actual native owner required")
    cold = count_tree(owner.cost_snapshot())
    if any(value for branch in cold.values() for value in branch.values()):
        raise ValueError("native owner was queried before policy construction")
    return owner


def eligible_engine(result, adapter, deadline):
    if result["status"] not in {"solver_success", "solver_terminated", "budget_exhausted", "maximum_steps"} or not result["costs_complete"]:
        return False
    points = [c for c in result["checkpoints"] if c["within_budget"] and c["elapsed_seconds"] <= deadline]
    if not points or result["adapter"] is None:
        return False
    last, header = points[-1], result["adapter"]["current_accepted"]
    return bool(header is not None and last["model_sha256"] == adapter.model_sha(result["final_model"])
        == header["model_sha256"] and last["objective"] == result["final_objective"] == header["objective"])


def execute_policy(*, prepare, make_native, runtime, p, weight, check_static, attempt, clock=perf_counter):
    """Complete measured API; truth is unavailable through this interface."""
    started = clock()
    attempt.update(global_started=started, phase="source_and_input_preparation", preparation_counts_unknown=True)
    check_static()
    problem = prepare()
    attempt.update(phase="composed_optimizer", preparation_counts_unknown=False)
    owners = []

    def factory():
        owner = make_native(problem)
        owners.append(owner)
        return owner

    options = runtime.dependencies.engine.TrajectoryOptions(budget_seconds=p["engine_deadline_seconds"],
        maximum_steps=p["maximum_steps"], initial_step=.002, maximum_step=.02,
        maximum_line_search_trials=p["lbfgsb"]["maximum_line_search_trials"])
    try:
        result = runtime.adapter.run_composed_exact(native_oracle_factory=factory,
            initial_model=problem.initial, shape=tuple(p["physics"]["shape"]),
            grid_spacing=p["physics"]["spacing"], smoothing_length=p["prior_smoothing_length_km"],
            weight=weight, dependencies=runtime.dependencies, options=options, global_started=started,
            clock=clock, lower_bound=p["squared_slowness_bounds"][0], upper_bound=p["squared_slowness_bounds"][1],
            **{key: p["lbfgsb"][key] for key in ("maximum_corrections", "function_tolerance", "gradient_tolerance")})
    except BaseException as exc:
        attempt.update(phase="interrupted_optimizer", current_native_work_unknown=True,
            bounded_prefix=getattr(exc, "bounded_exact_pending_prefix", None),
            composed_prefix=getattr(exc, "composed_adapter_prefix", None),
            construction_prefix=getattr(exc, "composed_construction_prefix", None))
        raise
    attempt["engine_result"] = result
    valid = eligible_engine(result, runtime.adapter, p["engine_deadline_seconds"])
    tail = {"native_evaluate_attempts": 0, "native_evaluate_completed": 0,
        "prior_evaluation_attempts": 0, "prior_evaluations_completed": 0, "native_snapshot_calls": 0}
    attempt["metric_tail_counts"] = tail
    final_raw = final_prior = final_F = None
    initial_header = None if result["adapter"] is None else result["adapter"]["initial_accepted"]
    if valid:
        attempt["phase"] = "paid_final_common8_and_prior_metrics"
        if len(owners) != 1 or initial_header is None:
            raise ValueError("one authentic owner and initial header required")
        tail["native_evaluate_attempts"] += 1
        receipt = owners[0].evaluate(result["final_model"])
        tail["native_evaluate_completed"] += 1
        final_raw = dict(receipt.to_dict())
        current = result["adapter"]["current_accepted"]
        if (final_raw["model_sha256"] != current["model_sha256"]
                or final_raw["objective"] != current["native_data_receipt_header"]["objective"]):
            raise ValueError("final actual raw receipt differs from eligible composed point")
        if weight:
            tail["prior_evaluation_attempts"] += 1
            prior = runtime.dependencies.prior.FractionalVelocityPrior(problem.initial.reshape(p["physics"]["shape"]),
                p["physics"]["spacing"], p["prior_smoothing_length_km"])
            # Extra metric-anchor construction is real paid work, disclosed here.
            tail["final_metric_prior_anchor_constructions"] = 1
            final_prior = prior.evaluate(result["final_model"].reshape(p["physics"]["shape"])).value
            tail["prior_evaluations_completed"] += 1
        else:
            tail["final_metric_prior_anchor_constructions"] = 0
        phi0 = result["adapter"]["initial_paid_Phi0"]
        if initial_header["native_data_receipt_header"]["objective"] != phi0 or phi0 <= 0:
            raise ValueError("paid positive initial Phi0/header mismatch")
        if any(initial_header["native_data_receipt_header"][key] != final_raw[key]
               for key in ("geometry_sha256", "data_sha256")):
            raise ValueError("initial/final observation geometry/data binding differs")
        final_F = final_raw["objective"] / phi0 + weight * (final_prior or 0.)
        if not math.isclose(final_F, result["final_objective"], rel_tol=2e-12, abs_tol=2e-14):
            raise ValueError("final rawPhi/prior composition differs")
    tail["native_snapshot_calls"] += int(bool(owners))
    final_native = None if not owners else count_tree(owners[0].cost_snapshot())
    attempt["final_native_snapshot"] = final_native
    attempt["final_metrics"] = {"initial_native_header": initial_header,
        "final_raw_native_header": final_raw, "final_R": final_prior, "final_F": final_F}
    attempt["phase"] = "full_return_payload_validation_and_serialization"
    np = runtime.np
    models = {key: result[key] for key in ("initial_model", "final_model")}
    for model in models.values():
        if model.dtype != np.float64 or model.shape != problem.initial.shape or not np.isfinite(model).all():
            raise ValueError("invalid returned native vectors")
    trace = {key: value for key, value in result.items() if key not in models}
    payload = {"engine": trace, "weight": weight, "method_valid": valid, "initial_native_header": initial_header,
        "final_raw_native_header": final_raw, "final_R": final_prior, "final_F": final_F,
        "common_raw_Phi_ratio": None if not valid else final_raw["objective"] / result["adapter"]["initial_paid_Phi0"],
        "final_native_owner_cumulative_costs": final_native, "metric_tail_counts": tail,
        "counter_scope": "native owner cumulative snapshot once; adapter arithmetic and metric dispatch counts separate, never summed as equivalent work",
        "no_certificate_claim": True, "physical_acceptance": None,
        "initial_model": models["initial_model"].tolist(), "final_model": models["final_model"].tolist()}
    attempt["return_payload_before_serialization"] = payload
    serialized = encoded(payload)  # Includes both full native vectors and all trace/scalar metadata.
    attempt["phase"] = "returned"
    return {"payload": payload, "serialized": serialized, "models": models, "global_started": started,
        "inner_return_age": clock()-started}


def return_gate(measurement, returned, cap):
    wall = float(returned) - measurement["global_started"]
    inner = measurement["inner_return_age"]
    if isinstance(returned, bool) or not math.isfinite(wall) or not 0 <= inner <= wall:
        raise ValueError("invalid nested complete return timing")
    eligible = measurement["payload"]["method_valid"] is True and wall <= cap
    return {"actual_complete_return_seconds": wall, "inner_return_age": inner,
        "complete_return_cap_seconds": cap, "eligible": eligible, "event_seconds": wall if eligible else None}


def scalar_prior(model, initial, shape, spacing, length):
    q = [math.sqrt(float(a) / float(b)) - 1. for a, b in zip(initial, model, strict=True)]
    rows, cols = shape
    terms = [value * value for value in q]
    for z in range(rows):
        for x in range(cols):
            i = z * cols + x
            for j in ([i + cols] if z + 1 < rows else []) + ([i + 1] if x + 1 < cols else []):
                terms.append((length / spacing)**2 * (q[i] - q[j])**2)
    return math.fsum(terms) / (2 * len(q))


def offline_check(context, measurement, p, original, runtime, case_index):
    """New composed-F layer; frozen graph API computes only rawPhi/quality."""
    module, np = runtime.audit, runtime.np
    preparation = copy.deepcopy(original)
    preparation["geometry"]["frequencies_hz"] = list(FREQUENCIES)
    checks = module.Checks()
    graph = module.IndependentCase(context["raw_case"]["raw_model"], preparation, checks, case_index)
    actual_initial = measurement["models"]["initial_model"]
    initial_matches = actual_initial.shape == graph.initial.shape and all(
        math.isclose(float(a), float(b), rel_tol=module.TOLERANCES["initial_model_rtol"],
            abs_tol=module.TOLERANCES["initial_model_atol"])
        for a, b in zip(actual_initial, graph.initial, strict=True))
    checks.require(initial_matches, "initial vector versus independent registered initializer")
    prepared, payload = context["prepared"], measurement["payload"]
    records = {key: graph.evaluate(value) for key, value in measurement["models"].items()}
    for frequency, reference in graph.observations.items():
        actual = np.column_stack([b.observed for b in prepared["blocks"] if b.frequency_hz == frequency])
        error = module.scalar_norm(actual - reference) / module.scalar_norm(reference)
        checks.require(math.isfinite(error) and error <= p["audit_tolerances"]["observation_rtol"], "fresh all8 observations", error)
    if payload["method_valid"]:
        for key, header in (("initial_model", payload["initial_native_header"]["native_data_receipt_header"]),
                            ("final_model", payload["final_raw_native_header"])):
            graph.objective_check(records[key]["objective"], header["objective"], "raw native " + key)
            checks.require(records[key]["model_sha256"] == header["model_sha256"], "actual native model binding")
        own_R = scalar_prior(measurement["models"]["final_model"], measurement["models"]["initial_model"],
            p["physics"]["shape"], p["physics"]["spacing"], p["prior_smoothing_length_km"]) if payload["weight"] else 0.
        if payload["weight"]:
            checks.near(own_R, payload["final_R"], rtol=p["audit_tolerances"]["prior_rtol"],
                atol=p["audit_tolerances"]["prior_atol"], context="new scalar prior R", family="prior")
        phi0 = payload["initial_native_header"]["native_data_receipt_header"]["objective"]
        expected_F = records["final_model"]["objective"] / phi0 + payload["weight"] * own_R
        checks.near(expected_F, payload["final_F"], rtol=p["audit_tolerances"]["composed_F_rtol"],
            atol=p["audit_tolerances"]["composed_F_atol"], context="new composed F", family="composed_F")
    return {"records": records, "checks": checks.count, "numeric_comparisons": checks.numeric_count,
        "failures": checks.failures, "factorizations": graph.factorizations, "forward_rhs": graph.forward_rhs,
        "maximum_relative_linear_residual": graph.maximum_relative_linear_residual,
        "normalization_scope": "F uses the actual paid nativePhi0 after a separate graph initial-Phi check; it is fixed policy normalization, not an independently regenerated ratio",
        "scope": "post-return graph-only rawPhi/quality plus new independent scalar R/F; shares generic NumPy/SciPy; no old controller/trajectory auditor, policy feedback or physical acceptance"}


def continuation_summary(rows):
    groups = []
    slots = {(r["case_index"], r["repetition"], r["method"]) for r in rows}
    expected = {(case, repetition, name) for case in range(8) for repetition in range(3) for name, _ in METHODS}
    all_complete = len(rows) == 48 and slots == expected and all(r.get("final_study_eligible") is True for r in rows)
    for case in range(8):
        selected = {name: [r for r in rows if r["case_index"] == case and r["method"] == name] for name, _ in METHODS}
        valid = all(len(v) == 3 and all(r.get("final_study_eligible") for r in v) for v in selected.values())
        row = {"case_index": case, "all3_each_arm_eligible": valid, "successful_case": False, "harm_gate_pass": None}
        if valid:
            med = lambda values: sorted(values)[1]
            initial = med([r["initial_quality"]["relative_physical_velocity_L2"] for r in selected[METHODS[0][0]]])
            baseline = med([r["primary_quality"]["relative_physical_velocity_L2"] for r in selected[METHODS[0][0]]])
            candidate = med([r["primary_quality"]["relative_physical_velocity_L2"] for r in selected[METHODS[1][0]]])
            ratio = med([r["common_raw_Phi_ratio"] for r in selected[METHODS[1][0]]])
            row.update(initial_error=initial, baseline_error=baseline, candidate_error=candidate,
                candidate_raw_Phi_ratio=ratio, successful_case=candidate <= .9 * initial and candidate <= .9 * baseline and ratio <= .75,
                harm_gate_pass=candidate <= 1.1 * initial)
        groups.append(row)
    return {"all48_eligible": all_complete, "cases": groups,
        "proposed_development_continuation_gate_pass": all_complete and sum(r["successful_case"] for r in groups) >= 6 and all(r["harm_gate_pass"] is True for r in groups),
        "scientific_goal_complete": False, "scope": "fixed heuristic development gates, not IID inference or confirmation"}


def failure_evidence(attempt, np):
    """Preserve all available branches without oracle or clock re-query."""
    def jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [jsonable(item) for item in value]
        return value
    return {"attempt": jsonable(dict(attempt)),
        "completed_policy_return_observed": attempt.get("completed_policy_return_observed", False),
        "current_native_work_unknown": attempt.get("current_native_work_unknown", True),
        "final_native_snapshot_available": attempt.get("final_native_snapshot") is not None,
        "quality_success_claim": False,
        "scope": "All bounded/engine/composed/construction/metric/snapshot/payload branches retained; cumulative snapshots overlap and must not be summed. An available snapshot does not prove a completed API return. Unknown work is not zero."}


def persist_failure(output, row, attempt, np, limit):
    name = f"run_{row['run_index']:03d}_failure_evidence.json.gz"
    artifact(output / name, gzip.compress(encoded(failure_evidence(attempt, np)), mtime=0), limit, reserve=0)
    row.update(failure_prefix_artifact=name, failure_prefix_sha256=sha(output / name))


def run(repo, output, path, root_go, expected_driver, expected_protocol):
    bootstrap = perf_counter()
    repo, output, path = repo.resolve(), output.resolve(), path.resolve()
    p, original = load_protocol(path)
    if sha(Path(__file__)) != expected_driver or sha(path) != expected_protocol:
        raise StaleFreeze("explicit driver/protocol hash mismatch")
    validate_go(json.loads(root_go.read_bytes()), p, expected_driver, expected_protocol)
    if not sys.dont_write_bytecode:
        raise ValueError("execute with Python -B; no frozen-bundle bytecode creation")
    if output.exists() or output.is_relative_to(repo) or output.is_relative_to(path.parent):
        raise ValueError("fresh output outside repo/source required")
    if any(os.environ.get(key) != value for key, value in p["threads"].items()):
        raise ValueError("fixed single-thread environment required")
    if shutil.disk_usage(output.parent).free < p["disk"]["initial_free_bytes"]:
        raise OSError("initial physical disk reserve unavailable")
    runtime = load_runtime(repo, path, p)
    freeze = static_receipt(repo, path, p, runtime)
    rows, bindings = schedule(p), {}
    output.mkdir()
    for name in ["protocol.json", *p["local_dependency_sha256"]]:
        exclusive(output / name, (path.parent / name).read_bytes())
    exclusive(output / "driver_source.py", Path(__file__).read_bytes())
    exclusive(output / "root_go.json", root_go.read_bytes())
    status = {"status": "running", "runs": rows, "source_before": freeze, "started_utc": utc(),
        "all48_scheduled_before_project_inputs": True, "shared_bootstrap_seconds": perf_counter()-bootstrap,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": runtime.np.__version__,
            "scipy": importlib.import_module("scipy").__version__}, "scientific_goal_complete": False, "physical_acceptance": None}
    atomic(output / "status.json", status)

    def check_static():
        if static_receipt(repo, path, p, runtime) != freeze:
            raise StaleFreeze("source freeze changed")

    def check_inputs_after():
        for relative, expected in p["inputs_sha256"].items():
            candidate = (repo / relative).resolve()
            if not candidate.is_relative_to(repo) or sha(candidate) != expected:
                raise StaleFreeze("input bytes changed after complete return")

    try:
        for row in rows:
            context, attempt = {}, {}
            row.update(status="attempted", started_utc=utc())
            atomic(output / "status.json", status)
            if (shutil.disk_usage(output).free < p["disk"]["per_run_reserve_bytes"]
                    or sum(x.stat().st_size for x in output.iterdir() if x.is_file()) + 2*1024**2 >= p["disk"]["maximum_output_bytes"]):
                raise OSError("per-run physical/retained reserve unavailable")
            try:
                measurement = execute_policy(prepare=lambda row=row, context=context: prepare_problem(repo, original, p, row, runtime,
                    context, bindings.get(row["case_index"])), make_native=lambda problem: native_owner(problem, runtime),
                    runtime=runtime, p=p, weight=row["weight"], check_static=check_static, attempt=attempt)
                returned = perf_counter()  # Complete API return, including full JSON serialization.
                attempt.update(completed_policy_return_observed=True, actual_complete_return_stamp=returned)
                timing = return_gate(measurement, returned, p["complete_return_cap_seconds"])
                name = f"run_{row['run_index']:03d}_return.json.gz"
                artifact(output / name, gzip.compress(measurement["serialized"], mtime=0), p["disk"]["maximum_output_bytes"])
                row.update(return_artifact=name, return_sha256=sha(output / name), timing=timing,
                    engine_status=measurement["payload"]["engine"]["status"], common_binding=context.get("common_binding"),
                    preparation_costs=context.get("preparation_costs"), preparation_costs_unknown=False,
                    common_raw_Phi_ratio=measurement["payload"]["common_raw_Phi_ratio"])
                bindings.setdefault(row["case_index"], context["common_binding"])
                check_static()
                check_inputs_after()
                audit = offline_check(context, measurement, p, original, runtime, row["case_index"])
                name = f"run_{row['run_index']:03d}_audit.json.gz"
                artifact(output / name, gzip.compress(encoded(audit), mtime=0), p["disk"]["maximum_output_bytes"])
                eligible = timing["eligible"] and not audit["failures"]
                row.update(status="complete", audit_artifact=name, audit_sha256=sha(output / name),
                    integrity_valid=not audit["failures"], final_study_eligible=bool(eligible),
                    initial_quality=audit["records"]["initial_model"],
                    primary_quality=audit["records"]["final_model"] if eligible else None)
            except BaseException as exc:
                row.update(status="failed" if isinstance(exc, Exception) else "interrupted",
                    error=f"{type(exc).__name__}: {exc}", final_study_eligible=False, primary_quality=None,
                    incomplete_work_unknown=True, phase=attempt.get("phase"), preparation_costs=context.get("preparation_costs"),
                    preparation_costs_unknown=attempt.get("preparation_counts_unknown", True))
                # No available branch is discarded and interrupted owners are never re-queried.
                try:
                    persist_failure(output, row, attempt, runtime.np, p["disk"]["maximum_output_bytes"])
                except (OSError, ValueError, TypeError, OverflowError) as save_exc:
                    row["failure_prefix_save_error"] = f"{type(save_exc).__name__}: {save_exc}"
                atomic(output / "status.json", status)
                raise  # Unexpected infrastructure/interrupt: preserve remaining unattempted slots, no retry.
            row["finished_utc"] = utc()
            atomic(output / "status.json", status)
            print(json.dumps({"run_index": row["run_index"], "method": row["method"], "status": row["status"],
                "eligible": row["final_study_eligible"]}), flush=True)
        check_static()
        check_inputs_after()
        status.update(status="complete_engineering_attempts", finished_utc=utc(), source_after=freeze,
            continuation=continuation_summary(rows), numerical_audits_all_pass=all(r.get("integrity_valid") is True for r in rows))
        atomic(output / "status.json", status)
        complete = dict(status, artifact_sha256={x.name: sha(x) for x in output.iterdir() if x.is_file()})
        artifact(output / "complete_receipt.json", encoded(complete), p["disk"]["maximum_output_bytes"], reserve=0)
        return complete
    except BaseException as exc:
        status.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}", finished_utc=utc())
        atomic(output / "status.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--execute-study", action="store_true")
    for name in ("repo", "output", "root-go"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--expected-driver-sha256")
    parser.add_argument("--expected-protocol-sha256")
    args = parser.parse_args(argv)
    p, _ = load_protocol(args.protocol)
    if not args.execute_study:
        print(json.dumps({"status": "metadata_only_no_project_arrays_or_PDE", "planned_runs": len(schedule(p)),
            "schedule_sha256": p["schedule_sha256"], "driver_sha256": sha(Path(__file__)), "protocol_sha256": sha(args.protocol)}))
        return 0
    if not all((args.repo, args.output, args.root_go, args.expected_driver_sha256, args.expected_protocol_sha256)):
        parser.error("explicit repo/freshoutput/rootGO/source/protocol hashes required")
    result = run(args.repo, args.output, args.protocol, args.root_go, args.expected_driver_sha256, args.expected_protocol_sha256)
    return 0 if result["numerical_audits_all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
