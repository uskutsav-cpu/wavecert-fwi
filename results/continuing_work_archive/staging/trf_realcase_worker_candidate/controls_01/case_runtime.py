"""Lazy real-case numerical source; imported only after exact worker GO/admission.

No numerical/production import occurs at this module's import. Manufactured
controls inject a declared fake public physics/runtime; they never call load_runtime.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def array_sha(np, value, dtype=None):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def save_arrays(np, path, **arrays):
    np.savez(path,**arrays)
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def load_runtime(repo, here, p):
    api = (here/"api").resolve()
    for name in ("wavecert.physics.helmholtz", "wavecert.physics.exact_merit", "matched_driver",
                 "normalized_exact", "bounded_exact", "cached_jacobian", "public_map_builder"):
        if name in sys.modules:
            raise ValueError("fresh worker has occupied numerical/native module: " + name)
    sys.path[:0] = [str(api), str(repo/"src")]
    np, scipy = importlib.import_module("numpy"), importlib.import_module("scipy")
    for module in (np, scipy):
        distribution = importlib.metadata.distribution(module.__name__)
        expected = Path(distribution.locate_file(module.__name__+"/__init__.py")).resolve()
        if Path(module.__file__).resolve() != expected or module.__version__ != p["runtime"][module.__name__]:
            raise ValueError("foreign executing numerical package/runtime")
    helm = importlib.import_module("wavecert.physics.helmholtz")
    exact = importlib.import_module("wavecert.physics.exact_merit")
    driver = importlib.import_module("matched_driver")
    binding = importlib.import_module("public_map_builder")
    package = (repo/"src/wavecert").resolve()
    for module, expected in ((helm, package/"physics/helmholtz.py"), (exact, package/"physics/exact_merit.py"),
            (driver, api/"matched_driver.py"), (binding, api/"public_map_builder.py")):
        if Path(module.__file__).resolve() != expected:
            raise ValueError("foreign imported native/API module origin")
    for cls, names, expected in ((helm.Helmholtz2D, ("operator", "source", "solve_state", "restrict", "cost_snapshot"), package/"physics/helmholtz.py"),
            (exact.ExactMeritOracle, ("evaluate", "accept", "gradient", "export_merit_fields", "validate_merit_fields", "cost_snapshot"), package/"physics/exact_merit.py")):
        for name in names:
            if Path(inspect.getsourcefile(getattr(cls, name))).resolve() != expected:
                raise ValueError("foreign executing public native method")
    scipy_opt = importlib.import_module("scipy.optimize")
    if driver.least_squares is not scipy_opt.least_squares or driver.engine.minimize is not scipy_opt.minimize:
        raise ValueError("real worker cannot use manufactured/replaced SciPy functions")
    interpolation = importlib.import_module("scipy.interpolate").RegularGridInterpolator
    return SimpleNamespace(np=np, helm=helm, exact=exact, driver=driver, binding=binding,
        interpolation=interpolation, api=api, package=package,
        origins={m.__name__: str(Path(m.__file__).resolve()) for m in (np, scipy, helm, exact, driver, binding)},
        manufactured_runtime=False)


def guard(clock, started, *, after=False):
    age = clock()-started
    if not math.isfinite(age) or age < 0:
        raise ValueError("invalid earlier absolute clock")
    if age > 7.5 or (not after and age >= 7.5):
        raise TimeoutError("absolute preparation/optimizer deadline")
    return age


def count_tree(value):
    if not isinstance(value, dict):
        raise TypeError("actual integer counter tree required")
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[key] = count_tree(item)
        elif type(item) is int and item >= 0:
            result[key] = item
        else:
            raise ValueError("native counter is not an actual nonnegative integer")
    return result


def snapshot(owner, prefix, *, failed=False):
    try:
        prefix["last_known_observation_counts"] = count_tree(owner.cost_snapshot())
        prefix["snapshot_completed"] = True
    except BaseException as exc:  # noqa: BLE001 -- do not replace failure or invent zeros
        prefix["snapshot_failure_type"] = type(exc).__name__
        prefix["unknown_work"] = True
    if failed:
        prefix["unknown_work"] = True


def decode_inputs(repo, here, p, slot, runtime):
    np = runtime.np
    prefix = "data/reference/subsurfacegen_parent_split/"
    raw_files = {}
    for relative, expected in p["inputs_sha256"].items():
        raw = (repo/relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("paid raw input bytes changed")
        raw_files[relative] = raw
    metadata = list(csv.DictReader(io.StringIO(raw_files[prefix+"validation_metadata.csv"].decode())))
    split = list(csv.DictReader(io.StringIO(raw_files[prefix+"split_manifest.csv"].decode())))
    with np.load(io.BytesIO(raw_files[prefix+"validation.npz"]), allow_pickle=False) as archive:
        if set(archive.files) != {"models", "source_indices", "frequencies_hz", "wavefields", "receiver_indices", "shape", "spacing"}:
            raise ValueError("raw validation NPZ key schema differs")
        # The full36d11 hash binds wavefields too; cached24grid wavefields are
        # deliberately never decoded or used to synthesize the new all8 data.
        models = archive["models"]
        sources, frequencies = archive["source_indices"], archive["frequencies_hz"]
        receivers, shape, spacing = archive["receiver_indices"], archive["shape"], archive["spacing"]
    if (models.shape != (192,24,24) or models.dtype != np.float32 or not np.isfinite(models).all()
            or np.any(models <= 0) or sources.shape != (192,) or sources.dtype != np.int64
            or frequencies.shape != (192,) or frequencies.dtype != np.float32
            or not np.isfinite(frequencies).all()
            or receivers.dtype != np.int64 or receivers.tolist() != p["original_receiver_indices"]
            or shape.dtype != np.int64 or shape.tolist() != [24,24]
            or spacing.shape != () or spacing.dtype != np.float64 or float(spacing) != .05
            or len(metadata) != 192):
        raise ValueError("raw validation NPZ/metadata dtype/shape/domain differs")
    case = p["cases"][slot["case_index"]]
    identity = case["identity"]
    rows = [i for i, row in enumerate(metadata) if row["slice_id"] == identity["slice_id"]]
    pairs = {(s,f) for s in p["original_source_indices"] for f in p["original_frequencies_hz"]}
    if (len(rows) != 24 or rows[0] != identity["array_index"]
            or {(int(sources[i]),float(frequencies[i])) for i in rows} != pairs
            or any(metadata[i]["split"] != "validation" or metadata[i]["model_id"] != identity["parent_id"]
                or metadata[i]["source_file"] != identity["source_file"]
                or metadata[i]["source_sha256"] != identity["source_sha256"] for i in rows)
            or any(not np.array_equal(models[i],models[rows[0]]) for i in rows)):
        raise ValueError("case24repeat/source/frequency/metadata binding differs")
    selected = [r for r in split if r["slice_id"] == identity["slice_id"]]
    if len(selected) != 1 or selected[0]["split"] != "validation" or selected[0]["model_id"] != identity["parent_id"]:
        raise ValueError("case split manifest differs")
    raw_model = models[rows[0]].astype(np.float64)
    if array_sha(np, raw_model, "<f8") != case["raw_float64_sha256"]:
        raise ValueError("raw24 model hash differs")
    initial_raw = (here/p["initializer_input"]["path"]).read_bytes()
    if hashlib.sha256(initial_raw).hexdigest() != p["initializer_input"]["sha256"]:
        raise ValueError("frozen initial input container differs")
    with np.load(io.BytesIO(initial_raw), allow_pickle=False) as archive:
        if set(archive.files) != {"initial_m", "case_indices", "shape"}:
            raise ValueError("frozen initializer NPZ keys differ")
        initial = archive["initial_m"]
        indices, initial_shape = archive["case_indices"], archive["shape"]
        if (initial.shape != (8,2401) or initial.dtype != np.float64
                or indices.shape != (8,) or indices.dtype != np.int64 or indices.tolist() != list(range(8))
                or initial_shape.shape != (2,) or initial_shape.dtype != np.int64 or initial_shape.tolist() != [49,49]):
            raise ValueError("frozen initializer NPZ schema differs")
        initial = initial[slot["case_index"]].copy()
    lo, hi = p["squared_slowness_bounds"]
    if (not np.isfinite(initial).all() or np.any(initial <= lo+1e-10) or np.any(initial >= hi-1e-10)
            or array_sha(np, initial, "<f8") != case["initial_float64_sha256"]):
        raise ValueError("exact retained direct initializer/box differs")
    return raw_model, initial


def prepare_case(args, p, accepted, gate, runtime, private, attempt, *, clock=time.perf_counter):
    np = runtime.np
    started = accepted["absolute_started"]
    prefix = {"phase": "raw_and_initial_loading", "observation_dispatches_attempted": 0,
        "observation_dispatches_completed": 0, "unknown_work": True,
        "last_known_observation_counts": None, "observation_call_inflight": False}
    attempt["preparation_prefix"] = prefix
    gate.context["preparation_prefix"] = prefix
    guard(clock, started)
    raw_model, initial = decode_inputs(args.repo, gate.base, p, accepted["slot"], runtime)
    velocity = 1/np.sqrt(raw_model)
    nodes = np.arange(1,50)/50*25
    zz, xx = np.meshgrid(nodes,nodes,indexing="ij")
    refined = runtime.interpolation((np.arange(26,dtype=float),)*2, np.pad(velocity,1,mode="edge"))(
        np.stack((zz,xx),axis=-1))
    truth = (1/refined**2).reshape(-1)
    if truth.shape != (2401,) or not np.isfinite(truth).all() or np.any(truth <= 0):
        raise ValueError("newly derived49 truth domain differs")
    # Private context is used only for generation and post-return metrics.
    # It is never a PolicyInput field or native factory argument.
    private.update(initial=initial, truth=truth, raw_model=raw_model)
    save_arrays(np,gate.output/"prepared_models.npz", raw24_m=raw_model, initial49_m=initial, truth49_m=truth)
    gate.event("raw_initial_truth_prepared_before_observations", initial_sha256=array_sha(np,initial,"<f8"),
        truth_sha256=array_sha(np,truth,"<f8"), old_truth_bitguard=False)
    guard(clock, started)
    config = dict(p["physics"], shape=tuple(p["physics"]["shape"]))
    physics = runtime.helm.Helmholtz2D(**config, factorization_cache_size=0)
    if type(physics) is not runtime.helm.Helmholtz2D or any(physics.cost_snapshot().values()):
        raise ValueError("genuine cold public observation owner required")
    private["observation_owner"] = physics
    snapshot(physics,prefix)
    rhs = np.column_stack([physics.source(source) for source in p["source_indices"]])
    observations, completed_indices = [], []
    try:
        for index, frequency in enumerate(p["frequencies_hz"]):
            guard(clock, started)
            prefix.update(phase="fresh_public_observations", observation_call_inflight=True,
                active_frequency_hz=frequency, unknown_work=True)
            prefix["observation_dispatches_attempted"] += 1
            gate.event("observation_dispatch_attempt", prefix=prefix)
            state = physics.solve_state(truth, rhs, frequency)
            if state.shape != (2401,6) or state.dtype != np.complex128 or not np.isfinite(state).all():
                raise ValueError("public observation state schema/domain differs")
            observed = physics.restrict(state,p["receiver_indices"]).T.copy()
            if observed.shape != (6,20) or not np.isfinite(observed).all():
                raise ValueError("ordered observation schema differs")
            observations.append(observed)
            completed_indices.extend(range(6*index,6*index+6))
            prefix["observation_dispatches_completed"] += 1
            prefix.update(observation_call_inflight=False, unknown_work=False)
            snapshot(physics,prefix)
            gate.event("observation_dispatch_returned", prefix=prefix)
            guard(clock, started, after=True)
    except BaseException:
        snapshot(physics,prefix,failed=True)
        if observations:
            save_arrays(np,gate.output/"partial_observations.npz", observations=np.concatenate(observations),
                block_indices=np.asarray(completed_indices,dtype=np.int64))
        raise
    expected = {key: 0 for key in physics.cost_snapshot()}
    expected.update(factorizations=8,forward_rhs=48)
    if physics.cost_snapshot() != expected or prefix["unknown_work"]:
        raise ValueError("actual complete public observation counts differ from8LU48RHS")
    observed = np.concatenate(observations)
    save_arrays(np,gate.output/"observations.npz", observed=observed)
    survey = {"frequencies_hz": p["frequencies_hz"], "sources": p["source_indices"], "receivers": p["receiver_indices"]}
    binding = {"case_index": accepted["slot"]["case_index"], "raw24_m_sha256": array_sha(np,raw_model,"<f8"),
        "initial49_m_sha256": array_sha(np,initial,"<f8"), "truth49_m_sha256": array_sha(np,truth,"<f8"),
        "all8_observations_sha256": array_sha(np,observed,"<c16"),
        "survey_sha256": hashlib.sha256(json.dumps(survey,sort_keys=True,separators=(",",":")).encode()).hexdigest()}
    gate.context["case_binding"] = binding
    gate.event("all8_data_prepared_before_policy", binding=binding, counts=expected)
    private["binding"] = binding
    blocks = tuple(runtime.helm.SurveyBlock(source,frequency,observed[6*fi+si].copy(),f"{source}@{frequency}")
        for fi,frequency in enumerate(p["frequencies_hz"]) for si,source in enumerate(p["source_indices"]))
    problem = runtime.driver.PolicyInput(initial_model=initial, blocks=blocks, receivers=tuple(p["receiver_indices"]),
        physics=tuple(config.items()), binding=tuple(binding.items()), preparation_costs=count_tree(expected))
    return problem


def make_native(runtime, problem, owners, event):
    """Only truth-free PolicyInput/runtime; no private prepared context here."""
    config = dict(problem.physics)
    template = runtime.helm.Helmholtz2D(**config,factorization_cache_size=0)
    native = runtime.exact.ExactMeritOracle(template,problem.blocks,problem.receivers,
        reference_package_root=runtime.package)
    if type(native) is not runtime.exact.ExactMeritOracle:
        raise ValueError("genuine frozen ExactMeritOracle required")
    owners.append(native)
    cold = count_tree(native.cost_snapshot())
    if any(value for branch in cold.values() for value in branch.values()):
        raise ValueError("policy owner was queried before construction")
    event("genuine_cold_native_owner_constructed", native_class=type(native).__name__, prefix=cold)
    return native


def quality_after_return(np, initial, selected, truth):
    reference = 1/np.sqrt(truth)
    denominator = float(np.linalg.norm(reference))
    return {"initial_relative_velocity_error": float(np.linalg.norm(1/np.sqrt(initial)-reference)/denominator),
        "selected_relative_velocity_error": None if selected is None else float(
            np.linalg.norm(1/np.sqrt(selected)-reference)/denominator), "computed_after_selection_returned": True,
        "used_for_selection": False}


def run_case(args, p, accepted, gate, *, runtime_factory=load_runtime, clock=time.perf_counter):
    runtime = runtime_factory(args.repo.resolve(),gate.base,p)
    np = runtime.np
    gate.context["runtime_origins"] = runtime.origins
    gate.event("actual_native_numerical_origins_before_case_work", origins=runtime.origins,
        manufactured_runtime=runtime.manufactured_runtime)
    owners, private, envelope = [], {}, None

    def admit_both():
        # Re-run project checks inside the reviewedAPIclock, on topofe967.
        gate.project_source_check()
        return {"project_protocol_sha256": gate.project_protocol_sha,
            "historical_API": runtime.driver.source_admission(runtime.api,
                json.loads((runtime.api/"protocol.json").read_bytes()))}

    try:
        envelope = runtime.driver.execute_one(arm=accepted["slot"]["arm"], admit=admit_both,
            prepare=lambda attempt: prepare_case(args,p,accepted,gate,runtime,private,attempt,clock=clock),
            native_factory=lambda problem: make_native(runtime,problem,owners,gate.event),
            callbacks_factory=lambda native,initial: runtime.binding.frozen_native_binding(native,
                reference_package_root=runtime.package,initial_model=initial,layout=runtime.binding.current_full_layout()),
            attempt=gate.attempt, lower_bound=p["squared_slowness_bounds"][0], upper_bound=p["squared_slowness_bounds"][1],
            absolute_started=accepted["absolute_started"],clock=clock)
        observed_return = clock()
        gate.event("API_returned_selection_frozen", observed_return=observed_return,
            selected=None if envelope["selected_model"] is None else array_sha(np,envelope["selected_model"],"<f8"))
        with (gate.output/"returned_payload.json").open("xb") as stream:
            stream.write(envelope["serialized"])
            stream.flush()
            os.fsync(stream.fileno())
        retained = {key: value for key,value in (("initial_m",envelope["initial_model"]),
            ("selected_m",envelope["selected_model"])) if value is not None}
        if retained:
            save_arrays(np,gate.output/"returned_models.npz",**retained)
        metrics = None
        if "truth" in private and envelope["initial_model"] is not None:
            metrics = quality_after_return(np,envelope["initial_model"],envelope["selected_model"],private["truth"])
        gate.event("post_return_quality_completed_without_reselection", metrics=metrics)
        return {"slot": accepted["slot"], "case_binding": private.get("binding"),
            "method_status": envelope["payload"]["status"], "method_policy_valid": envelope["payload"]["policy_valid"],
            "API_return_gate_only": runtime.driver.observe_return(envelope,observed_return),
            "post_return_quality": metrics, "complete_OS_process_eligibility_available": False,
            "entry_origin": accepted["absolute_started"], "worker_pre_result_age": clock()-accepted["absolute_started"]}
    except BaseException as exc:
        gate.context["original_case_exception"] = {"exception_type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        for index, owner in enumerate(owners):
            try:
                gate.context[f"native_final_cumulative_{index}"] = count_tree(owner.cost_snapshot())
            except BaseException as exc:  # noqa: BLE001 -- preserve unknown final native snapshot
                gate.context[f"native_snapshot_failure_{index}"] = type(exc).__name__
        if envelope is None:
            # Save available actual returned/current vectors separately when
            # the API failed before producing its fullyserialized envelope.
            failure_models = {}
            for label, value in (("initial_m",gate.attempt.get("initial_model")),
                    ("last_eligible_m",gate.attempt.get("last_eligible",{}).get("model")),
                    ("pending_final_m",(gate.attempt.get("engine_pending_prefix") or {}).get("final_model"))):
                if value is not None:
                    failure_models[label] = value
            if failure_models:
                try:
                    save_arrays(np,gate.output/"failure_models.npz",**failure_models)
                except BaseException as exc:  # noqa: BLE001 -- never mask the original API/serialization exception
                    gate.context["failure_model_persistence_error"] = {"exception_type": type(exc).__name__,
                        "message": str(exc), "available_vectors_remain_in_attempt_prefix": True}
