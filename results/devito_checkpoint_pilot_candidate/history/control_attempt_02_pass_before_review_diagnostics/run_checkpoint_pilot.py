"""Held constant-medium replay control. Default: metadata, no numerical imports.

Future Linux workers are separately admitted by an exact source-bound root GO.
This file never installs, downloads, submits a job, retries, or chooses K.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("checkpoint_pilot_accounting", ROOT/"pilot_accounting.py")
_accounting = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_accounting)
AuditedKernel, AuditedStorage, Ledger = _accounting.AuditedKernel, _accounting.AuditedStorage, _accounting.Ledger
centered_difference, checkpoint_budget = _accounting.centered_difference, _accounting.checkpoint_budget
edge_pullback, relative_scalar_error = _accounting.edge_pullback, _accounting.relative_scalar_error
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def contract():
    return {name: sha(ROOT/name) for name in
            ("run_checkpoint_pilot.py", "pilot_accounting.py", "protocol.json", "RUNTIME_SOURCE_PINS.json")}


def admit(go_path):
    value = json.loads(go_path.read_bytes())
    if (value.get("authorization") != "GO_CONSTANT_MEDIUM_CHECKPOINT_PILOT"
            or value.get("execution_authorized") is not True
            or value.get("independent_source_review_clear") is not True
            or value.get("criteria_frozen_before_execution") is not True
            or value.get("source_files") != contract()):
        raise ValueError("separate exact source/protocol/runtime-pin root GO required")
    return {"path": str(go_path.resolve()), "sha256": sha(go_path), "content": value}


def environment(protocol):
    """Only called after GO; distribution metadata/source reads precede imports."""
    requested = protocol["runtime"]
    if platform.system() != "Linux" or platform.python_version() != requested["python"]:
        raise ValueError("registered Linux/Python runtime differs")
    versions = {name: importlib.metadata.version(name) for name in
                ("devito", "pyrevolve", "numpy", "scipy")}
    if any(versions[name] != requested[name] for name in versions):
        raise ValueError("runtime versions differ; driver performs no install or fallback")
    pins = json.loads((ROOT/"RUNTIME_SOURCE_PINS.json").read_bytes())
    checked = []
    for row in pins["files"]:
        installed = Path(importlib.metadata.distribution(row["distribution"]).locate_file(row["path"]))
        if sha(installed) != row["sha256"]:
            raise ValueError("installed primary source differs: "+row["path"])
        checked.append({**row, "installed_path": str(installed)})
    for name in THREADS:
        os.environ[name] = "1"
    for name in ("GITHUB_TOKEN", "GH_TOKEN", "ACTIONS_RUNTIME_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN"):
        os.environ.pop(name, None)
    os.environ.update(DEVITO_LANGUAGE="C", DEVITO_MPI="0", DEVITO_AUTOTUNING="off",
                      DEVITO_AUTOPADDING="0", DEVITO_SAFE_MATH="1", MPLBACKEND="Agg")
    return {"python": platform.python_version(), "packages": versions,
            "source_checks": checked, "threads": {name: os.environ[name] for name in THREADS},
            "devito_configuration_environment": {name: os.environ[name] for name in
                ("DEVITO_LANGUAGE", "DEVITO_MPI", "DEVITO_AUTOTUNING", "DEVITO_AUTOPADDING", "DEVITO_SAFE_MATH")}}


def rss():
    # Linux ru_maxrss is KiB. Compiler children are separate, nonadditive peaks.
    sampled = None
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                sampled = int(line.split()[1])*1024
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"worker_peak_RSS_bytes": int(own.ru_maxrss)*1024,
            "sampled_worker_RSS_bytes": sampled,
            "terminated_child_peak_RSS_bytes": int(children.ru_maxrss)*1024,
            "worker_user_CPU_seconds": own.ru_utime, "worker_system_CPU_seconds": own.ru_stime,
            "terminated_child_user_CPU_seconds": children.ru_utime,
            "terminated_child_system_CPU_seconds": children.ru_stime}


def array_hash(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def array_metadata(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype), "bytes": int(array.nbytes),
            "sha256": array_hash(array)}


def direction(np, name, shape, m0):
    i, j = np.indices(shape)
    if name == "interior":
        values = np.exp(-((i-20)/5)**2-((j-16)/4)**2)
        values[(i < 8) | (i > 32) | (j < 8) | (j > 24)] = 0
    elif name == "left_edge":
        values = np.exp(-(i/3)**2-((j-16)/5)**2)
        values[(i > 10) | (j < 6) | (j > 26)] = 0
    else:
        raise ValueError("unregistered physical-m direction")
    return m0*values/np.max(np.abs(values))


def native_arrays(model, u, v, grad, src, rec, dual):
    values = {"forward_u_allocated": u._data, "vp_allocated": model.vp._data,
        "damp_allocated": model.damp._data, "source": src.data,
        "source_coordinates": src.coordinates.data, "forward_receiver_scratch": rec.data,
        "forward_receiver_coordinates": rec.coordinates.data, "fixed_receiver_dual": dual.data,
        "dual_coordinates": dual.coordinates.data}
    if v is not None:
        values.update(reverse_v_allocated=v._data, gradient_allocated=grad._data)
    return {name: array_metadata(array) for name, array in values.items()}


def operator_metadata(operator, c_path):
    # Observational only; never request cfunction or an extra compilation here.
    compiler = getattr(operator, "_compiler", None)
    library = getattr(operator, "_lib", None)
    library_name = getattr(library, "_name", None)
    profiler = getattr(operator, "_profiler", None)
    timers = getattr(profiler, "py_timers", {})
    flags = {}
    for key in ("cc", "ld", "cflags", "ldflags", "src_ext", "so_ext"):
        value = getattr(compiler, key, None)
        if isinstance(value, (str, int, float, bool, list, tuple)):
            flags[key] = value
    path = Path(library_name) if isinstance(library_name, str) else None
    stat = path.stat() if path is not None and path.is_file() else None
    return {"generated_C_sha256": sha(c_path), "generated_C_bytes": c_path.stat().st_size,
        "compiler_class": type(compiler).__name__, "compiler_flags": flags,
        "loaded_library_path": library_name if isinstance(library_name, str) else None,
        "loaded_library_stat": None if stat is None else {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "available_native_python_timers": {str(key): float(value) for key, value in timers.items()
            if isinstance(value, (int, float)) and math.isfinite(value)},
        "cache_hit_or_cold_compile_inferred": False,
        "policy": "Shared cache may be warm; retain external compiler console logs. No cache clearing and no fair-speed claim."}


def worker(args, protocol, output, ledger):
    # No Devito/PyRevolve/NumPy imports occur at module import/default CLI/GO admission.
    import numpy as np
    from devito import Function, TimeFunction, CheckpointOperator, DevitoCheckpoint
    from examples.seismic.model import Model
    from examples.seismic.utils import AcquisitionGeometry
    from examples.seismic.acoustic.operators import ForwardOperator, GradientOperator
    from pyrevolve import Revolver

    p = protocol["constant_control"]
    nt, nbl, shape = p["nt"], p["nbl"], tuple(p["shape"])
    nsteps = nt-2
    dt, velocity = p["dt_ms"], p["velocity_m_per_ms"]
    m0 = 1.0/velocity**2
    physical_m = np.full(shape, m0, dtype=np.float64)
    if args.phase == "fd":
        dm = direction(np, args.direction, shape, m0)
        physical_m += args.sign*args.epsilon*dm
    if not np.all(np.isfinite(physical_m)) or not np.all(physical_m > 0):
        raise ValueError("nonpositive/nonfinite manufactured m")
    physical_vp = 1.0/np.sqrt(physical_m)
    before = time.perf_counter()
    model = Model(vp=physical_vp, origin=tuple(p["origin_m"]), spacing=tuple(p["spacing_m"]),
        shape=shape, nbl=nbl, space_order=p["space_order"], dtype=np.float64,
        fs=False, bcs="damp")
    if dt > float(model.critical_dt):
        raise ValueError("registered fixed dt exceeds native CFL estimate; no changed dt fallback")
    if not np.array_equal(model.vp.data, np.pad(physical_vp, nbl, mode="edge")):
        raise ValueError("actual Model extension differs from the registered edge map")
    geometry = AcquisitionGeometry(model, np.array(p["receiver_positions_m"], dtype=np.float64),
        np.array(p["source_position_m"], dtype=np.float64), p["t0_ms"], p["tn_ms"], src_type=None)
    geometry.resample(dt)
    axis = geometry.time_axis
    if (geometry.nt, axis.start, axis.stop, axis.step) != (nt, p["t0_ms"], p["tn_ms"], dt):
        raise ValueError("actual TimeAxis differs; no time interpolation or fit allowed")
    src = geometry.new_src(name="src", src_type=None)
    rec = geometry.new_rec(name="rec")
    dual = geometry.new_rec(name="fixed_dual")
    t = axis.time_values
    src.data[:, 0] = np.exp(-((t-40)/8)**2)-0.6*np.exp(-((t-160)/8)**2)
    dual.data[:] = np.sin(0.037*np.arange(nt)[:, None])*np.cos(0.29*(np.arange(geometry.nrec)+1)[None, :])
    dual.data[[0, -1], :] = 0
    rec.data.fill(0)
    if np.shares_memory(rec.data, dual.data) or np.shares_memory(src.data, rec.data):
        raise ValueError("replay scratch aliases source or fixed reverse dual")
    frozen = {"source": lambda: array_hash(src.data), "source_coordinates": lambda: array_hash(src.coordinates.data),
        "dual": lambda: array_hash(dual.data), "dual_coordinates": lambda: array_hash(dual.coordinates.data),
        "model_vp": lambda: array_hash(model.vp.data), "damping": lambda: array_hash(model.damp.data)}
    frozen_hashes = {key: value() for key, value in frozen.items()}
    full_save = args.phase == "saved"
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=p["space_order"],
                     save=nt if full_save else None)
    u.data  # allocate before explicitly zeroing every ring/history/halo plane
    u._data.fill(0)
    v = grad = None
    if args.phase != "fd":
        v = TimeFunction(name="v", grid=model.grid, time_order=2, space_order=p["space_order"])
        grad = Function(name="grad", grid=model.grid)
        v.data
        grad.data
        v._data.fill(0)
        grad._data.fill(0)
    if u.time_size != (nt if full_save else 3):
        raise ValueError("unexpected forward allocated time levels")
    setup_seconds = time.perf_counter()-before
    ledger.add("manufactured_setup_complete", seconds=setup_seconds, RSS=rss())
    fwd = ForwardOperator(model, geometry, save=full_save, kernel="OT2", space_order=p["space_order"])
    fw_native = CheckpointOperator(fwd, src=src, rec=rec, u=u, dt=dt, time_m=1, time_M=nt-2,
                                   **model.physical_params())
    if (int(fw_native.args["time_m"]), int(fw_native.args["time_M"]), int(fw_native.start_offset)) != (1, nt-2, 1):
        raise ValueError("prepared forward native bounds differ from the chosen discrete problem")
    fw = AuditedKernel(fw_native, "forward", nsteps, ledger, frozen)
    reverse = rev = wrp = storage_audit = budget = None
    if args.phase != "fd":
        reverse = GradientOperator(model, geometry, save=full_save, kernel="OT2", space_order=p["space_order"])
        rev_native = CheckpointOperator(reverse, u=u, v=v, grad=grad, rec=dual, dt=dt,
                                        time_m=1, time_M=nt-2, **model.physical_params())
        if (int(rev_native.args["time_m"]), int(rev_native.args["time_M"]), int(rev_native.start_offset)) != (1, nt-2, 1):
            raise ValueError("prepared reverse native bounds differ from the chosen discrete problem")
        rev = AuditedKernel(rev_native, "reverse", nsteps, ledger, frozen)
    if args.phase.startswith("checkpoint"):
        cp = DevitoCheckpoint([u])
        itemsize = int(np.dtype(cp.dtype).itemsize)
        budget = checkpoint_budget(int(cp.size), itemsize, args.checkpoints,
                                    protocol["memory"]["checkpoint_capacity_cap_bytes"])
        if cp.nbytes != budget["checkpoint_bytes"] or cp.size != 2*u._data[0].size:
            raise ValueError("checkpoint does not contain exactly two allocated temporal planes")
        ledger.add("checkpoint_capacity_admitted_before_storage_allocation", **budget, RSS=rss())
        wrp = Revolver(cp, fw, rev, args.checkpoints, nsteps, compression_params=None)
        if len(wrp.storage_list) != 1 or type(wrp.storage_list[0]).__name__ != "NumpyStorage":
            raise ValueError("only one ordinary uncompressed memory storage admitted")
        storage = wrp.storage_list[0]
        if (storage.storage.shape != (args.checkpoints, cp.size)
                or storage.storage.nbytes != budget["capacity_bytes"]
                or storage.storage.dtype != np.dtype(cp.dtype)):
            raise ValueError("actual checkpoint allocation differs from admitted capacity")
        storage_audit = AuditedStorage(storage, budget["checkpoint_bytes"], ledger)
        wrp.storage_list[0] = storage_audit
    ledger.phase = "initial_forward"
    start = time.perf_counter()
    if wrp is None:
        fw.apply(t_start=0, t_end=nsteps)
    else:
        wrp.apply_forward()
    forward_seconds = time.perf_counter()-start
    forward_counts = fw.counts().copy()
    if forward_counts["returned_steps"] != nsteps:
        raise ValueError("initial forward coverage count differs")
    record_data = np.array(rec.data, copy=True)
    objective = float(np.sum(record_data*dual.data))
    if not np.isfinite(objective) or not np.all(np.isfinite(record_data)):
        raise ValueError("nonfinite forward objective or record")
    ledger.add("initial_forward_complete", seconds=forward_seconds, native_counts=forward_counts,
               objective=objective, RSS=rss())
    reverse_seconds = 0.0
    if args.phase != "fd":
        ledger.phase = "reverse_with_paid_replay"
        start = time.perf_counter()
        if wrp is None:
            rev.apply(t_start=0, t_end=nsteps)
        else:
            wrp.apply_reverse()
        reverse_seconds = time.perf_counter()-start
        if rev.counts()["returned_steps"] != nsteps:
            raise ValueError("native reverse coverage count differs")
        reverse_indices = [index for call in rev.calls if call["status"] == "returned"
            for index in range(call["native_time_m"], call["native_time_M"]+1)]
        if sorted(reverse_indices) != list(range(1, nt-1)):
            raise ValueError("reverse native indices are not an exact once-only coverage")
        padded_gradient = np.array(grad.data, copy=True)
        physical_gradient = np.array(edge_pullback(padded_gradient.tolist(), shape, nbl), dtype=np.float64)
        if not np.all(np.isfinite(padded_gradient)) or not np.all(np.isfinite(physical_gradient)):
            raise ValueError("nonfinite native gradient")
        np.savez(output/"arrays.npz", receiver=record_data, padded_gradient=padded_gradient,
                 physical_gradient=physical_gradient)
        derivative = {name: float(np.sum(physical_gradient*direction(np, name, shape, m0))) for name in p["directions"]}
    else:
        derivative = None
    if {key: value() for key, value in frozen.items()} != frozen_hashes:
        raise ValueError("fixed native inputs changed")
    compiled_metadata = {}
    for name, operator in (("forward", fwd), ("reverse", reverse)):
        if operator is None:
            continue
        code = str(operator.ccode).encode()
        if len(code) > 512*1024:
            raise ValueError("generated C exceeds the fixed evidence cap")
        with (output/(name+".c")).open("xb") as stream:
            stream.write(code)
        compiled_metadata[name] = operator_metadata(operator, output/(name+".c"))
    arrays = native_arrays(model, u, v, grad, src, rec, dual)
    counts = fw.counts()
    resource_record = rss()
    if resource_record["worker_peak_RSS_bytes"] > protocol["memory"]["whole_child_peak_RSS_review_cap_bytes"]:
        raise ValueError("observed worker peak RSS exceeds review cap; no alternate configuration")
    return {"status": "WORKER_RETURNED_NO_SCIENTIFIC_SUCCESS_INFERRED", "phase": args.phase,
        "checkpoints": args.checkpoints, "fd_direction": args.direction, "fd_epsilon": args.epsilon,
        "fd_sign": args.sign, "objective": objective, "directional_adjoint": derivative,
        "input_hashes": frozen_hashes, "physical_m": array_metadata(physical_m),
        "physical_vp": array_metadata(physical_vp), "time_axis": {"start": axis.start, "stop": axis.stop,
            "step": axis.step, "num": axis.num}, "native_schedule_steps": nsteps,
        "prepared_bounds": {"forward": [int(fw_native.args["time_m"]), int(fw_native.args["time_M"])],
            "forward_offset": fw.start_offset, "reverse_offset": None if rev is None else rev.start_offset},
        "array_allocations": arrays, "checkpoint_budget": budget,
        "native_forward": counts, "initial_native_forward": forward_counts,
        "native_replay_forward_steps": counts["returned_steps"]-forward_counts["returned_steps"],
        "native_reverse": None if rev is None else rev.counts(),
        "checkpoint_copy_calls": [] if storage_audit is None else storage_audit.calls,
        "generated_operators_and_JIT_metadata": compiled_metadata,
        "JIT_cache_policy": protocol["JIT_cache_policy"],
        "timing_seconds": {"manufactured_model_source_allocation_setup": setup_seconds,
            "initial_forward_including_required_JIT": forward_seconds,
            "reverse_and_replay_including_required_JIT": reverse_seconds},
        "resources": resource_record,
        "working_buffer_accounting_limit": "Array totals are storage subtotals, not RSS estimates. Python/import/JIT/compiler/temporary and allocator memory is paid and observed separately. Initial-forward record_data and output gradient copies are extra arrays whose bytes are not hidden: receiver copy=%d; gradient copies=%d." % (record_data.nbytes, 0 if grad is None else padded_gradient.nbytes+physical_gradient.nbytes)}


def compare(args, protocol, output):
    import numpy as np
    inputs = args.inputs.resolve()
    rows, arrays = {}, {}
    def verified_child(directory, phase):
        row = json.loads((directory/"result.json").read_bytes())
        if (row["source_files"] != contract() or row["phase"] != phase
                or row["status"] != "WORKER_RETURNED_NO_SCIENTIFIC_SUCCESS_INFERRED"
                or row["intended_worker_return_code"] != 0):
            raise ValueError("comparison source/phase/completion report differs")
        for name, expected in row["output_artifacts"].items():
            if Path(name).name != name or sha(directory/name) != expected:
                raise ValueError("retained child evidence differs: "+name)
        return row
    for name in ("saved", "checkpoint2", "checkpoint4"):
        rows[name] = verified_child(inputs/name, name)
        with np.load(inputs/name/"arrays.npz", allow_pickle=False) as values:
            arrays[name] = {key: np.array(values[key], copy=True) for key in values.files}
    criteria = protocol["candidate_criteria_pending_review"]
    comparisons = []
    for name in ("checkpoint2", "checkpoint4"):
        if (rows[name]["input_hashes"] != rows["saved"]["input_hashes"]
                or rows[name]["time_axis"] != rows["saved"]["time_axis"]):
            raise ValueError("saved/checkpoint source/model/dual/time inputs differ")
        for key, threshold in (("receiver", criteria["receiver_relative_L2"]),
                ("padded_gradient", criteria["padded_gradient_relative_L2"]),
                ("physical_gradient", criteria["physical_gradient_relative_L2"])):
            reference, candidate = arrays["saved"][key], arrays[name][key]
            denominator = float(np.linalg.norm(reference))
            if reference.shape != candidate.shape or denominator <= 0 or not np.isfinite(denominator):
                raise ValueError("noninformative/nonfinite reference vector or shape mismatch")
            error = float(np.linalg.norm(candidate-reference)/denominator)
            comparisons.append({"variant": name, "quantity": key, "relative_L2": error,
                                "threshold": threshold, "pass": bool(np.isfinite(error) and error <= threshold)})
    finite_difference = []
    shape = tuple(protocol["constant_control"]["shape"])
    m0 = 1/protocol["constant_control"]["velocity_m_per_ms"]**2
    for name in protocol["constant_control"]["directions"]:
        dm = direction(np, name, shape, m0)
        g = arrays["saved"]["physical_gradient"]
        dot = float(np.sum(g*dm))
        denominator = float(np.linalg.norm(g)*np.linalg.norm(dm))
        cosine = abs(dot)/denominator if denominator > 0 else 0.0
        for index, epsilon in enumerate(protocol["finite_difference"]["epsilon"]):
            children = {}
            for sign in (-1, 1):
                child = verified_child(inputs/f"fd_{name}_{index}_{sign}", "fd")
                if (child["source_files"] != contract() or child["phase"] != "fd"
                        or child["fd_direction"] != name or child["fd_epsilon"] != epsilon or child["fd_sign"] != sign
                        or child["time_axis"] != rows["saved"]["time_axis"]
                        or any(child["input_hashes"][key] != rows["saved"]["input_hashes"][key]
                            for key in ("source", "source_coordinates", "dual", "dual_coordinates", "damping"))):
                    raise ValueError("FD child source/registration/fixed inputs differ")
                children[sign] = child
            difference = centered_difference(children[1]["objective"], children[-1]["objective"], epsilon)
            for variant in ("saved", "checkpoint2", "checkpoint4"):
                adjoint = rows[variant]["directional_adjoint"][name]
                error = relative_scalar_error(adjoint, difference)
                required = epsilon in protocol["finite_difference"]["required_middle_epsilon"]
                finite_difference.append({"direction": name, "epsilon": epsilon, "variant": variant,
                    "adjoint": adjoint, "centered_difference": difference, "relative_error": error,
                    "direction_cosine_against_saved_gradient": cosine, "required_middle_step": required,
                    "pass": bool(cosine >= criteria["nontrivial_direction_cosine_minimum"]
                        and (not required or error <= criteria["fd_middle_relative_error"]))})
    passed = all(row["pass"] for row in comparisons+finite_difference)
    result = {"status": "IMPLEMENTATION_AGREEMENT_PASS" if passed else "IMPLEMENTATION_AGREEMENT_FAIL",
        "source_files": contract(), "vector_comparisons": comparisons, "finite_difference": finite_difference,
        "criteria": criteria, "no_field_or_reconstruction_success_claim": True,
        "resource_and_actual_wait_review_required_separately": True}
    record(output/"comparison.json", result)
    if not passed:
        raise ValueError("one or more preregistered engineering comparisons failed")
    return {"status": result["status"], "comparison_sha256": sha(output/"comparison.json")}


def validate_worker_args(args, protocol):
    variants = {row["name"]: row for row in protocol["variants"]}
    if args.phase in variants:
        if args.checkpoints != variants[args.phase]["checkpoints"]:
            raise ValueError("explicit checkpoint count must exactly match the selected variant")
        if any(value is not None for value in (args.direction, args.epsilon, args.sign)):
            raise ValueError("gradient variant cannot carry FD parameters")
    elif args.phase == "fd":
        if (args.checkpoints is not None or args.direction not in protocol["constant_control"]["directions"]
                or args.epsilon not in protocol["finite_difference"]["epsilon"]
                or args.sign not in (-1, 1)):
            raise ValueError("exact registered FD direction/step/sign required")
    elif args.phase == "compare":
        if args.inputs is None or args.checkpoints is not None:
            raise ValueError("comparison needs retained child inputs and no checkpoint allocation")
    else:
        raise ValueError("an explicit registered phase is required")


def main():
    entered = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-go", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("saved", "checkpoint2", "checkpoint4", "fd", "compare"))
    parser.add_argument("--checkpoints", type=int)
    parser.add_argument("--direction", choices=("interior", "left_edge"))
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--sign", type=int, choices=(-1, 1))
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    protocol = json.loads((ROOT/"protocol.json").read_bytes())
    if not args.execute:
        print(json.dumps({"status": "HELD_METADATA_ONLY_NO_IMPORT_NO_PDE", "source_files": contract(),
            "runtime_install_authorized": False, "actual_study_launches": 0,
            "note": "No production or project data are used. Every future worker requires a separate reviewed root GO; this default constructs no arrays, operator or output directory."}))
        return 0
    if args.root_go is None or args.output is None:
        raise ValueError("reviewed root GO and fresh output directory required")
    validate_worker_args(args, protocol)
    go = admit(args.root_go)
    output = args.output.resolve()
    output.mkdir()  # refuses an existing directory; its parent must already exist
    ledger = Ledger(output/"events.jsonl")
    try:
        record(output/"admission.json", {"root_go": go, "source_files": contract(), "argv": sys.argv,
            "worker_bootstrap_scope": "External caller must record launch-to-return wall time and actual wait exit; entry clock excludes Python/std-library startup before main."})
        runtime = environment(protocol)
        record(output/"runtime.json", runtime)
        ledger.add("runtime_source_gate_pass", RSS=rss())
        if args.phase == "compare":
            result = compare(args, protocol, output)
        else:
            result = worker(args, protocol, output, ledger)
        result.update(source_files=contract(), root_go_sha256=go["sha256"],
                      worker_entry_through_pre_result_record_seconds=time.perf_counter()-entered,
                      intended_worker_return_code=0, resources_before_result_record=rss(),
                      output_artifacts={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
        record(output/"result.json", result)
        print(json.dumps({"status": result["status"], "output": str(output), "intended_worker_return_code": 0}))
        return 0
    except BaseException as error:
        record(output/"failure.json", {"status": "FAILED_OR_INTERRUPTED_NO_RETRY", "phase": args.phase,
            "error": type(error).__name__+": "+str(error), "source_files": contract(),
            "worker_entry_through_failure_seconds": time.perf_counter()-entered, "resources": rss(),
            "native_attempt_prefix_retained": (output/"events.jsonl").exists(),
            "no_scientific_success_inferred": True})
        print(json.dumps({"status": "FAILED_OR_INTERRUPTED_NO_RETRY", "output": str(output),
                          "error": type(error).__name__+": "+str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
