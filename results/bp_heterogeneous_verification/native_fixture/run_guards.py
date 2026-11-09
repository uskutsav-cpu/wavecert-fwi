"""Bounded BP local-subdomain verification; default CLI is metadata only."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import resource
import shutil
import signal
import subprocess
import sys
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_protocol(path):
    raw = path.read_bytes()
    p = json.loads(raw)
    if (p["physics"]["dt_ms"] != .5 or p["physics"]["half_dt_ms"] != .25 or p["physics"]["tn_ms"] != 1000
            or p["call_count"]["forward_total"] != 54 or p["call_count"]["adjoint_total"] != 3):
        raise ValueError("fixed bounded guard scope changed")
    if sha(path.with_name("guard_math.py").read_bytes()) != p["fixture_helper_sha256"]:
        raise ValueError("fixture helper source changed")
    return p, raw


def plan(protocol, only=None):
    rows = []
    for case in protocol["inputs"]:
        if only is not None and case["id"] != only:
            continue
        for variant in protocol["forward_calls"]:
            rows.append(dict(case_id=case["id"], call_id=variant["id"], kind="forward", status="unattempted"))
        rows.extend([dict(case_id=case["id"], call_id="probe_saved", kind="forward_saved", status="unattempted"),
                     dict(case_id=case["id"], call_id="probe_adjoint", kind="adjoint", status="unattempted")])
        for direction in ["interior", "edge"]:
            for step in protocol["gradient"]["relative_steps"]:
                for sign in ["plus", "minus"]:
                    rows.append(dict(case_id=case["id"], call_id=f"fd_{direction}_{step}_{sign}",
                                     kind="forward_fd", status="unattempted"))
    return rows


def load_runtime(repo, protocol):
    sys.path.insert(0, str(repo / "src"))
    backend = importlib.import_module("wavecert.physics.devito_backend")
    expected = repo / "src/wavecert/physics/devito_backend.py"
    if Path(backend.__file__).resolve() != expected.resolve() or sha(expected.read_bytes()) != protocol["source_pins"]["devito_backend.py"]:
        raise RuntimeError("frozen backend source/origin changed")
    package_modules = {"model.py": "examples.seismic.model", "operators.py": "examples.seismic.acoustic.operators",
                       "wavesolver.py": "examples.seismic.acoustic.wavesolver", "source.py": "examples.seismic.source",
                       "utils.py": "examples.seismic.utils"}
    paths = {"devito_backend.py": expected}
    for filename, name in package_modules.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if sha(path.read_bytes()) != protocol["source_pins"][filename]:
            raise RuntimeError("pinned installed source changed: " + filename)
        paths[filename] = path
    provenance = importlib.import_module("wavecert.provenance")
    if provenance.source_digest(repo) != protocol["production_source_sha256"]:
        raise RuntimeError("source9f changed")
    spec = importlib.util.spec_from_file_location("_bp_guard_math_frozen", Path(__file__).with_name("guard_math.py"))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return dict(np=importlib.import_module("numpy"), backend=backend, devito=importlib.import_module("devito"),
                seismic=importlib.import_module("examples.seismic"), psutil=importlib.import_module("psutil"),
                helpers=helpers, paths=paths, provenance=provenance)


def source_record(runtime, repo, input_path, protocol_path):
    return dict(production=runtime["provenance"].source_digest(repo), input_sha256=sha(input_path.read_bytes()),
                protocol_sha256=sha(protocol_path.read_bytes()), driver_sha256=sha(Path(__file__).read_bytes()),
                helper_sha256=sha(Path(__file__).with_name("guard_math.py").read_bytes()),
                installed={name: sha(path.read_bytes()) for name, path in runtime["paths"].items()})


def save_arrays(path, arrays, runtime, protocol):
    runtime["helpers"].ensure_no_wavefield_arrays(arrays, protocol["memory"]["per_artifact_uncompressed_array_byte_cap"])
    with path.open("xb") as stream:
        runtime["np"].savez_compressed(stream, **arrays)


def model_record(adapter, requested, runtime):
    np, helpers = runtime["np"], runtime["helpers"]
    padded = np.asarray(adapter.model.vp.data, dtype=np.float64)
    nbl, shape = int(adapter.model.nbl), tuple(adapter.model.shape)
    physical = padded[nbl:nbl + shape[0], nbl:nbl + shape[1]]
    if not np.array_equal(physical, requested):
        raise ValueError("actual physical velocity differs from requested values")
    return dict(physical_velocity_sha256=helpers.array_sha(physical), padded_velocity_sha256=helpers.array_sha(padded),
                actual_physical_m_sha256=helpers.array_sha(1 / physical ** 2), actual_padded_m_sha256=helpers.array_sha(1 / padded ** 2),
                shape_x_z=list(shape), padded_shape=list(padded.shape), grid_origin_m=[float(v) for v in adapter.model.grid.origin],
                maximum_velocity_kmps=float(np.max(padded)), dt_ms=adapter.dt, critical_dt_ms=float(adapter.model.critical_dt))


def make_adapter(velocity, spacing, nbl, dt, protocol, runtime):
    source, receivers = runtime["helpers"].geometry(protocol)
    config = runtime["backend"].DevitoAcousticConfig(spacing=(spacing, spacing), nbl=nbl,
        t0=0., tn=1000., f0=.005, dt=dt, max_velocity=5., space_order=4, dtype="float64")
    return runtime["backend"].DevitoAcousticFWI(velocity, source_coordinates=source, receiver_coordinates=receivers, config=config)


def forward(adapter, velocity, scale, save, runtime, protocol):
    np, helpers = runtime["np"], runtime["helpers"]
    adapter.update_velocity(velocity)
    source, before = helpers.held_source(adapter, scale)
    receiver = adapter.geometry.rec
    receiver.data[:] = 0
    nt = len(adapter.time_values)
    u = runtime["devito"].TimeFunction(name="u", grid=adapter.model.grid, time_order=2,
                                      space_order=4, save=nt if save else None, dtype=np.float64)
    allocated = tuple(int(n) for n in u.shape_allocated)
    allocated_bytes = math.prod(allocated) * 8
    if save:
        if adapter.config.nbl != 40 or adapter.config.spacing != (12.5, 12.5) or allocated_bytes > protocol["memory"]["saved_byte_cap"]:
            raise MemoryError("only bounded coarse40 saved field is permitted")
        if allocated != tuple(protocol["memory"]["actual_saved_layout"]):
            raise MemoryError("saved layout differs from the reviewed metadata-only allocation estimate")
        available = runtime["psutil"].virtual_memory().available
        if available < protocol["memory"]["minimum_available_RAM_before_saved_allocation_bytes"]:
            raise MemoryError("insufficient available RAM for the single saved field")
    u.data_with_halo[:] = 0
    actual_allocated_shape = tuple(int(n) for n in u._data.shape)
    actual_allocated_bytes = int(u._data.nbytes)
    if actual_allocated_shape != allocated or actual_allocated_bytes != allocated_bytes:
        raise MemoryError("materialized allocation differs from the pre-allocation byte estimate")
    operator = adapter.solver.op_fwd(save=save)
    arguments = operator.arguments(src=source, rec=receiver, u=u, vp=adapter.model.vp, dt=adapter.dt)
    bounds = {name: int(arguments[name]) for name in ["time_m", "time_M"]}
    del arguments
    if bounds != dict(time_m=1, time_M=nt - 2):
        raise ValueError("computed receiver range differs from frozen1..nt-2")
    receiver, u = adapter.solver.forward(src=source, rec=receiver, u=u, vp=adapter.model.vp, save=save, dt=adapter.dt)[0:2]
    times = adapter.time_values
    actual_times = np.asarray(receiver._time_range.time_values, dtype=np.float64)
    source_times_after = np.asarray(source._time_range.time_values, dtype=np.float64)
    source_coordinates_after = np.array(source.coordinates.data, copy=True)
    mask = (np.arange(nt) >= 1) & (np.arange(nt) <= nt - 2)
    values = np.array(receiver.data, copy=True)
    if (not np.array_equal(times, actual_times) or values.shape != (nt, 10) or not np.isfinite(values).all()
            or np.linalg.norm(values[mask]) == 0 or not np.all(values[[0, -1]] == 0)
            or not np.array_equal(before["source"], source.data)
            or not np.array_equal(before["source_times_ms"], times)
            or not np.array_equal(before["source_times_ms"], source_times_after)
            or not np.array_equal(before["source_coordinates"], source_coordinates_after)):
        raise ValueError("raw receiver/source sampling, nontriviality or sentinel invariant failed")
    expected = scale * (1 - 2 * (np.pi * .005 * (times - 200)) ** 2) * np.exp(-(np.pi * .005 * (times - 200)) ** 2)
    source_error = helpers.relative_record(source.data[:, 0] - expected, expected)
    if source_error["ratio"] is None or source_error["ratio"] > 1e-14:
        raise ValueError("held actual Source differs from fixed scaled Ricker")
    arrays = dict(receiver=values, times_ms=times, computed_mask=mask, source_before=before["source"],
        source_after=np.array(source.data, copy=True), source_times_ms=before["source_times_ms"],
        source_times_after_ms=source_times_after.copy(), source_coordinates=before["source_coordinates"],
        source_coordinates_after=source_coordinates_after,
        receiver_coordinates=np.array(receiver.coordinates.data, copy=True))
    arrays.update(actual_physical_velocity_kmps=np.array(velocity, copy=True), actual_physical_m=1 / velocity ** 2)
    origin = np.asarray(adapter.model.grid.origin, dtype=np.float64)
    spacing = np.asarray(adapter.config.spacing, dtype=np.float64)
    record = dict(model=model_record(adapter, velocity, runtime), source_scale=scale,
        physical_point_strength_m2=scale * adapter.config.spacing[0] * adapter.config.spacing[1], time_bounds=bounds,
        source_relative_error=source_error, source_sha256=helpers.array_sha(before["source"]),
        time_sha256=helpers.array_sha(times), receiver_sha256=helpers.array_sha(values),
        source_time_after_sha256=helpers.array_sha(source_times_after),
        source_coordinates_sha256=helpers.array_sha(before["source_coordinates"]),
        source_coordinates_after_sha256=helpers.array_sha(source_coordinates_after),
        receiver_coordinates_sha256=helpers.array_sha(arrays["receiver_coordinates"]),
        computational_source_indices=((before["source_coordinates"] - origin) / spacing).tolist(),
        computational_receiver_indices=((arrays["receiver_coordinates"] - origin) / spacing).tolist(),
        allocated_wavefield_shape=list(allocated), allocated_wavefield_bytes=allocated_bytes, save_wavefield=save,
        materialized_wavefield_shape=list(actual_allocated_shape), materialized_wavefield_bytes=actual_allocated_bytes,
        call="solver.forward(src=held_source,rec=held_receiver,u=explicit_u,save=fixed,dt=fixed,vp=model.vp)")
    return arrays, record, u, source



def adjoint_gradient(adapter, u, held, residual_values, runtime, on_bounds=lambda bounds: None):
    """Native adjoint wiring shared by project workers and manufactured API control."""
    np, helpers = runtime["np"], runtime["helpers"]
    gradient = runtime["devito"].Function(name="grad", grid=adapter.model.grid, dtype=np.float64)
    gradient.data[:] = 0
    residual = runtime["seismic"].Receiver(name="bp_residual", grid=adapter.model.grid,
        time_range=adapter.geometry.time_axis, coordinates=adapter.geometry.rec_positions)
    residual.data[:] = residual_values
    adjoint_field = runtime["devito"].TimeFunction(name="v", grid=adapter.model.grid,
        time_order=2, space_order=4, dtype=np.float64)
    adjoint_field.data_with_halo[:] = 0
    arguments = adapter.solver.op_grad().arguments(rec=residual, u=u, v=adjoint_field,
        vp=adapter.model.vp, grad=gradient, dt=adapter.dt)
    bounds = {name: int(arguments[name]) for name in ["time_m", "time_M"]}
    del arguments
    on_bounds(bounds)
    if bounds != dict(time_m=1, time_M=len(residual_values) - 2):
        raise ValueError("actual adjoint time bounds differ from fixed residual mask")
    adapter.solver.gradient(rec=residual, u=u, v=adjoint_field, src=held,
                           vp=adapter.model.vp, grad=gradient, dt=adapter.dt)
    padded = np.array(gradient.data, dtype=np.float64, copy=True)
    shape, nbl = tuple(adapter.model.shape), int(adapter.model.nbl)
    prod = runtime["backend"]._edge_padding_pullback(padded, shape, nbl)
    independent = helpers.independent_edge_fold(padded, shape, nbl)
    crop = padded[nbl:nbl + shape[0], nbl:nbl + shape[1]].copy()
    fold = helpers.relative_record(prod - independent, independent)
    numerical_agreement = bool(np.isfinite(padded).all() and fold["finite"]
        and not (fold["denominator"] == 0 and fold["numerator"] != 0)
        and (fold["ratio"] is None or fold["ratio"] <= 1e-13))
    arrays = dict(padded_gradient=padded, production_fold=prod, independent_fold=independent,
                  naive_crop=crop, residual=residual_values)
    record = dict(fold=fold, padded_gradient_sha256=helpers.array_sha(padded), actual_time_bounds=bounds,
        pullback_numerical_agreement=numerical_agreement,
        adjoint_rolling_allocated_shape=[int(n) for n in adjoint_field.shape_allocated])
    return arrays, record


def worker(repo, inputs, output, protocol_path, case_id):
    protocol, _ = load_protocol(protocol_path)
    runtime = load_runtime(repo, protocol)
    np, helpers = runtime["np"], runtime["helpers"]
    case = next(item for item in protocol["inputs"] if item["id"] == case_id)
    input_path = inputs / case["file"]
    before = source_record(runtime, repo, input_path, protocol_path)
    if before["input_sha256"] != case["sha256"] or output.exists():
        raise ValueError("new output and frozen BP input required")
    output.mkdir()
    status = dict(status="running", case=case, source_before=before, calls=plan(protocol, case_id), physical_acceptance=None,
        derivative_checks=[dict(direction=name, relative_step=step, status="unattempted")
                           for name in ["interior", "edge"] for step in protocol["gradient"]["relative_steps"]])
    source_coordinates, receiver_coordinates = helpers.geometry(protocol)
    global_origin = np.asarray(case["global_origin_z_x_m"])[::-1]
    status["geometry"] = dict(role=protocol["geometry"]["role"], global_crop_origin_x_z_m=global_origin.tolist(),
        local_source_x_z_m=source_coordinates.tolist(), local_receivers_x_z_m=receiver_coordinates.tolist(),
        global_source_x_z_m=(source_coordinates + global_origin).tolist(),
        global_receivers_x_z_m=(receiver_coordinates + global_origin).tolist())
    write_json(output / "status.json", status)
    base_records = {}
    def attempt(call_id, operation, summarize=lambda value: value):
        row = next(item for item in status["calls"] if item["call_id"] == call_id)
        row["status"] = "attempted"
        write_json(output / "status.json", status)
        try:
            result = operation()
            row["result"] = summarize(result)
            row["status"] = "complete"
            return result
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, (MemoryError, OSError)):
                raise
            return None
        except BaseException as exc:
            row.update(status="interrupted", error=f"{type(exc).__name__}: {exc}", in_flight_wave_work_unknown=True)
            raise
        finally:
            write_json(output / "status.json", status)
    try:
        raw = np.load(input_path, allow_pickle=False)
        if raw.shape != (96, 192) or raw.dtype != np.float32 or not np.isfinite(raw).all() or (raw <= 0).any():
            raise ValueError("retained stride BP schema changed")
        velocity = np.ascontiguousarray(raw.T, dtype=np.float64) / 1000
        fine = helpers.refine_2x(velocity)
        for variant in protocol["forward_calls"]:
            def run_variant(variant=variant):
                values = fine if variant["id"] == "fine160" else velocity
                adapter = make_adapter(values, variant["spacing_m"], variant["nbl"], variant["dt_ms"], protocol, runtime)
                arrays, record, u, source = forward(adapter, values, variant["source_scale"], False, runtime, protocol)
                filename = variant["id"] + ".npz"
                save_arrays(output / filename, arrays, runtime, protocol)
                record.update(artifact=filename, artifact_sha256=sha((output / filename).read_bytes()))
                base_records[variant["id"]] = dict(arrays=arrays, record=record)
                del u, source, adapter
                gc.collect()
                return record
            record = attempt(variant["id"], run_variant)
            if record is not None:
                status.setdefault("forward_records", {})[variant["id"]] = record
        if "coarse40" not in base_records:
            raise RuntimeError("gradient observation unavailable; retain unattempted dependents")
        observed = base_records["coarse40"]["arrays"]["receiver"]
        probe_velocity = .99 * velocity
        adapter = make_adapter(probe_velocity, 12.5, 40, .5, protocol, runtime)
        def saved_probe():
            result = forward(adapter, probe_velocity, 1., True, runtime, protocol)
            arrays, record = result[:2]
            filename = "probe_saved_record.npz"
            save_arrays(output / filename, arrays, runtime, protocol)
            record.update(artifact=filename, artifact_sha256=sha((output / filename).read_bytes()))
            return result
        saved = attempt("probe_saved", saved_probe, lambda result: result[1])
        if saved is None:
            raise RuntimeError("saved forward unavailable; retain dependent calls")
        predicted, forward_record, u, held = saved
        status["probe_forward"] = forward_record
        valid = predicted["computed_mask"]
        residual_values = predicted["receiver"] - observed
        residual_values[~valid] = 0
        if np.linalg.norm(residual_values) == 0:
            raise ValueError("zero probe residual is uninformative")
        def adjoint(u=u, held=held):
            row = next(item for item in status["calls"] if item["call_id"] == "probe_adjoint")
            def persist_bounds(bounds):
                row["actual_time_bounds"] = bounds
                write_json(output / "status.json", status)
            arrays, record = adjoint_gradient(adapter, u, held, residual_values, runtime, persist_bounds)
            filename = "gradient_arrays.npz"
            save_arrays(output / filename, arrays, runtime, protocol)
            record.update(artifact=filename, artifact_sha256=sha((output / filename).read_bytes()))
            row["retained_gradient_record"] = record
            write_json(output / "status.json", status)
            if not record["pullback_numerical_agreement"]:
                raise ValueError("independent/production padding pullback mismatch; raw gradient retained")
            return arrays["production_fold"], arrays["naive_crop"], record
        gradients = attempt("probe_adjoint", adjoint, lambda result: result[2])
        del adjoint, u, saved, held
        gc.collect()
        if gradients is None:
            raise RuntimeError("adjoint unavailable; no FD agreement claim")
        production, crop, gradient_record = gradients
        status["padding_pullback_comparison"] = gradient_record["fold"]
        model = 1 / probe_velocity ** 2
        derivatives = []
        for label in ["interior", "edge"]:
            direction, scale, mask = helpers.parameter_direction(model, label)
            filename = label + "_direction.npz"
            save_arrays(output / filename, dict(direction=direction, mask=mask, probe_m=model), runtime, protocol)
            status.setdefault("direction_witnesses", {})[label] = dict(artifact=filename,
                artifact_sha256=sha((output / filename).read_bytes()), norm_scale=scale,
                direction_sha256=helpers.array_sha(direction), probe_m_sha256=helpers.array_sha(model))
            write_json(output / "status.json", status)
            adjoint_value = float(np.sum(production * direction))
            cropped_value = float(np.sum(crop * direction))
            for step in protocol["gradient"]["relative_steps"]:
                epsilon, results = step * scale, {}
                row = next(item for item in status["derivative_checks"] if item["direction"] == label and item["relative_step"] == step)
                row.update(status="attempted", epsilon=epsilon, adjoint_derivative=adjoint_value,
                           naive_crop_derivative=cropped_value, finite_difference=None, informative=False)
                write_json(output / "status.json", status)
                for sign, sign_value in [("plus", 1.), ("minus", -1.)]:
                    def fd_call(sign=sign, sign_value=sign_value, epsilon=epsilon, direction=direction, label=label, step=step):
                        requested_m = model + sign_value * epsilon * direction
                        trial = 1 / np.sqrt(requested_m)
                        try:
                            arrays, record, field, source = forward(adapter, trial, 1., False, runtime, protocol)
                            arrays.update(requested_m=requested_m, requested_trial_velocity_kmps=trial)
                            filename = f"fd_{label}_{step}_{sign}.npz"
                            save_arrays(output / filename, arrays, runtime, protocol)
                            record.update(requested_m_sha256=helpers.array_sha(requested_m),
                                artifact=filename, artifact_sha256=sha((output / filename).read_bytes()),
                                objective=.5 * float(np.sum((arrays["receiver"][valid] - observed[valid]) ** 2)))
                            del field, source
                            return record
                        finally:
                            call_row = next(item for item in status["calls"] if item["call_id"] == f"fd_{label}_{step}_{sign}")
                            try:
                                adapter.update_velocity(probe_velocity)
                                restored = model_record(adapter, probe_velocity, runtime)
                                call_row["restoration"] = dict(status="restored", model=restored)
                                if restored["actual_physical_m_sha256"] != helpers.array_sha(model):
                                    raise ValueError("FD finally did not restore the exact probe model")
                                if "record" in locals():
                                    record["restored_probe_model"] = restored
                            except BaseException as restore_error:
                                call_row["restoration"] = dict(status="failed_or_unknown", error=f"{type(restore_error).__name__}: {restore_error}")
                                raise
                            finally:
                                write_json(output / "status.json", status)
                    results[sign] = attempt(f"fd_{label}_{step}_{sign}", fd_call)
                    row[sign] = results[sign]
                    write_json(output / "status.json", status)
                row.update(status="complete" if all(results.values()) else "failed")
                if all(results.values()):
                    difference = (results["plus"]["objective"] - results["minus"]["objective"]) / (2 * epsilon)
                    row.update(helpers.derivative_diagnostic(adjoint_value, difference, np.linalg.norm(production), np.linalg.norm(direction)))
                derivatives.append(row)
                write_json(output / "status.json", status)
        status["gradient_numerical_gate"] = all(any(row["direction"] == label and row["relative_step"] == .0003
            and row["informative"] and row["relative_error"] <= .001 for row in derivatives) for label in ["interior", "edge"])
        comparisons = {}
        pairs = protocol["forward_comparisons"]["boundary"] + [protocol["forward_comparisons"]["space"], protocol["forward_comparisons"]["time"]]
        for a, b in pairs:
            if a in base_records and b in base_records:
                comparisons[a + "__" + b] = helpers.waveform_comparison(base_records[a]["arrays"], base_records[b]["arrays"])
        status["forward_comparisons"] = comparisons
        status["source_after"] = source_record(runtime, repo, input_path, protocol_path)
        if status["source_after"] != before:
            raise RuntimeError("input/source changed during case")
        status.update(status="complete_technical_diagnostics", maxrss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      no_physical_acceptance_claim=True, one_ancestor=True)
        write_json(output / "status.json", status)
        write_json(output / "complete_receipt.json", status)
        return 0
    except BaseException as exc:
        status.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        try:
            status["source_after"] = source_record(runtime, repo, input_path, protocol_path)
            status["source_unchanged"] = status["source_after"] == before
        except Exception as pin_error:
            status["source_after_error"] = f"{type(pin_error).__name__}: {pin_error}"
        write_json(output / "status.json", status)
        raise


def run_child(command):
    """One isolated worker; no surviving child after an interrupt to the parent."""
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait()
    except BaseException:
        # Repeated user SIGINT must not interrupt the kill/reap cleanup itself.
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except BaseException:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
        raise


def merge_child_status(status, child, case_id):
    if (child / "status.json").exists():
        record = json.loads((child / "status.json").read_bytes())
        for updated in record["calls"]:
            next(row for row in status["calls"] if row["case_id"] == case_id and row["call_id"] == updated["call_id"]).update(updated)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--execute-guards", action="store_true")
    parser.add_argument("--worker-case")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-protocol-sha256")
    args = parser.parse_args(argv)
    protocol, raw = load_protocol(args.protocol)
    if not args.execute_guards:
        print(json.dumps(dict(status="metadata_only_no_BP_or_PDE", planned_wave_calls=len(plan(protocol)),
            protocol_sha256=sha(raw), source_sha256=sha(Path(__file__).read_bytes()))))
        return 0
    if not all([args.repo, args.inputs, args.output, args.expected_source_sha256, args.expected_protocol_sha256]):
        parser.error("explicit run requires repo/input/newoutput paths and exact hashes")
    if sha(raw) != args.expected_protocol_sha256 or sha(Path(__file__).read_bytes()) != args.expected_source_sha256:
        raise ValueError("reviewed source/protocol SHA mismatch")
    disk_requirement = protocol["memory"]["per_case_disk_reserve_bytes" if args.worker_case else "minimum_free_disk_bytes"]
    if args.output.exists() or shutil.disk_usage(args.output.parent).free < disk_requirement:
        raise OSError("new output and the fixed initial/per-case disk reserve are required")
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if os.environ.get(key) != "1":
            raise ValueError("fixed single-thread environment required")
    if args.worker_case:
        return worker(args.repo.resolve(), args.inputs.resolve(), args.output.resolve(), args.protocol.resolve(), args.worker_case)
    args.output.mkdir()
    status = dict(status="running", calls=plan(protocol), physical_acceptance=None,
        case_processes=[dict(case_id=case["id"], status="unattempted", output=str(args.output / case["id"]))
                        for case in protocol["inputs"]],
        source_sha256=sha(Path(__file__).read_bytes()), protocol_sha256=sha(raw))
    write_json(args.output / "status.json", status)
    try:
        for case, entry in zip(protocol["inputs"], status["case_processes"], strict=True):
            if shutil.disk_usage(args.output).free < protocol["memory"]["per_case_disk_reserve_bytes"]:
                raise OSError("per-case disk reserve exhausted; remaining cases unattempted")
            child = args.output / case["id"]
            entry["status"] = "attempted"
            write_json(args.output / "status.json", status)
            command = [sys.executable, str(Path(__file__).resolve()), "--execute-guards", "--worker-case", case["id"],
                "--protocol", str(args.protocol.resolve()), "--repo", str(args.repo.resolve()), "--inputs", str(args.inputs.resolve()),
                "--output", str(child.resolve()), "--expected-source-sha256", args.expected_source_sha256,
                "--expected-protocol-sha256", args.expected_protocol_sha256]
            exit_code = run_child(command)
            entry.update(status="complete" if exit_code == 0 else "failed", exit_code=exit_code)
            merge_child_status(status, child, case["id"])
            write_json(args.output / "status.json", status)
    except BaseException as exc:
        if entry["status"] == "attempted":
            entry.update(status="interrupted", in_flight_wave_work_unknown=True)
            merge_child_status(status, Path(entry["output"]), entry["case_id"])
        status.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        write_json(args.output / "status.json", status)
        raise
    status.update(status="complete_guard_attempts", all_case_workers_complete=all(r["exit_code"] == 0 for r in status["case_processes"]))
    write_json(args.output / "status.json", status)
    write_json(args.output / "complete_receipt.json", status)
    return 0 if status["all_case_workers_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
