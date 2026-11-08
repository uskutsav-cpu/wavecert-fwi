"""Causal 2D Green comparison; default CLI only checks metadata/source hashes."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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


def save_npz(path, **arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def gauss_legendre(order):
    """Golub-Welsch with Legendre orthonormal recurrence, DLMF3.5.31-32."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("positive integer quadrature order required")
    k = np.arange(1, order, dtype=np.float64)
    off = k / np.sqrt(4 * k * k - 1)
    matrix = np.diag(off, 1) + np.diag(off, -1) if order > 1 else np.zeros((1, 1))
    nodes, vectors = np.linalg.eigh(matrix)
    return nodes, 2 * vectors[0] ** 2


def ricker_causal(times, frequency, delay):
    times = np.asarray(times, dtype=np.float64)
    argument = np.pi * frequency * (times - delay)
    values = (1 - 2 * argument ** 2) * np.exp(-argument ** 2)
    return np.where(times >= 0, values, 0.)


def convolve_green(times, distance, velocity, strength, forcing, nodes, weights):
    """A/(2pi) integral_0^acosh(t/a) q(t-a cosh(v))dv; no fitted scale."""
    if not all(np.isfinite(v) and v > 0 for v in [distance, velocity, strength]):
        raise ValueError("positive finite distance, velocity and source strength required")
    times = np.asarray(times, dtype=np.float64)
    result = np.zeros_like(times)
    travel = distance / velocity
    for index in np.flatnonzero(times > travel):
        upper = np.arccosh(times[index] / travel)
        v = upper * (nodes + 1) / 2
        result[index] = strength / (2 * np.pi) * upper / 2 * np.dot(weights, forcing(times[index] - travel * np.cosh(v)))
    return result


def step_closed_form(times, distance, velocity, strength):
    times = np.asarray(times, dtype=np.float64)
    result = np.zeros_like(times)
    travel = distance / velocity
    mask = times > travel
    result[mask] = strength / (2 * np.pi) * np.arccosh(times[mask] / travel)
    return result


def comparison_mask(times, lower, upper):
    indices = np.arange(len(times))
    return (indices >= lower) & (indices <= upper)


def norm_ratio(error, reference):
    numerator, denominator = float(np.linalg.norm(error)), float(np.linalg.norm(reference))
    return dict(numerator=numerator, denominator=denominator,
                value=numerator / denominator if denominator else None, zero_denominator=denominator == 0)


def trace_metrics(discrete, analytic, times, valid, cutoff):
    if not (discrete.shape == analytic.shape == times.shape == valid.shape):
        raise ValueError("aligned one-dimensional inputs required")
    early, late = valid & (times <= cutoff), valid & (times > cutoff)
    error = discrete - analytic
    return dict(full_relative=norm_ratio(error[valid], analytic[valid]),
                early_relative=norm_ratio(error[early], analytic[early]),
                late_mismatch_over_full_analytic=norm_ratio(error[late], analytic[valid]),
                sample_counts=dict(full=int(valid.sum()), early=int(early.sum()), late=int(late.sum())),
                physical_quality_pass=None)


def load_protocol(path):
    raw = path.read_bytes()
    protocol = json.loads(raw)
    for name, expected in protocol["input_pins"].items():
        if sha((path.parent / name).read_bytes()) != expected:
            raise ValueError("pinned primary input changed: " + name)
    original = json.loads((path.parent / "original_boundary_protocol.json").read_bytes())
    metadata = json.loads((path.parent / "primary_metadata" / "operator_time_arguments.json").read_bytes())
    if [r["mode"] for r in metadata["records"]] != ["undamped", "damped"] or any(
        r["scalar_arguments"]["time_m"] != 1 or r["scalar_arguments"]["time_M"] != 1099
        or r["scalar_arguments"]["dt"] != 1 or not r["byte_identical_to_cached_primary"] for r in metadata["records"]
    ):
        raise ValueError("pinned receiver time semantics disagree")
    common = original["common"]
    if ([c["id"] for c in original["cases"]] != protocol["case_ids"] or protocol["quadrature"]["orders"] != [128, 256, 512]
            or not common["velocity_km_per_s"] == protocol["velocity_m_per_ms"] == 2
            or common["spacing_m"] != [10., 10.] or protocol["cell_area_point_strength_m2"] != 100
            or common["f0_per_ms"] != protocol["source"]["frequency_per_ms"]
            or protocol["source"]["delay_ms"] != 1 / common["f0_per_ms"]
            or protocol["valid_comparison_indices_inclusive"] != [1, 1099]):
        raise ValueError("fixed source, geometry or comparison fixture changed")
    distance = float(np.linalg.norm(np.array(common["receiver_base_x_z_m"]) - common["source_base_x_z_m"]))
    if distance != protocol["source_receiver_distance_m"] or distance != 50:
        raise ValueError("fixed receiver distance changed")
    return protocol, raw


def bundle_snapshot(directory):
    return {str(p.relative_to(directory)): sha(p.read_bytes()) for p in sorted(directory.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}


def old_snapshot(directory, protocol):
    return {name: sha((directory / name).read_bytes())
            for name in {**protocol["old_run_metadata_pins"], **protocol["old_raw_pins"]}}


def analytic_orders(protocol, output, status, persist):
    config = protocol["time_ms"]
    times = np.arange(config["samples"], dtype=np.float64) * config["step"] + config["start"]
    if times[-1] != config["stop"]:
        raise ValueError("time-axis definition mismatch")
    valid = comparison_mask(times, *protocol["valid_comparison_indices_inclusive"])
    source = protocol["source"]
    def forcing(t):
        return ricker_causal(t, source["frequency_per_ms"], source["delay_ms"])
    parameters = [protocol[k] for k in ["source_receiver_distance_m", "velocity_m_per_ms", "cell_area_point_strength_m2"]]
    step = step_closed_form(times, *parameters)
    curves, numeric_flags = {}, []
    qconfig = protocol["quadrature"]
    for order in qconfig["orders"]:
        entry = next(r for r in status["orders"] if r["order"] == order)
        entry.update(status="attempted", started_utc=utc())
        persist()
        nodes, weights = gauss_legendre(order)
        moments = [float(abs(np.dot(weights, nodes ** k) - (2 / (k + 1) if k % 2 == 0 else 0))) for k in range(7)]
        node_ok = bool(np.isfinite(nodes).all() and np.isfinite(weights).all() and (weights > 0).all()
                       and (np.abs(nodes) < 1).all() and (np.diff(nodes) > 0).all()
                       and max(moments) <= qconfig["node_weight_and_moment_absolute_tolerance"])
        curve = convolve_green(times, *parameters, forcing, nodes, weights)
        step_quad = convolve_green(times, *parameters, np.ones_like, nodes, weights)
        step_error = float(np.max(np.abs(step_quad - step)) / (1 + np.max(np.abs(step))))
        finite = bool(np.isfinite(curve).all() and np.isfinite(step_quad).all())
        step_ok = step_error <= qconfig["step_control_normalized_max_abs_tolerance"]
        filename = "analytic_order_" + str(order) + ".npz"
        save_npz(output / filename, times_ms=times, source=forcing(times), valid_mask=valid, analytic=curve,
                 step_exact=step, step_quadrature=step_quad, quadrature_nodes=nodes, quadrature_weights=weights)
        entry.update(status="complete", finished_utc=utc(), artifact=filename, artifact_sha256=sha((output / filename).read_bytes()),
                     moment_absolute_errors=moments, node_weight_gate=node_ok, step_control_error=step_error,
                     step_control_gate=step_ok, finite=finite)
        numeric_flags.append(node_ok and step_ok and finite)
        curves[order] = curve
        persist()
    convergence = []
    for lower, upper in qconfig["convergence_pairs"]:
        values = norm_ratio((curves[lower] - curves[upper])[valid], curves[upper][valid])
        values.update(lower_order=lower, upper_order=upper,
                      gate=values["value"] is not None and values["value"] <= qconfig["convergence_relative_L2_maximum"])
        convergence.append(values)
        numeric_flags.append(values["gate"])
    status.update(quadrature_convergence=convergence, internal_numeric_gates_pass=all(numeric_flags),
                  all_three_analytic_orders_saved=True)
    persist()
    return times, valid, forcing(times), curves[qconfig["comparison_order"]]


def run(protocol_path, original, output, expected_source, expected_protocol):
    protocol, raw = load_protocol(protocol_path)
    bundle = protocol_path.parent.resolve()
    if sha(raw) != expected_protocol or sha(Path(__file__).read_bytes()) != expected_source:
        raise ValueError("explicit source/protocol hash gate failed")
    original, output = original.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(original) or output.is_relative_to(bundle):
        raise ValueError("new output must be outside immutable input/bundle")
    if shutil.disk_usage(output.parent).free < protocol["execution"]["minimum_free_disk_bytes"]:
        raise OSError("minimum100MiB free disk required")
    before = bundle_snapshot(bundle)
    old_before = old_snapshot(original, protocol)
    if old_before != {**protocol["old_run_metadata_pins"], **protocol["old_raw_pins"]}:
        raise ValueError("original seven-case input hash mismatch")
    output.mkdir()
    status = dict(status="running", started_utc=utc(), source_before=before, original_before=old_before,
                  orders=[dict(order=order, status="unattempted") for order in protocol["quadrature"]["orders"]],
                  cases=[dict(case_id=case, status="unattempted") for case in protocol["case_ids"]],
                  all_three_analytic_orders_saved=False, comparison_started=False,
                  original_numeric_array_access="not_started", student_design=False, physical_quality_pass=None)
    def persist():
        write_json(output / "status.json", status)
    persist()
    current = None
    try:
        times, valid, own_source, analytic = analytic_orders(protocol, output, status, persist)
        if not status["all_three_analytic_orders_saved"]:
            raise RuntimeError("all three own orders must be saved before old numerical array access")
        status.update(comparison_started=True, original_numeric_array_access="started_or_unknown")
        persist()
        software_flags = []
        for case in protocol["case_ids"]:
            current = next(r for r in status["cases"] if r["case_id"] == case)
            current.update(status="attempted", started_utc=utc())
            persist()
            try:
                with np.load(original / (case + "_raw.npz"), allow_pickle=False) as saved:
                    receiver = saved["receiver"].copy()
                    stored_source = saved["source_before"].copy()
                    sample_checks = [np.array_equal(saved[k], times) for k in ["times_before_ms", "times_after_ms", "source_times_before_ms", "source_times_after_ms", "receiver_times_ms"]]
                    sample_checks.append(np.array_equal(stored_source, saved["source_after"]))
                    geometry_ok = bool(np.linalg.norm(saved["receiver_coordinates_m"] - saved["source_coordinates_before_m"]) == 50
                                       and np.array_equal(saved["source_coordinates_before_m"], saved["source_coordinates_after_m"]))
                if receiver.shape != (len(times), 1) or stored_source.shape != (len(times), 1):
                    raise ValueError("original shape mismatch")
                excitation = norm_ratio(stored_source[:, 0] - own_source, own_source)
                gate = bool(all(sample_checks) and geometry_ok and np.isfinite(receiver).all()
                            and receiver[0, 0] == receiver[-1, 0] == 0 and excitation["value"] is not None
                            and excitation["value"] <= protocol["source_sample_matching_relative_L2_maximum"])
                values = trace_metrics(receiver[:, 0], analytic, times, valid, protocol["windows"]["early_upper_ms_inclusive"])
                filename = case + "_comparison.npz"
                save_npz(output / filename, times_ms=times, receiver=receiver, source=stored_source, valid_mask=valid,
                         analytic512=analytic, difference=receiver[:, 0] - analytic)
                current.update(status="complete", finished_utc=utc(), metrics=values, source_relative_L2=excitation,
                               source_bitwise_equal=bool(np.array_equal(stored_source[:, 0], own_source)),
                               exact_time_and_source_unchanged_checks=sample_checks, geometry_matches=geometry_ok,
                               input_software_gate=gate, primary_case=case == protocol["primary_case"],
                               artifact=filename, artifact_sha256=sha((output / filename).read_bytes()))
                software_flags.append(gate)
            except Exception as exc:
                current.update(status="failed", error=type(exc).__name__ + ": " + str(exc), finished_utc=utc())
                software_flags.append(False)
            persist()
        status.update(original_numeric_array_access="all_seven_attempted", all_input_software_gates_pass=all(software_flags),
                      source_after=bundle_snapshot(bundle), original_after=old_snapshot(original, protocol))
        unchanged = status["source_after"] == before and status["original_after"] == old_before
        status.update(source_and_original_unchanged=unchanged, status="complete_analytical_diagnostic", finished_utc=utc())
        persist()
        write_json(output / "complete_receipt.json", dict(status))
        return bool(unchanged and all(software_flags) and status["internal_numeric_gates_pass"])
    except BaseException as exc:
        if current is not None and current["status"] == "attempted":
            current.update(status="failed" if isinstance(exc, Exception) else "interrupted",
                           error=type(exc).__name__ + ": " + str(exc), finished_utc=utc())
        for entry in status["orders"]:
            if entry["status"] == "attempted":
                entry.update(status="failed" if isinstance(exc, Exception) else "interrupted",
                             error=type(exc).__name__ + ": " + str(exc), finished_utc=utc())
        status.update(status="failed", error=type(exc).__name__ + ": " + str(exc), finished_utc=utc(),
                      source_after=bundle_snapshot(bundle), original_after=old_snapshot(original, protocol))
        persist()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--execute-analytical-comparison", action="store_true")
    parser.add_argument("--original-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-protocol-sha256")
    args = parser.parse_args(argv)
    _, raw = load_protocol(args.protocol)
    if not args.execute_analytical_comparison:
        print(json.dumps(dict(status="metadata_only_no_trace_evaluation", source_sha256=sha(Path(__file__).read_bytes()),
                              protocol_sha256=sha(raw), python=sys.version, numpy=np.__version__, platform=platform.platform())))
        return 0
    if not all([args.original_run, args.output, args.expected_source_sha256, args.expected_protocol_sha256]):
        parser.error("explicit run requires original-run, new output and both approved hashes")
    passed = run(args.protocol, args.original_run, args.output, args.expected_source_sha256, args.expected_protocol_sha256)
    print(json.dumps(dict(status="complete_analytical_diagnostic", internal_numerical_and_input_gates_pass=passed, physical_quality_pass=None)))
    return 0 if passed else 1


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
