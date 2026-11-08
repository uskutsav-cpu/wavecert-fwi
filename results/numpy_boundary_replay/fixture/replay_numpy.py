"""Reference-informed NumPy verification; default CLI performs no propagation.

Only Python's standard library and NumPy participate in propagation. Original
receiver/source arrays are opened only after all fourteen own solves finish.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _json(value):
    return (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()


def _exclusive(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def _atomic_owned(path, value):
    temporary = path.with_name(path.name+".tmp")
    try:
        _exclusive(temporary, _json(value))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rational_weights():
    """Derive a five-node second derivative from exact polynomial moments."""
    offsets = [-2, -1, 0, 1, 2]
    rows = [[Fraction(j)**power for j in offsets]+[Fraction(2 if power == 2 else 0)]
            for power in range(5)]
    for column in range(5):
        pivot = next(i for i in range(column, 5) if rows[i][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [v/divisor for v in rows[column]]
        for i in range(5):
            if i != column:
                multiplier = rows[i][column]
                rows[i] = [a-multiplier*b for a, b in zip(rows[i], rows[column], strict=True)]
    return tuple(row[-1] for row in rows)


def coefficients(variant, spacing, protocol):
    if variant == "faithful_literals":
        if spacing != 10.:
            raise ValueError("generated literals are pinned to spacing10m")
        values = protocol["faithful_spatial_coefficients_h10"]
        return tuple(float(values[k]) for k in ["positive_far", "positive_near", "positive_two_dimensional_center"])
    if variant != "rational_spatial":
        raise ValueError("unknown fixed replay variant")
    weights = rational_weights()
    h2 = Fraction(str(spacing))**2
    return float(-weights[0]/h2), float(weights[1]/h2), float(-2*weights[2]/h2)


def damping(shape, width, spacing):
    """Independent analytic profile from the pinned face-index definitions."""
    result = np.zeros(shape, dtype=np.float64)
    if width == 0:
        return result
    for axis, (n, h) in enumerate(zip(shape, spacing, strict=True)):
        profile = np.zeros(n, dtype=np.float64)
        indices = np.arange(width)
        left = (width-indices+1)/float(width)
        right = (indices+2)/float(width)
        coefficient = 1.5*np.log(1000.)/(width*h)
        profile[:width] = coefficient*(left-np.sin(2*np.pi*left)/(2*np.pi))
        profile[-width:] = coefficient*(right-np.sin(2*np.pi*right)/(2*np.pi))
        result += profile[:, None] if axis == 0 else profile[None, :]
    return result


def geometry(case, common):
    h = np.asarray(common["spacing_m"], dtype=np.float64)
    translation = np.asarray(case["translation_m"], dtype=np.float64)
    origin = -case["nbl"]*h
    source = np.asarray(common["source_base_x_z_m"])+translation
    receiver = np.asarray(common["receiver_base_x_z_m"])+translation
    positions = [(source-origin)/h, (receiver-origin)/h]
    if any(not np.array_equal(p, np.round(p)) for p in positions):
        raise ValueError("fixture interpolation must reduce to aligned point nodes")
    shape = tuple(int(n+2*case["nbl"]) for n in case["shape_x_z"])
    indices = [tuple(int(i) for i in p) for p in positions]
    if any(any(i < 0 or i >= n for i, n in zip(index, shape, strict=True)) for index in indices):
        raise ValueError("source or receiver outside the logical grid")
    return shape, indices[0], indices[1], dict(origin_m=origin.tolist(),
        source_coordinates_m=source.tolist(), receiver_coordinates_m=receiver.tolist(),
        computational_source_index=list(indices[0]), computational_receiver_index=list(indices[1]))


def ricker(times, frequency):
    argument = np.pi*frequency*(times-1/frequency)
    return (1-2.*argument**2)*np.exp(-argument**2)


def one_step(state, time, damp, mass, timestep, weights, source_value, source_index):
    """Actual OT2 recurrence, forward damping derivative, zero exterior ghosts.

    Update next, add the node source, and leave current available to sample.
    The two-cell halos are never updated or forced onto logical outer nodes.
    """
    current, previous, following = time % 3, (time+2) % 3, (time+1) % 3
    u = state[current]
    center = u[2:-2, 2:-2]
    far, near, central = weights
    laplace = (far*(-u[:-4, 2:-2]-u[2:-2, :-4]-u[2:-2, 4:]-u[4:, 2:-2])
               +near*(u[1:-3, 2:-2]+u[2:-2, 1:-3]+u[2:-2, 3:-1]+u[3:-1, 2:-2])
               -central*center)
    inverse_dt2, inverse_dt = 1/timestep**2, 1/timestep
    state[following, 2:-2, 2:-2] = (
        -mass*(-2*inverse_dt2*center+inverse_dt2*state[previous, 2:-2, 2:-2])
        +inverse_dt*damp*center+laplace)/(mass*inverse_dt2+inverse_dt*damp)
    source_position = tuple(i+2 for i in source_index)
    state[(following, *source_position)] += timestep**2/mass*source_value
    return current, following


def solve_case(case, common, variant, protocol):
    shape, source, receiver, positions = geometry(case, common)
    timestep = common["dt_ms"]
    times = np.arange(common["t0_ms"], common["tmax_ms"]+timestep/2, timestep, dtype=np.float64)
    source_values = ricker(times, common["f0_per_ms"])
    damp = damping(shape, case["nbl"], common["spacing_m"])
    mass = 1/common["velocity_km_per_s"]**2
    if damp[source] != 0:
        raise ValueError("source injection specialization requires the pinned undamped source node")
    state = np.zeros((3, shape[0]+4, shape[1]+4), dtype=np.float64)
    receiver_values = np.zeros((len(times), 1), dtype=np.float64)
    weights = coefficients(variant, common["spacing_m"][0], protocol)
    minimum, maximum = protocol["time_bounds"]["time_m"], protocol["time_bounds"]["time_M"]
    for time in range(minimum, maximum+1):
        current, _ = one_step(state, time, damp, mass, timestep, weights, source_values[time], source)
        receiver_values[time, 0] = state[(current, receiver[0]+2, receiver[1]+2)]
    finite = all(np.all(np.isfinite(a)) for a in [state, damp, receiver_values, source_values])
    if not finite:
        raise FloatingPointError("nonfinite independent NumPy state or traces")
    arrays = dict(times_ms=times, source=source_values[:, None], receiver=receiver_values, damping=damp)
    record = dict(case=case, variant=variant, geometry=positions,
        spatial_coefficients=dict(far_positive=weights[0], near_positive=weights[1], center_positive=weights[2]),
        mass=mass, time_bounds=dict(protocol["time_bounds"]),
        state_shape=list(state.shape), state_bytes=state.nbytes, logical_grid_shape=list(shape),
        halo_width_cells=2, full_time_history_saved=False,
        final_cyclic_state_sha256=sha(state.tobytes()), damping_sha256=sha(damp.tobytes()),
        damping_min=float(damp.min()), damping_max=float(damp.max()), finite=finite,
        original_receiver_or_source_arrays_used=False)
    return arrays, record


def load_protocol(path, *, require_frozen=True):
    path = Path(path)
    raw = path.read_bytes()
    p = json.loads(raw)
    original_raw = (path.parent/"original_protocol.json").read_bytes()
    if sha(original_raw) != p["original_protocol_sha256"]:
        raise ValueError("original fixed protocol changed")
    original = json.loads(original_raw)
    if p["variants"] != ["faithful_literals", "rational_spatial"] or p["replays"] != 14 or len(original["cases"]) != 7:
        raise ValueError("fixed two-variant/seven-case scope changed")
    if (p["agreement_relative_L2_tolerance"] != 1e-9
            or p["matching_excitation_relative_L2_tolerance"] != 1e-14
            or p["student_final_research_design"] is not False):
        raise ValueError("agreement/scope contract changed")
    if require_frozen and (p["time_bounds"] is None or p.get("argument_evidence_sha256") is None):
        raise ValueError("primary argument/time-bound evidence is not yet frozen")
    if p.get("argument_evidence_sha256") is not None:
        evidence = path.parent/"primary_metadata/operator_time_arguments.json"
        query = path.parent/"primary_metadata/query_time_arguments.py"
        if sha(evidence.read_bytes()) != p["argument_evidence_sha256"] or sha(query.read_bytes()) != p["argument_query_source_sha256"]:
            raise ValueError("primary argument query/evidence changed")
        metadata = json.loads(evidence.read_text())
        if (p["time_bounds"]["time_m"] != 1 or p["time_bounds"]["time_M"] != 1099
                or p["time_bounds"]["nt"] != 1101
                or len(metadata["records"]) != 2
                or {r["mode"] for r in metadata["records"]} != {"damped", "undamped"}
                or any(r["scalar_arguments"]["time_m"] != 1 or r["scalar_arguments"]["time_M"] != 1099
                    or r["byte_identical_to_cached_primary"] is not True
                    or r["generated_code_sha256"] != r["cached_primary_code_sha256"]
                    or r["generated_code_sha256"] not in p["primary_generated_C_sha256"].values()
                    for r in metadata["records"])):
            raise ValueError("replay bounds differ from primary metadata")
    for name, expected in p["primary_generated_C_sha256"].items():
        if sha((path.parent/"primary"/name).read_bytes()) != expected:
            raise ValueError("generated primary C snapshot changed")
    for name, expected in p["copied_primary_python_sha256"].items():
        if sha((path.parent/name).read_bytes()) != expected:
            raise ValueError("copied primary Python source changed")
    return p, raw, original, original_raw


def _production_digest(repo):
    files = [repo/"pyproject.toml"]
    files += list((repo/"src").rglob("*.py"))
    files += [p for p in (repo/"configs").rglob("*") if p.is_file()]
    files += [p for p in (repo/"scripts").rglob("*") if p.suffix in {".py", ".sh"}]
    files += list((repo/".github/workflows").glob("*.yml"))
    files += list(repo.glob("requirements*.txt"))
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        if path.is_file():
            digest.update(path.relative_to(repo).as_posix().encode()+b"\0")
            digest.update(sha(path.read_bytes()).encode()+b"\0")
    return digest.hexdigest()


def source_receipt(repo, protocol, base):
    digest = _production_digest(Path(repo).resolve())
    if digest != protocol["production_source_sha256"]:
        raise ValueError("native production source changed")
    copied = {}
    for relative, checksum in protocol["copied_primary_python_sha256"].items():
        actual = sha((base/relative).read_bytes())
        if actual != checksum:
            raise ValueError("copied primary example source changed")
        copied[relative] = actual
    local = {str(p.relative_to(base)): sha(p.read_bytes()) for p in sorted(base.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".json", ".c"}}
    return dict(production_source_sha256=digest, copied_primary_sha256=copied,
                local_input_source_sha256=local)


def compare_after_all_solves(output, reference_run, records, protocol):
    """This is the first function allowed to open any original numeric array."""
    receipt_path = reference_run/"complete_receipt.json"
    if sha(receipt_path.read_bytes()) != protocol["original_complete_receipt_sha256"]:
        raise ValueError("original completed-run identity changed")
    receipt = json.loads(receipt_path.read_text())
    references = {r["case_id"]: r for r in receipt["case_records"]}
    outcomes = []
    for row in records:
        reference = references[row["case_id"]]
        path = reference_run/reference["raw_artifact"]
        if sha(path.read_bytes()) != reference["raw_sha256"]:
            raise ValueError("original trace artifact hash mismatch")
        with np.load(path, allow_pickle=False) as values:
            actual, times, source = values["receiver"], values["times_before_ms"], values["source_before"]
        with np.load(output/row["raw_artifact"], allow_pickle=False) as own:
            if own["receiver"].shape != actual.shape or own["source"].shape != source.shape:
                raise ValueError("own/reference receiver or source shapes differ; alignment changes forbidden")
            same_time = np.array_equal(own["times_ms"], times)
            same_source = np.array_equal(own["source"], source)
            numerator = float(np.linalg.norm(own["receiver"]-actual))
            denominator = float(np.linalg.norm(actual))
            source_numerator = float(np.linalg.norm(own["source"]-source))
            source_denominator = float(np.linalg.norm(source))
            source_max_absolute_difference = float(np.max(np.abs(own["source"]-source)))
        value = None if denominator == 0 else numerator/denominator
        source_value = None if source_denominator == 0 else source_numerator/source_denominator
        matching_source = bool(source_value is not None and np.isfinite(source_value)
            and source_value <= protocol["matching_excitation_relative_L2_tolerance"])
        outcomes.append(dict(case_id=row["case_id"], variant=row["variant"],
            numerator=numerator, denominator=denominator, relative_L2=value,
            source_samples_bitwise_equal=same_source, time_samples_bitwise_equal=same_time,
            source_L2_numerator=source_numerator, source_L2_denominator=source_denominator,
            source_relative_L2=source_value,
            source_max_absolute_difference=source_max_absolute_difference,
            matching_excitation_tolerance=protocol["matching_excitation_relative_L2_tolerance"],
            matching_excitation=matching_source, bitwise_source_flag_is_diagnostic=True,
            tolerance=protocol["agreement_relative_L2_tolerance"],
            agreement=bool(same_time and matching_source and value is not None and np.isfinite(value)
                           and value <= protocol["agreement_relative_L2_tolerance"])))
    return outcomes


def run(protocol_path, repo, reference_run, output, expected_protocol_sha256, expected_source_sha256):
    protocol_path = Path(protocol_path).resolve()
    base = protocol_path.parent
    protocol, protocol_raw, original, original_raw = load_protocol(protocol_path)
    source_raw = Path(__file__).read_bytes()
    if sha(protocol_raw) != expected_protocol_sha256 or sha(source_raw) != expected_source_sha256:
        raise ValueError("reviewed runner/protocol SHA256 mismatch before propagation")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("existing output is never overwritten or reused")
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < 100*1024**2:
        raise OSError("existing parent with100MiB free is required")
    before = source_receipt(repo, protocol, base)
    output.mkdir()
    status = dict(status="running", started_utc=utc(), source_before=before,
        source_sha256=sha(source_raw), protocol_sha256=sha(protocol_raw), records=[],
        reference_numeric_arrays_opened=False, comparison_started=False,
        propagation_dependencies=["stdlib", "NumPy"],
        student_final_research_design=False, goal4_completion_claim=False)
    _exclusive(output/"runner_source.py", source_raw)
    _exclusive(output/"protocol.json", protocol_raw)
    _exclusive(output/"original_protocol.json", original_raw)
    _atomic_owned(output/"status.json", status)
    try:
        for variant in protocol["variants"]:
            for case in original["cases"]:
                entry = dict(variant=variant, case_id=case["id"], status="attempted", started_utc=utc())
                status["records"].append(entry)
                _atomic_owned(output/"status.json", status)
                try:
                    arrays, record = solve_case(case, original["common"], variant, protocol)
                    filename = variant+"_"+case["id"]+"_raw.npz"
                    with (output/filename).open("xb") as stream:
                        np.savez_compressed(stream, **arrays)
                    entry.update(status="complete", record=record, raw_artifact=filename,
                                 raw_sha256=sha((output/filename).read_bytes()))
                except Exception as exc:
                    entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                except BaseException as exc:
                    entry.update(status="interrupted", error=f"{type(exc).__name__}: {exc}", finished_utc=utc())
                    _atomic_owned(output/"status.json", status)
                    raise
                entry["finished_utc"] = utc()
                _atomic_owned(output/"status.json", status)
        after = source_receipt(repo, protocol, base)
        if (before != after or protocol_path.read_bytes() != protocol_raw
                or Path(__file__).read_bytes() != source_raw):
            raise ValueError("sources changed during independent propagation")
        if len(status["records"]) != 14 or any(r["status"] != "complete" for r in status["records"]):
            raise ValueError("not all fourteen independent solves completed")
        status.update(all_own_solves_complete_utc=utc(), source_after=after,
                      reference_numeric_arrays_opened=False)
        _atomic_owned(output/"status.json", status)
        status.update(comparison_started=True, reference_numeric_arrays_opened=None)
        _atomic_owned(output/"status.json", status)
        outcomes = compare_after_all_solves(output, Path(reference_run).resolve(), status["records"], protocol)
        _exclusive(output/"agreement.json", _json(outcomes))
        status.update(status="complete_independent_software_replay", finished_utc=utc(),
            reference_numeric_arrays_opened=True, all_errors_retained=True,
            agreement_by_variant={v: sum(r["agreement"] for r in outcomes if r["variant"] == v) for v in protocol["variants"]},
            environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                threads={k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]}))
        _atomic_owned(output/"status.json", status)
        complete = dict(status, artifacts={p.name: sha(p.read_bytes()) for p in output.iterdir() if p.is_file()})
        _atomic_owned(output/"complete_receipt.json", complete)
        return complete
    except BaseException as exc:
        status.update(status="failed", finished_utc=utc(), error=f"{type(exc).__name__}: {exc}")
        _atomic_owned(output/"status.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args(argv)
    _, raw, _, _ = load_protocol(args.protocol, require_frozen=args.run)
    if not args.run:
        print(json.dumps(dict(status="metadata_only_no_propagation", protocol_sha256=sha(raw), runner_sha256=sha(Path(__file__).read_bytes()))))
        return 0
    if not all((args.repo, args.reference_run, args.output, args.expected_protocol_sha256, args.expected_source_sha256)):
        parser.error("--run requires repository/reference/new-output paths and both reviewed hashes")
    receipt = run(args.protocol, args.repo, args.reference_run, args.output, args.expected_protocol_sha256, args.expected_source_sha256)
    print(json.dumps(dict(status=receipt["status"], agreement_by_variant=receipt["agreement_by_variant"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
