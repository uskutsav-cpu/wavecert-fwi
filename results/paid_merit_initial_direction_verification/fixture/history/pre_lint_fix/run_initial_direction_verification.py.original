"""Eight fixed initial-direction witnesses; default CLI performs no project reads.

No optimizer, method selection, timing comparison or unseen geology is involved.
This separate candidate must be reviewed/frozen and run only after root confirms
the live timed benchmark terminal. Production code9f and the807 auditor stay intact.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path

OBJECTIVE_SOURCE_SHA = "807f4f2f0e0aeea1d2f61b64ee5f64a99819c38d4f6164529e664f0a235edeae"
PRODUCTION_SHA = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
ORIGINAL_PROTOCOL_SHA = "fb6b304fd719441be8cd6eb8d101ca864d0856254f40618583d8d986fd1c945c"
CANDIDATE_FILES = {
    "__init__.py": "8f5dde1292081eedb460cc9752025c72350673a2bd705bd407a5e6fa531bf22b",
    "adapter.py": "f793f4439551bf7614bd72421057d209f3cb353710fe06f8c13de85b44eecf61",
    "verification.py": "b7bee25257caf456d52932962ceba2fde00197f0a78b585cf90c138e795bba4c",
    "protocol.json": ORIGINAL_PROTOCOL_SHA,
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def json_raw(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def new_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json_raw(value))


def status_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    new_json(temporary, value)
    temporary.replace(path)


def import_file(name, path, *, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load frozen source: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def array_record(values):
    values = np.asarray(values)
    canonical = np.ascontiguousarray(values, dtype=values.dtype.newbyteorder("<"))
    return {"shape": list(values.shape), "dtype": canonical.dtype.str,
            "sha256": sha(canonical.tobytes()), "raw_values_saved": True}


def scalar_dot(left, right):
    return math.fsum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def near(checks, actual, recorded, scale, protocol, context):
    allowance = protocol["numeric_checks"]["scalar_relative_allowance"] * max(
        abs(actual), abs(recorded), scale, protocol["numeric_checks"]["scale_floor"])
    checks.require(math.isfinite(actual) and math.isfinite(recorded)
                   and abs(actual - recorded) <= allowance, context,
                   {"independent": actual, "recorded": recorded, "allowance": allowance})


def independent_gradient(case, model, protocol, checks, arrays):
    """Direct sparse state and physical adjoint from the pinned second assembly."""
    gradient = np.zeros(2401, dtype=np.float64)
    phi_terms, rows = [], []
    for index, frequency in enumerate(case.frequencies):
        matrix = case.operator(model, frequency)
        lu = spla.splu(matrix)
        state = lu.solve(case.rhs)
        residual = state[case.receivers, :] - case.observations[frequency]
        adjoint_rhs = np.zeros_like(case.rhs)
        np.add.at(adjoint_rhs, np.asarray(case.receivers), residual)
        adjoint = lu.solve(adjoint_rhs, trans="H")
        omega = 2.0 * math.pi * float(frequency)
        gradient += omega ** 2 * np.real(np.sum(np.conjugate(adjoint) * state, axis=1))
        phi_terms.append(0.5 * math.fsum(float(v.real) ** 2 + float(v.imag) ** 2
                                         for v in residual.ravel()))
        errors = {}
        for name, defect, forcing in (
                ("state", matrix @ state - case.rhs, case.rhs),
                ("adjoint", matrix.conjugate().T @ adjoint - adjoint_rhs, adjoint_rhs)):
            relative = max(case.scalar_norm(defect[:, col]) / max(case.scalar_norm(forcing[:, col]), 1e-300)
                           for col in range(forcing.shape[1]))
            checks.require(math.isfinite(relative) and relative <= protocol["numeric_checks"]["linear_relative_residual_maximum"],
                           f"independent/frequency{frequency}/{name}_residual", {"relative": relative})
            errors[name + "_maximum_relative_residual"] = relative
        prefix = f"f{index}_independent_"
        arrays[prefix + "state"] = state
        arrays[prefix + "adjoint"] = adjoint
        arrays[prefix + "observed"] = case.observations[frequency]
        arrays[prefix + "residual"] = residual
        arrays[prefix + "adjoint_rhs"] = adjoint_rhs
        rows.append({"frequency_hz": frequency, **errors,
                     "factorizations": 1, "state_rhs": 6, "adjoint_rhs": 6})
    if not np.all(np.isfinite(gradient)):
        raise ValueError("independent physical adjoint gradient is nonfinite")
    return gradient, math.fsum(phi_terms), rows


def load_production(root):
    directory = root / "results/provisional_software_benchmark/matched_cost_pilot"
    for name, expected in CANDIDATE_FILES.items():
        if sha((directory / name).read_bytes()) != expected:
            raise ValueError("production candidate source mismatch: " + name)
    sys.path.insert(0, str(root / "src"))
    package_name = "_wavecert_initial_direction_replay_candidate"
    import_file(package_name, directory / "__init__.py", package=True)
    helpers = importlib.import_module(package_name + ".verification")
    adapter = importlib.import_module(package_name + ".adapter")
    certificate_module = importlib.import_module("wavecert.certificates.cached_merit_certificate")
    for module in (helpers, adapter):
        if Path(module.__file__).resolve().parent != directory:
            raise ValueError("foreign archived candidate import")
    for name, module in tuple(sys.modules.items()):
        if name == "wavecert" or name.startswith("wavecert."):
            filename = getattr(module, "__file__", None)
            if filename is not None and not Path(filename).resolve().is_relative_to(root / "src/wavecert"):
                raise ValueError("foreign wavecert import: " + name)
    if adapter.evaluate_cached_merit is not certificate_module.evaluate_cached_merit:
        raise ValueError("adapter certificate binding differs before observation wrapper")
    return helpers, adapter


def production_witness(helpers, adapter, prepared, original, arrays):
    """Call the actual adapter once; transparently retain its certificate inputs."""
    physics = helpers.Helmholtz2D(**{**original["physics"], "shape": (49, 49)})
    oracle = helpers.ExactMeritOracle(physics, prepared["blocks"], prepared["receivers"])
    proxy = adapter.CachedFieldProposal(oracle, physics, mode="fixed_sweeps_certified",
        rank=64, sweeps=4, gmres=original["gmres"])
    receipt = oracle.evaluate(prepared["initial"])
    oracle.accept(receipt)
    original_evaluate = adapter.evaluate_cached_merit
    captured = {}

    def capture(*args, **kwargs):
        if captured:
            raise RuntimeError("more than one certificate evaluation at fixed initial model")
        issuing_oracle, snapshot, missing, contexts = args
        certificate = original_evaluate(*args, **kwargs)
        captured.update(oracle=issuing_oracle, snapshot=snapshot, missing=missing,
                        contexts=contexts, certificate=certificate)
        return certificate

    adapter.evaluate_cached_merit = capture
    try:
        proposal = proxy.propose(prepared["initial"], receipt)
    finally:
        adapter.evaluate_cached_merit = original_evaluate
    if not captured:
        raise RuntimeError("production proposal returned no captured certificate call")
    snapshot, missing, certificate = (captured[name] for name in ("snapshot", "missing", "certificate"))
    frame_rows = []
    for index, frame in enumerate(snapshot.frequencies):
        if frame.cache_kind != "forward_states":
            raise ValueError("frozen six-source/twenty-receiver case unexpectedly changed merit route")
        prefix = f"f{index}_production_"
        for name in ("cached_field", "observed_slots", "residual_slots", "residual_unique", "receiver_expand_indices"):
            arrays[prefix + name] = np.asarray(getattr(frame, name)).copy()
        arrays[prefix + "missing_field"] = np.asarray(missing[frame.frequency_hz]).copy()
        frame_rows.append(frame.to_dict())
    arrays["production_damping"] = np.asarray(snapshot.damping).copy()
    if certificate.direction is not None:
        arrays["direction"] = np.asarray(certificate.direction).copy()
    if certificate.approximate_gradient is not None:
        arrays["approximate_gradient"] = np.asarray(certificate.approximate_gradient).copy()
    return {"certificate": certificate.to_dict(), "snapshot": snapshot.to_dict(),
            "frames": frame_rows, "production_initial_objective": receipt.objective,
            "proposal_flags": {"fallback_required": bool(proposal.fallback_required),
                "certified_descent": bool(proposal.certified_descent),
                "direction_available": proposal.direction is not None},
            "proposal_diagnostics": proposal.diagnostics,
            "observation_wrapper": "private adapter evaluate_cached_merit binding captures references and returns original object; restored in finally; source files unchanged; not timing evidence",
            "costs": {"oracle": oracle.cost_snapshot(), "proposal": proxy.cost_snapshot()},
            "unknown_fields_are_not_zero": True}


def run(root, benchmark, output, objective_path, protocol_path, expected_source, expected_protocol):
    global np, scipy, ndi, sp, spla
    source_raw, protocol_raw = Path(__file__).read_bytes(), protocol_path.read_bytes()
    if sha(source_raw) != expected_source or sha(protocol_raw) != expected_protocol:
        raise ValueError("reviewed runner/protocol SHA mismatch before project input reads")
    protocol = json.loads(protocol_raw)
    replayer_raw = Path(__file__).with_name("replay_raw_witnesses.py").read_bytes()
    if sha(replayer_raw) != protocol["raw_replayer_source_sha256"]:
        raise ValueError("reviewed raw replayer source differs before project input reads")
    if (protocol["schema_version"] != 1 or protocol["student_final_research_design"] is not False
            or protocol["case_indices"] != list(range(8)) or protocol["rank"] != 64
            or protocol["fixed_sweeps"] != 4 or protocol["optimizer_trajectories"] is not False):
        raise ValueError("only frozen eight-initial-model software witness scope is supported")
    root, benchmark, output, objective_path = [Path(p).resolve() for p in (root, benchmark, output, objective_path)]
    if output.exists() or output.is_relative_to(root) or output.is_relative_to(benchmark):
        raise FileExistsError("new outer output directory required")
    terminal = json.loads((benchmark / "status.json").read_bytes())
    if terminal.get("status") not in {"complete_engineering_benchmark", "failed_or_interrupted"}:
        raise RuntimeError("original timing benchmark is not terminal; no numeric imports or arrays")
    objective_raw = objective_path.read_bytes()
    if sha(objective_raw) != OBJECTIVE_SOURCE_SHA:
        raise ValueError("independent assembly source is not frozen807")
    import numpy as np
    import scipy
    import scipy.ndimage as ndi
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    independent = import_file("_wavecert_initial_direction_independent_807", objective_path)
    for name, value in (("np", np), ("scipy", scipy), ("ndi", ndi), ("sp", sp), ("spla", spla)):
        setattr(independent, name, value)
    source_before = independent.source_digest(root)
    if source_before["sha256"] != PRODUCTION_SHA:
        raise ValueError("production source is no longer frozen9f")
    original_path = root / "results/provisional_software_benchmark/matched_cost_pilot/protocol.json"
    original_raw = original_path.read_bytes()
    if sha(original_raw) != ORIGINAL_PROTOCOL_SHA:
        raise ValueError("original benchmark protocol changed")
    original = json.loads(original_raw)
    for key, value in original["threads"].items():
        if os.environ.get(key) != value:
            raise ValueError("fixed one-thread environment required: " + key)
    checks = independent.Checks()
    raw_cases = independent.load_cases(root, original, checks)
    helpers, adapter = load_production(root)
    output.mkdir(parents=False)
    for name, raw in (("runner_source.py", source_raw), ("protocol.json", protocol_raw),
                      ("independent_assembly_source.py", objective_raw), ("original_protocol.json", original_raw),
                      ("replay_raw_witnesses.py", replayer_raw)):
        with (output / name).open("xb") as stream:
            stream.write(raw)
    status = {"status": "running", "started_utc": utc(), "scope": protocol["scope"],
              "source_before": source_before, "runner_sha256": sha(source_raw),
              "protocol_sha256": sha(protocol_raw), "independent_assembly_sha256": OBJECTIVE_SOURCE_SHA,
              "all8_registered_before_case_solves": True,
              "cases": [{"case_index": i, "identity": original["identities"][i], "status": "unattempted"}
                        for i in range(8)],
              "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                          "platform": platform.platform(), "threads": original["threads"]},
              "student_final_research_design": False, "competition_evidence": False,
              "optimizer_or_performance_study": False}
    status_json(output / "status.json", status)
    try:
        for row in status["cases"]:
            index = row["case_index"]
            row.update(status="attempted", started_utc=utc())
            status_json(output / "status.json", status)
            arrays = {}
            interruption = None
            record = {"case_index": index, "identity": row["identity"], "status": "attempted",
                      "scope": protocol["scope"], "raw_arrays_available": False}
            try:
                prepared = helpers.prepare_case({"identity": row["identity"], "raw_model": raw_cases[index]}, original)
                case = independent.IndependentCase(raw_cases[index], original, checks, index)
                # Give the standalone gradient routine the independent scalar norm,
                # not any production helper or field solution.
                case.scalar_norm = independent.scalar_norm
                arrays["initial_model"] = prepared["initial"].copy()
                arrays["independent_truth_model"] = case.truth.copy()
                arrays["independent_damping"] = case.damping.copy()
                arrays["source_rhs"] = case.rhs.copy()
                arrays["source_indices"] = np.asarray(case.sources, dtype="<i8")
                arrays["receiver_indices"] = np.asarray(case.receivers, dtype="<i8")
                checks.require(np.allclose(prepared["initial"], case.initial,
                    rtol=2e-12, atol=2e-14), f"case{index}/independent_initial_reconstruction")
                gradient, objective, linear_rows = independent_gradient(case, prepared["initial"], protocol, checks, arrays)
                arrays["independent_true_gradient"] = gradient
                witness = production_witness(helpers, adapter, prepared, original, arrays)
                certificate = witness["certificate"]
                initial_sha = independent.array_sha(prepared["initial"])
                snapshot = witness["snapshot"]
                checks.require(initial_sha == snapshot["model_sha256"] == snapshot["model"]["sha256"]
                               and snapshot["model"]["shape"] == [2401]
                               and np.dtype(snapshot["model"]["dtype"]) == np.dtype("float64"),
                               f"case{index}/raw_initial_model_snapshot_binding")
                checks.require(array_record(arrays["production_damping"])["sha256"]
                               == snapshot["damping"]["sha256"],
                               f"case{index}/raw_production_damping_snapshot_binding")
                checks.require(not certificate["certified_descent"]
                               or ("direction" in arrays and "approximate_gradient" in arrays),
                               f"case{index}/certified_flag_requires_raw_vectors")
                near(checks, objective, witness["production_initial_objective"],
                     max(objective, 1e-12), protocol, f"case{index}/independent_Phi")
                for fi, frequency in enumerate(case.frequencies):
                    own = arrays[f"f{fi}_independent_observed"]
                    production = arrays[f"f{fi}_production_observed_slots"]
                    relative = independent.scalar_norm(own - production) / max(independent.scalar_norm(own), 1e-300)
                    checks.require(relative <= protocol["numeric_checks"]["observation_relative_L2_maximum"],
                                   f"case{index}/frequency{frequency}/independent_observations", {"relative": relative})
                direction = arrays.get("direction")
                approximate = arrays.get("approximate_gradient")
                dd = center = None
                if direction is not None and approximate is not None:
                    dd, center = scalar_dot(gradient, direction), scalar_dot(approximate, direction)
                    near(checks, center, certificate["approximate_directional_derivative"],
                         independent.scalar_norm(approximate) * independent.scalar_norm(direction),
                         protocol, f"case{index}/raw_approximate_gradient_dot_direction")
                    checks.require(independent.array_sha(direction) == certificate["direction_sha256"],
                                   f"case{index}/direction_SHA")
                    checks.require(independent.array_sha(approximate) == certificate["approximate_gradient_sha256"],
                                   f"case{index}/approximate_gradient_SHA")
                bound = certificate["directional_error_bound"]
                covers_literal = None if dd is None or bound is None else abs(dd - center) <= bound
                old_allowance = None if dd is None or bound is None else 1e-9 * max(abs(dd), abs(center), bound, 1.0)
                covers_historical = None if old_allowance is None else abs(dd - center) <= bound + old_allowance
                if covers_historical is False:
                    checks.require(False, f"case{index}/bound_misses_independent_DD_beyond_historical_allowance",
                                   {"true_DD": dd, "center": center, "bound": bound, "allowance": old_allowance})
                false_certification = None if dd is None else bool(certificate["certified_descent"] and dd >= 0)
                if false_certification:
                    checks.require(False, f"case{index}/independent_true_DD_false_certification",
                                   {"true_DD": dd, "center": center, "bound": bound})
                record.update(status="complete", raw_arrays_available=True, production=witness,
                    production_observation_preparation_costs=prepared["preparation_costs"],
                    independent_initial_objective=objective, independent_linear_solves=linear_rows,
                    independent_807_constructor_work={"factorizations": case.factorizations,
                        "forward_rhs": case.forward_rhs,
                        "scope": "own truth data plus own rebuilt-initial Phi; separate from fresh gradient state/adjoint solves"},
                    independent_true_directional_derivative=dd, reconstructed_approximate_center=center,
                    bound_covers_without_extra_allowance=covers_literal,
                    historical_helper_FP_allowance=old_allowance,
                    bound_covers_with_historical_helper_allowance=covers_historical,
                    false_certification_using_independent_gradient=false_certification,
                    raw_vector_scope="initial-model pointwise raw direction only; no clipped/finite-step certificate",
                    no_formal_FP_enclosure=True)
            except BaseException as exc:
                interruption = exc if not isinstance(exc, Exception) else None
                record.update(status="interrupted" if interruption is not None else "failed",
                              error=f"{type(exc).__name__}: {exc}",
                              missing_information_is_unknown_not_zero=True)
                checks.require(False, f"case{index}/verification_failed", record["error"])
            if arrays:
                array_path = output / f"case_{index:02d}_arrays.npz"
                with array_path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                record.update(raw_arrays_artifact=array_path.name, raw_arrays_file_sha256=sha(array_path.read_bytes()),
                              arrays={name: array_record(value) for name, value in arrays.items()},
                              raw_arrays_available=True, raw_witness_complete=record["status"] == "complete")
            record["finished_utc"] = utc()
            record_path = output / f"case_{index:02d}_receipt.json"
            new_json(record_path, record)
            row.update(status=record["status"], finished_utc=record["finished_utc"],
                       receipt_artifact=record_path.name, receipt_sha256=sha(record_path.read_bytes()),
                       direction_available="direction" in arrays)
            status_json(output / "status.json", status)
            if interruption is not None:
                raise interruption
            if independent.source_digest(root) != source_before:
                raise RuntimeError("production source changed; remaining cases unattempted")
        status.update(status="complete_initial_direction_verification", finished_utc=utc(),
            source_after=independent.source_digest(root), checks=checks.count,
            verification_result="clear" if not checks.failures else "issues_require_review", failures=checks.failures,
            cases_attempted=8, cases_complete=sum(row["status"] == "complete" for row in status["cases"]),
            replay_scope="Only these eight newly recomputed initial queries; not360intermediate-query replay or formalFP proof")
        status_json(output / "status.json", status)
        receipt = {**status, "artifact_sha256": {p.name: sha(p.read_bytes()) for p in output.iterdir() if p.is_file()}}
        new_json(output / "complete_receipt.json", receipt)
        return {"status": status["status"], "verification_result": status["verification_result"],
                "checks": checks.count, "issues": len(checks.failures), "output": str(output)}
    except BaseException as exc:
        status.update(status="failed_or_interrupted", finished_utc=utc(), error=f"{type(exc).__name__}: {exc}")
        status_json(output / "status.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-after-terminal-confirmation", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--objective-auditor", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-protocol-sha256")
    args = parser.parse_args(argv)
    if not args.execute_after_terminal_confirmation:
        print(json.dumps({"status": "source_only_no_numeric_imports_or_project_reads",
                          "source_sha256": sha(Path(__file__).read_bytes()),
                          "protocol_sha256": sha(args.protocol.read_bytes())}))
        return 0
    if not all((args.repo, args.benchmark, args.output, args.objective_auditor,
                args.expected_source_sha256, args.expected_protocol_sha256)):
        parser.error("execution requires all paths, both reviewed SHA values and explicit terminal-confirmation flag")
    result = run(args.repo, args.benchmark, args.output, args.objective_auditor, args.protocol,
                 args.expected_source_sha256, args.expected_protocol_sha256)
    print(json.dumps(result))
    return 0 if result["verification_result"] == "clear" else 1


if __name__ == "__main__":
    raise SystemExit(main())
