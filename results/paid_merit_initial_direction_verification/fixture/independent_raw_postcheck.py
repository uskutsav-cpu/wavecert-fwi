"""Second raw-witness arithmetic check; NumPy/stdlib, no production or PDE solve."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np

PINS = {
    "runner_source.py": "52cdbb46e8337b955cbdec13a0b1ed96637c8c34da337e1f3269ad423ffe38f9",
    "replay_raw_witnesses.py": "1e9af84e461861c98dfc69f34d85d88c539b40f6658d3f1924cd32cb10729ccc",
    "protocol.json": "e53db61db1a6c3d83a5fdf3cba195c357f37f54e125e793980a76e33d5923e46",
    "independent_assembly_source.py": "807f4f2f0e0aeea1d2f61b64ee5f64a99819c38d4f6164529e664f0a235edeae",
}
SOURCE_SHA = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    canonical = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def norm(array):
    flat = np.asarray(array).ravel()
    if np.iscomplexobj(flat):
        return math.hypot(*(float(x) for pair in zip(flat.real, flat.imag, strict=True) for x in pair))
    return math.hypot(*(float(x) for x in flat))


def dot(left, right):
    return math.fsum(float(x)*float(y) for x, y in zip(left, right, strict=True))


def stencil(model, damping, fields, frequency, adjoint=False):
    # Pad-then-neighbor formula, distinct from the copied replay slice updates.
    u = fields.reshape(49, 49, -1)
    padded = np.pad(u, ((1, 1), (1, 1), (0, 0)))
    lap = (4*u-padded[:-2, 1:-1]-padded[2:, 1:-1]
           - padded[1:-1, :-2]-padded[1:-1, 2:])/(.025*.025)
    omega = 2*math.pi*frequency
    potential = -omega*omega*model + (1j if not adjoint else -1j)*omega*damping
    return (lap + potential.reshape(49, 49, 1)*u).reshape(fields.shape)


def source_guard(repo):
    paths = [repo/"pyproject.toml"]
    paths += list((repo/"src").rglob("*.py"))
    paths += [p for p in (repo/"configs").rglob("*") if p.is_file()]
    paths += [p for p in (repo/"scripts").rglob("*") if p.suffix in {".py", ".sh"}]
    paths += list((repo/".github/workflows").glob("*.yml"))
    paths += list(repo.glob("requirements*.txt"))
    digest = hashlib.sha256()
    count = 0
    for path in sorted(set(paths)):
        if path.is_file():
            digest.update(path.relative_to(repo).as_posix().encode()+b"\0"+sha(path).encode()+b"\0")
            count += 1
    return {"sha256": digest.hexdigest(), "files": count}


def audit(repo, run, report):
    issues, results, maxima = [], [], {}
    count = 0

    def check(condition, context):
        nonlocal count
        count += 1
        if not bool(condition):
            issues.append(context)

    def vector_check(actual, saved, context):
        relative = norm(actual-saved)/max(norm(actual), norm(saved), 1e-300)
        maxima[context.split("/")[-1]] = max(maxima.get(context.split("/")[-1], 0.), relative)
        check(math.isfinite(relative) and relative <= numeric["vector_relative_L2_maximum"], context)
        return relative

    def scalar_check(actual, saved, scale, context):
        difference = abs(actual-saved)
        allowance = numeric["scalar_relative_allowance"]*max(abs(actual), abs(saved), scale, numeric["scale_floor"])
        maxima["scalar_fraction_of_allowance"] = max(maxima.get("scalar_fraction_of_allowance", 0.), difference/allowance)
        check(math.isfinite(actual) and math.isfinite(saved) and difference <= allowance, context)

    try:
        status = json.loads((run/"status.json").read_bytes())
        complete = json.loads((run/"complete_receipt.json").read_bytes())
        check(status["status"] == "complete_initial_direction_verification", "producer terminal status")
        check({k: v for k, v in complete.items() if k != "artifact_sha256"} == status, "complete/status equality")
        check(status["source_before"] == status["source_after"], "producer source before/after")
        check(status["verification_result"] == "clear" and status["failures"] == [], "producer verification clear")
        for name, digest in PINS.items():
            check(sha(run/name) == digest, "copied source pin: "+name)
        for name, digest in complete["artifact_sha256"].items():
            path = (run/name).resolve()
            check(path.parent == run and sha(path) == digest, "manifest: "+name)
        guard = source_guard(repo)
        check(guard == status["source_before"] and guard["sha256"] == SOURCE_SHA, "actual current source9f guard")
        protocol = json.loads((run/"protocol.json").read_bytes())
        numeric = protocol["numeric_checks"]
        check([r["case_index"] for r in status["cases"]] == list(range(8)), "all8 denominator")
        for row in status["cases"]:
            index = row["case_index"]
            check(row["status"] == "complete", f"case{index}/complete")
            receipt_path = run/row["receipt_artifact"]
            check(sha(receipt_path) == row["receipt_sha256"], f"case{index}/receipt SHA")
            receipt = json.loads(receipt_path.read_bytes())
            path = run/receipt["raw_arrays_artifact"]
            check(sha(path) == receipt["raw_arrays_file_sha256"], f"case{index}/NPZ SHA")
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            check(set(arrays) == set(receipt["arrays"]), f"case{index}/all arrays")
            for name, value in arrays.items():
                desc = receipt["arrays"][name]
                check(array_sha(value) == desc["sha256"] and list(value.shape) == desc["shape"]
                      and value.dtype.newbyteorder("<").str == desc["dtype"] and bool(np.all(np.isfinite(value))),
                      f"case{index}/{name}/canonical finite raw descriptor")
            model, damping = arrays["initial_model"], arrays["independent_damping"]
            receivers, sources, rhs = arrays["receiver_indices"], arrays["source_indices"], arrays["source_rhs"]
            cert = receipt["production"]["certificate"]
            snapshot = receipt["production"]["snapshot"]
            check(array_sha(model) == snapshot["model_sha256"] == snapshot["model"]["sha256"], f"case{index}/model binding")
            check(array_sha(arrays["production_damping"]) == snapshot["damping"]["sha256"], f"case{index}/damping binding")
            expected_rhs = np.zeros((2401, 6), dtype=np.complex128)
            expected_rhs[sources, np.arange(6)] = 1/(.025*.025)
            check(np.array_equal(rhs, expected_rhs) and len(receivers) == 20, f"case{index}/source amplitude and receiver count")
            true_gradient, approximate = np.zeros(2401), np.zeros(2401)
            max_residual = 0.
            scalar_bounds = []
            for fi, frequency in enumerate(protocol["frequencies_hz"]):
                own_state = arrays[f"f{fi}_independent_state"]
                own_adjoint = arrays[f"f{fi}_independent_adjoint"]
                paid = arrays[f"f{fi}_production_cached_field"]
                missing = arrays[f"f{fi}_production_missing_field"]
                residual = own_state[receivers]-arrays[f"f{fi}_independent_observed"]
                forcing = np.zeros_like(rhs)
                np.add.at(forcing, receivers, residual)
                for field, load, adjoint in ((own_state, rhs, False), (own_adjoint, forcing, True), (paid, rhs, False)):
                    defect = stencil(model, damping, field, frequency, adjoint)-load
                    relative = max(norm(defect[:, j])/max(norm(load[:, j]), 1e-300) for j in range(6))
                    max_residual = max(max_residual, relative)
                    check(relative <= numeric["linear_relative_residual_maximum"], f"case{index}/f{fi}/direct stencil residual")
                omega2 = (2*math.pi*frequency)**2
                # Expand complex conjugate products and accumulate one source
                # at a time, rather than reuse production np.sum(conj(a)*u).
                for j in range(6):
                    true_gradient += omega2*(own_adjoint[:, j].real*own_state[:, j].real
                                            + own_adjoint[:, j].imag*own_state[:, j].imag)
                    approximate += omega2*(missing[:, j].real*paid[:, j].real
                                          + missing[:, j].imag*paid[:, j].imag)
                freq = cert["frequency_records"][fi]
                if freq.get("directional_error_bound") is not None:
                    terms = math.fsum(float(x) for component in freq["directional_bound_terms"] for x in component)
                    scalar_check(terms, freq["directional_error_bound"], terms, f"case{index}/f{fi}/term sum")
                    scalar_bounds.append(terms)
            true_error = vector_check(true_gradient, arrays["independent_true_gradient"], f"case{index}/true_gradient_relativeL2")
            result = {"case_index": index, "identity": row["identity"], "raw_arrays": len(arrays),
                      "in_denominator": True, "certificate_available": cert["available"],
                      "certified": cert["certified_descent"], "true_gradient_relativeL2": true_error,
                      "maximum_direct_stencil_relative_residual": max_residual}
            if "direction" in arrays and "approximate_gradient" in arrays:
                direction = arrays["direction"]
                approx_error = vector_check(approximate, arrays["approximate_gradient"], f"case{index}/approximate_gradient_relativeL2")
                normalized = -approximate/norm(approximate) if norm(approximate) else np.zeros_like(approximate)
                direction_error = vector_check(normalized, direction, f"case{index}/direction_relativeL2")
                true_dd, center = dot(true_gradient, direction), dot(approximate, direction)
                scalar_check(true_dd, receipt["independent_true_directional_derivative"], norm(true_gradient)*norm(direction), f"case{index}/true DD")
                scalar_check(center, cert["approximate_directional_derivative"], norm(approximate)*norm(direction), f"case{index}/center DD")
                bound = cert["directional_error_bound"]
                literal = historical = None
                if bound is not None:
                    scalar_check(math.fsum(scalar_bounds), bound, bound, f"case{index}/survey bound")
                    literal = bool(abs(true_dd-center) <= bound)
                    allowance = 1e-9*max(abs(true_dd), abs(center), bound, 1.)
                    historical = bool(abs(true_dd-center) <= bound+allowance)
                    check(literal == receipt["bound_covers_without_extra_allowance"], f"case{index}/literal coverage flag")
                    check(historical == receipt["bound_covers_with_historical_helper_allowance"], f"case{index}/historical coverage flag")
                    check(historical, f"case{index}/historical bound coverage")
                false = bool(cert["certified_descent"] and true_dd >= 0)
                check(not false, f"case{index}/no false certification")
                result.update(approximate_gradient_relativeL2=approx_error, direction_relativeL2=direction_error,
                              true_directional_derivative=true_dd, center=center, bound=bound,
                              literal_coverage=literal, historical_coverage=historical, false_certification=false)
            else:
                result.update(direction_unavailable=True, literal_coverage=None, historical_coverage=None,
                              false_certification=None)
            results.append(result)
        report.update(source_guard=guard, manifest_files=len(complete["artifact_sha256"]), cases=results,
                      max_reconstruction_discrepancies=maxima,
                      summary={"planned": 8, "raw_checked": len(results),
                               "available": sum(r["certificate_available"] for r in results),
                               "certified": sum(r["certified"] for r in results),
                               "literal_covered": sum(r.get("literal_coverage") is True for r in results),
                               "historical_covered": sum(r.get("historical_coverage") is True for r in results),
                               "false_certifications": sum(r.get("false_certification") is True for r in results)})
    except BaseException as exc:
        issues.append(type(exc).__name__+": "+str(exc))
        report.update(exception_traceback=traceback.format_exc(), partial_cases=results)
    report.update(result="CLEAR_SECOND_RAW_CHECK" if not issues else "ISSUES_REQUIRE_REVIEW", checks=count, issues=issues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, run, output = args.repo.resolve(), args.run.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(repo) or output.is_relative_to(run):
        parser.error("fresh outer output required")
    report = {"role": "Independent AI second raw arithmetic check, only eight initial queries",
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "checker_sha256": sha(Path(__file__)), "PDE_or_production_imports_or_producer_rerun": False,
              "scope": "Raw fields/complex contractions/direct stencil/DD and stored-term sums; no independent coarse constants, FP enclosure, performance,360query or ownership conclusion"}
    audit(repo, run, report)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"result": report["result"], "checks": report["checks"], "issues": len(report["issues"]),
                      "summary": report.get("summary"), "sha256": sha(output), "output": str(output)}))
    return 0 if not report["issues"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
