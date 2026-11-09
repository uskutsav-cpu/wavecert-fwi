"""NumPy-only replay of saved eight-initial-query raw field/vector witnesses.

No production modules, SciPy solver, PDE solve, original fixture or optimizer is
used. Source-only preparation; do not run before reviewed witness acquisition.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(values):
    values = np.asarray(values).ravel()
    if np.iscomplexobj(values):
        return math.hypot(*(float(v) for pair in zip(values.real, values.imag, strict=True) for v in pair))
    return math.hypot(*(float(v) for v in values))


def dot(left, right):
    return math.fsum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def apply(model, damping, field, frequency, *, adjoint=False):
    # Independent direct five-point array action, avoiding any sparse assembly.
    omega = 2.0 * math.pi * frequency
    interior = field.reshape(49, 49, -1)
    result = (4.0 / 0.025 ** 2 - omega ** 2 * model.reshape(49, 49)
              + (-1j if adjoint else 1j) * omega * damping.reshape(49, 49))[:, :, None] * interior
    result[1:, :, :] -= interior[:-1, :, :] / 0.025 ** 2
    result[:-1, :, :] -= interior[1:, :, :] / 0.025 ** 2
    result[:, 1:, :] -= interior[:, :-1, :] / 0.025 ** 2
    result[:, :-1, :] -= interior[:, 1:, :] / 0.025 ** 2
    return result.reshape(field.shape)


def run(directory, output, expected_source_sha256):
    global np
    if sha(Path(__file__).read_bytes()) != expected_source_sha256:
        raise ValueError("reviewed raw replayer SHA mismatch")
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(directory):
        raise FileExistsError("new replay output outside witness directory required")
    status = json.loads((directory / "status.json").read_bytes())
    if status.get("status") != "complete_initial_direction_verification":
        raise RuntimeError("terminal complete eight-initial-query witness receipt required")
    complete = json.loads((directory / "complete_receipt.json").read_bytes())
    if {key: value for key, value in complete.items() if key != "artifact_sha256"} != status:
        raise ValueError("terminal complete/status differ")
    for name, expected in complete["artifact_sha256"].items():
        path = (directory / name).resolve()
        if path.parent != directory or sha(path.read_bytes()) != expected:
            raise ValueError("witness artifact hash mismatch: " + name)
    protocol = json.loads((directory / "protocol.json").read_bytes())
    if sha((directory / "protocol.json").read_bytes()) != status["protocol_sha256"]:
        raise ValueError("witness protocol SHA mismatch")
    import numpy as np

    issues, results = [], []
    checks = 0

    def require(condition, context, detail=None):
        nonlocal checks
        checks += 1
        if not bool(condition):
            issues.append({"context": context, "detail": detail})

    def vector_near(actual, stored, context):
        relative = norm(actual - stored) / max(norm(actual), norm(stored), 1e-300)
        require(math.isfinite(relative) and relative <= protocol["numeric_checks"]["vector_relative_L2_maximum"],
                context, {"relative_L2": relative})
        return relative

    def scalar_near(actual, stored, scale, context):
        allowance = protocol["numeric_checks"]["scalar_relative_allowance"] * max(
            abs(actual), abs(stored), scale, protocol["numeric_checks"]["scale_floor"])
        require(math.isfinite(actual) and math.isfinite(stored) and abs(actual - stored) <= allowance,
                context, {"replayed": actual, "stored": stored, "allowance": allowance})

    require(status["verification_result"] == "clear", "upstream_verification_must_be_clear",
            {"result": status["verification_result"], "failures": status.get("failures")})
    if len(status["cases"]) != 8 or [row["case_index"] for row in status["cases"]] != list(range(8)):
        raise ValueError("all eight fixed case slots required")
    for row in status["cases"]:
        index = row["case_index"]
        receipt_path = directory / row["receipt_artifact"]
        if sha(receipt_path.read_bytes()) != row["receipt_sha256"]:
            raise ValueError("case receipt hash mismatch")
        receipt = json.loads(receipt_path.read_bytes())
        record = {"case_index": index, "witness_status": receipt["status"], "in_denominator": True}
        if receipt["status"] != "complete":
            require(False, f"case{index}/witness_not_fully_available", receipt.get("error"))
            record.update(replay_available=False, unavailable_information_is_unknown_not_zero=True)
            results.append(record)
            continue
        array_path = directory / receipt["raw_arrays_artifact"]
        if sha(array_path.read_bytes()) != receipt["raw_arrays_file_sha256"]:
            raise ValueError("raw array NPZ file hash mismatch")
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        require(set(arrays) == set(receipt["arrays"]), f"case{index}/all_raw_array_keys")
        for name, values in arrays.items():
            canonical = np.ascontiguousarray(values, dtype=values.dtype.newbyteorder("<"))
            descriptor = receipt["arrays"][name]
            require(list(values.shape) == descriptor["shape"] and canonical.dtype.str == descriptor["dtype"]
                    and sha(canonical.tobytes()) == descriptor["sha256"], f"case{index}/{name}/shape_dtype_SHA")
        model, damping = arrays["initial_model"], arrays["independent_damping"]
        rhs, receivers = arrays["source_rhs"], arrays["receiver_indices"]
        snapshot = receipt["production"]["snapshot"]
        initial_sha = sha(np.ascontiguousarray(model, dtype="<f8").tobytes())
        require(initial_sha == snapshot["model_sha256"] == snapshot["model"]["sha256"]
                and list(model.shape) == snapshot["model"]["shape"]
                and np.dtype(snapshot["model"]["dtype"]) == model.dtype,
                f"case{index}/raw_initial_model_actual_snapshot_binding")
        production_damping = arrays["production_damping"]
        require(sha(np.ascontiguousarray(production_damping, dtype="<f8").tobytes())
                == snapshot["damping"]["sha256"]
                and list(production_damping.shape) == snapshot["damping"]["shape"]
                and np.dtype(snapshot["damping"]["dtype"]) == production_damping.dtype,
                f"case{index}/raw_production_damping_actual_snapshot_binding")
        vector_near(damping, arrays["production_damping"], f"case{index}/production_vs_independent_damping")
        approximate, exact = np.zeros(2401), np.zeros(2401)
        frame_centers, frequency_bounds, field_rows = [], [], []
        certificate = receipt["production"]["certificate"]
        for fi, frequency in enumerate(protocol["frequencies_hz"]):
            own_state = arrays[f"f{fi}_independent_state"]
            own_adjoint = arrays[f"f{fi}_independent_adjoint"]
            own_observed = arrays[f"f{fi}_independent_observed"]
            own_residual = own_state[receivers, :] - own_observed
            own_rhs = np.zeros_like(rhs)
            np.add.at(own_rhs, receivers, own_residual)
            vector_near(own_residual, arrays[f"f{fi}_independent_residual"], f"case{index}/f{fi}/own_data_residual")
            vector_near(own_rhs, arrays[f"f{fi}_independent_adjoint_rhs"], f"case{index}/f{fi}/own_adjoint_rhs")
            for name, field, forcing, adjoint in (
                    ("state", own_state, rhs, False), ("adjoint", own_adjoint, own_rhs, True)):
                defect = apply(model, damping, field, frequency, adjoint=adjoint) - forcing
                relative = max(norm(defect[:, col]) / max(norm(forcing[:, col]), 1e-300)
                               for col in range(forcing.shape[1]))
                require(relative <= protocol["numeric_checks"]["linear_relative_residual_maximum"],
                        f"case{index}/f{fi}/raw_{name}_discrete_residual", {"relative": relative})
            paid = arrays[f"f{fi}_production_cached_field"]
            missing = arrays[f"f{fi}_production_missing_field"]
            frame = receipt["production"]["frames"][fi]
            require(frame["source_indices"] == arrays["source_indices"].tolist()
                    and frame["receiver_indices"] == receivers.tolist(),
                    f"case{index}/f{fi}/ordered_source_receiver_binding")
            for name in ("cached_field", "observed_slots", "residual_slots", "residual_unique", "receiver_expand_indices"):
                field = arrays[f"f{fi}_production_{name}"]
                descriptor = frame[name]
                require(list(field.shape) == descriptor["shape"] and np.dtype(descriptor["dtype"]) == field.dtype
                        and sha(np.ascontiguousarray(field).tobytes()) == descriptor["sha256"],
                        f"case{index}/f{fi}/{name}/actual_snapshot_binding")
            missing_sha = sha(np.ascontiguousarray(missing, dtype="<c16").tobytes())
            require(missing_sha == receipt["production"]["proposal_diagnostics"]["field_sha256"][str(frequency)],
                    f"case{index}/f{fi}/actual_missing_field_binding")
            production_observed = arrays[f"f{fi}_production_observed_slots"]
            production_residual = paid[receivers, :] - production_observed
            vector_near(production_residual, arrays[f"f{fi}_production_residual_slots"],
                        f"case{index}/f{fi}/production_data_residual")
            omega = 2.0 * math.pi * frequency
            contribution = omega ** 2 * np.real(np.sum(np.conjugate(missing) * paid, axis=1))
            approximate += contribution
            exact += omega ** 2 * np.real(np.sum(np.conjugate(own_adjoint) * own_state, axis=1))
            paid_relative = norm(apply(model, damping, paid, frequency) - rhs) / max(norm(rhs), 1e-300)
            require(paid_relative <= protocol["numeric_checks"]["linear_relative_residual_maximum"],
                    f"case{index}/f{fi}/production_paid_state_residual", {"relative": paid_relative})
            field_rows.append({"frequency_hz": frequency, "production_paid_field_relative_residual": paid_relative})
            frequency_record = (certificate["frequency_records"][fi]
                                if fi < len(certificate["frequency_records"]) else None)
            if frequency_record is None:
                field_rows[-1]["certificate_frequency_record_unavailable"] = True
                continue
            require(frequency_record["frequency_hz"] == frequency, f"case{index}/f{fi}/certificate_frequency")
            production_rhs = np.zeros_like(rhs)
            np.add.at(production_rhs, receivers, production_residual)
            missing_defect = apply(model, damping, missing, frequency, adjoint=True) - production_rhs
            for column, stored in enumerate(frequency_record["missing_raw_residual_norms"]):
                actual = norm(missing_defect[:, column])
                action_bound = (8.0 / 0.025 ** 2 + omega ** 2 * float(np.max(model))
                                + omega * float(np.max(damping)))
                action_scale = action_bound * norm(missing[:, column]) + norm(production_rhs[:, column])
                allowance = (protocol["numeric_checks"]["residual_norm_action_scale_allowance"] * action_scale
                             + protocol["numeric_checks"]["scalar_relative_allowance"] * max(
                                 actual, stored, protocol["numeric_checks"]["scale_floor"]))
                require(abs(actual - stored) <= allowance,
                        f"case{index}/f{fi}/raw_missing_residual_norm{column}",
                        {"replayed": actual, "stored": stored, "action_scale": action_scale,
                         "allowance": allowance, "subtractive_residual_roundoff_not_formal_enclosure": True})
            if "direction" in arrays:
                frame_center = dot(contribution, arrays["direction"])
                frame_centers.append(frame_center)
                if frequency_record["approximate_directional_derivative"] is not None:
                    scalar_near(frame_center, frequency_record["approximate_directional_derivative"],
                                norm(contribution) * norm(arrays["direction"]), f"case{index}/f{fi}/center")
            bound = frequency_record.get("directional_error_bound")
            if bound is not None:
                terms = frequency_record["directional_bound_terms"]
                reconstructed_bound = math.fsum(float(value) for component in terms for value in component)
                scalar_near(reconstructed_bound, bound, bound, f"case{index}/f{fi}/scalar_bound_terms")
                frequency_bounds.append(bound)
        exact_error = vector_near(exact, arrays["independent_true_gradient"], f"case{index}/raw_true_gradient")
        record.update(replay_available=True, array_count=len(arrays), independent_gradient_relative_error=exact_error,
                      field_diagnostics=field_rows)
        if "direction" in arrays and "approximate_gradient" in arrays:
            error = vector_near(approximate, arrays["approximate_gradient"], f"case{index}/raw_approximate_gradient")
            direction = arrays["direction"]
            require(sha(np.ascontiguousarray(direction, dtype="<f8").tobytes()) == certificate["direction_sha256"],
                    f"case{index}/raw_direction_certificate_SHA")
            require(sha(np.ascontiguousarray(arrays["approximate_gradient"], dtype="<f8").tobytes())
                    == certificate["approximate_gradient_sha256"], f"case{index}/raw_approximate_gradient_certificate_SHA")
            normalized = -approximate / norm(approximate) if norm(approximate) > 0 else np.zeros_like(approximate)
            vector_near(normalized, direction, f"case{index}/unclipped_normalized_raw_direction")
            true_dd, center = dot(exact, direction), dot(approximate, direction)
            scalar_near(true_dd, receipt["independent_true_directional_derivative"], norm(exact) * norm(direction),
                        f"case{index}/true_DD")
            scalar_near(center, certificate["approximate_directional_derivative"], norm(approximate) * norm(direction),
                        f"case{index}/approximate_DD")
            bound = certificate["directional_error_bound"]
            if bound is not None:
                scalar_near(math.fsum(frequency_bounds), bound, bound, f"case{index}/summed_bound")
                scalar_near(center + bound, certificate["certified_upper_derivative"], abs(center) + bound,
                            f"case{index}/center_plus_bound")
            false = bool(certificate["certified_descent"] and true_dd >= 0)
            require(receipt["false_certification_using_independent_gradient"] is false,
                    f"case{index}/false_certification_flag")
            require(not false, f"case{index}/no_independently_replayed_false_certification",
                    {"true_DD": true_dd, "certified": certificate["certified_descent"]})
            literal = None if bound is None else abs(true_dd - center) <= bound
            require(receipt["bound_covers_without_extra_allowance"] is literal,
                    f"case{index}/literal_bound_coverage_flag")
            historical_allowance = None if bound is None else 1e-9 * max(abs(true_dd), abs(center), bound, 1.0)
            historical_covers = (None if historical_allowance is None
                                 else abs(true_dd - center) <= bound + historical_allowance)
            require(receipt["bound_covers_with_historical_helper_allowance"] is historical_covers,
                    f"case{index}/historical_allowance_coverage_flag")
            if historical_allowance is None:
                require(receipt["historical_helper_FP_allowance"] is None,
                        f"case{index}/unavailable_bound_no_historical_allowance")
            else:
                scalar_near(historical_allowance, receipt["historical_helper_FP_allowance"],
                            historical_allowance, f"case{index}/historical_allowance_arithmetic")
                require(historical_covers, f"case{index}/no_replayed_bound_miss_beyond_historical_allowance",
                        {"true_DD": true_dd, "center": center, "bound": bound,
                         "historical_allowance": historical_allowance})
            record.update(approximate_gradient_relative_error=error, true_directional_derivative=true_dd,
                          approximate_center=center, bound=bound, certified=certificate["certified_descent"],
                          false_certification=false, bound_covers_without_extra_allowance=literal,
                          historical_helper_FP_allowance=historical_allowance,
                          bound_covers_with_historical_helper_allowance=historical_covers)
        else:
            require(certificate["certified_descent"] is False,
                    f"case{index}/missing_raw_direction_cannot_be_certified")
            record.update(direction_DD_available=False, unavailable_information_is_unknown_not_zero=True)
        results.append(record)
    report = {"result": "clear_raw_witness_replay" if not issues else "issues_require_review",
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "replayer_source_sha256": sha(Path(__file__).read_bytes()), "checks": checks, "issues": issues,
              "all8_case_slots_retained": len(results) == 8, "cases": results,
              "upstream_verification_result": status["verification_result"],
              "directions_assessed": sum("true_directional_derivative" in row for row in results),
              "false_certification_count": sum(row.get("false_certification", False) for row in results),
              "literal_bound_miss_count": sum(row.get("bound_covers_without_extra_allowance") is False for row in results),
              "scope": "NumPy-only raw saved-vector/field arithmetic and discrete residual checks; no new PDE solve",
              "limitations": ["Eight newly recomputed initial queries only, not all360intermediate trajectory certificates.",
                             "No independently recomputed coarse stability constants or formal FP enclosure.",
                             "No optimizer, timing benefit, independent geology, human or student ownership claim."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write((json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return {"result": report["result"], "checks": checks, "issues": len(issues),
            "output": str(output), "sha256": sha(output.read_bytes())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-after-terminal-confirmation", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args(argv)
    if not args.execute_after_terminal_confirmation:
        print(json.dumps({"status": "source_only_no_project_or_numeric_reads",
                          "source_sha256": sha(Path(__file__).read_bytes())}))
        return 0
    if not all((args.run, args.output, args.expected_source_sha256)):
        parser.error("explicit terminal flag, run/new output and reviewed source SHA required")
    result = run(args.run, args.output, args.expected_source_sha256)
    print(json.dumps(result))
    return 0 if result["result"] == "clear_raw_witness_replay" else 1


if __name__ == "__main__":
    raise SystemExit(main())
