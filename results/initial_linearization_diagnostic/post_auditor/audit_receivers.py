"""Small independent scalar receiver audit; execution requires frozen hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def real_product(a, b):
    return math.fsum(term for x, y in zip(np.asarray(a).ravel(), np.asarray(b).ravel(), strict=True)
                     for term in (float(x.real) * float(y.real), float(x.imag) * float(y.imag)))


def squared_norm(a):
    return math.fsum(term for x in np.asarray(a).ravel() for term in (float(x.real) ** 2, float(x.imag) ** 2))


def metrics(records):
    corrections = [truth - initial for truth, initial, tangent in records]
    tangents = [tangent for truth, initial, tangent in records]
    remainders = [change - tangent for change, tangent in zip(corrections, tangents, strict=True)]
    energies = [math.fsum(squared_norm(value) for value in values) for values in [corrections, tangents, remainders]]
    data, tangent, remainder = (math.sqrt(value) for value in energies)
    dot = math.fsum(real_product(change, linear) for change, linear in zip(corrections, tangents, strict=True))
    return dict(data_change_norm=data, tangent_norm=tangent, remainder_norm=remainder,
        remainder_relative_to_data_change=remainder / data if data else None,
        real_alignment=dot / (data * tangent) if data and tangent else None,
        zero_data_change=data == 0, zero_tangent=tangent == 0,
        initial_objective=.5 * energies[0], linearized_truth_step_objective=.5 * energies[2], g_data_dot_delta=-dot)


def full_band(payload, cases, frequencies):
    keys = [(case, frequency) for case in cases for frequency in frequencies]
    return metrics([payload[key] for key in keys]) if all(key in payload for key in keys) else None


def descriptor(a):
    canonical = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))
    return dict(shape=list(a.shape), dtype=canonical.dtype.str,
        sha256=hashlib.sha256(canonical.tobytes()).hexdigest(), raw_values_saved=True)


def manifest(directory):
    return {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def source_digest(root):
    files = [root / "pyproject.toml"] + list((root / "src").rglob("*.py"))
    files += [p for p in (root / "configs").rglob("*") if p.is_file()]
    files += [p for p in (root / "scripts").rglob("*") if p.suffix in {".py", ".sh"}]
    files += list((root / ".github/workflows").glob("*.yml")) + list(root.glob("requirements*.txt"))
    digest = hashlib.sha256()
    for p in sorted(set(files)):
        if p.is_file():
            digest.update(p.relative_to(root).as_posix().encode() + b"\0" + sha(p).encode() + b"\0")
    return digest.hexdigest()


def write(path, value):
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def run(args, criteria):
    if args.output.exists() or any(args.output.resolve().is_relative_to(path.resolve())
        for path in [args.run, args.witness, args.repo, args.producer.parent]):
        raise FileExistsError("new outer audit output outside immutable inputs/source required")
    if sha(args.producer) != criteria["producer_sha256"] or sha(args.producer.with_name("protocol.json")) != criteria["producer_protocol_sha256"]:
        raise ValueError("frozen producer/protocol changed")
    if sha(args.run / "complete_receipt.json") != criteria["native_complete_sha256"]:
        raise ValueError("root-confirmed native complete receipt changed")
    protocol = json.loads(args.producer.with_name("protocol.json").read_text())
    args.output.mkdir()
    checks, payload = [], {}
    report = dict(status="running", physical_acceptance=None, scope=criteria["scope"], checks=checks,
        planned_counts=dict(cases=8, frequencies=64, stages=192, factorizations=128, rhs_columns=1152,
                            case_bands=16, all8_bands=2, gradient_contracts=8),
        cases=[dict(case_index=i, status="unattempted", frequencies=[dict(frequency_hz=f, status="unattempted")
            for f in protocol["frequencies_hz"]], bands={b: dict(status="not_read") for b in protocol["bands_hz"]})
            for i in range(8)], all8_bands={b: dict(status="not_read") for b in protocol["bands_hz"]})
    def check(name, condition, **details):
        checks.append(dict(check=name, passed=bool(condition), **details))
    def near(name, actual, expected, scale=None):
        if actual is None or expected is None:
            check(name, actual is expected, reported=actual, recomputed=expected)
            return
        bound = criteria["scalar_report_absolute_tolerance"] + criteria["scalar_report_relative_tolerance"] * (abs(expected) if scale is None else abs(scale))
        check(name, math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= bound,
              reported=actual, recomputed=expected, bound=bound)
    def compare(label, reported, value, keys=None):
        keys = value if keys is None else keys
        for key in keys:
            if isinstance(value[key], bool):
                check(label + ":" + key, reported[key] is value[key])
            else:
                scale = value["data_change_norm"] * value["tangent_norm"] if key == "g_data_dot_delta" else None
                near(label + ":" + key, reported[key], value[key], scale)
    before, old_pins = manifest(args.run), {}
    write(args.output / "status.json", report)
    try:
        status = json.loads((args.run / "complete_receipt.json").read_text())
        check("root_status_receipt_equal", json.loads((args.run / "status.json").read_text()) == status)
        check("native_status_complete", status["status"] == "complete_diagnostic_attempts" and status["numerical_integrity_pass"] is True)
        check("native_source_protocol", status["source_sha256"] == criteria["producer_sha256"]
            and status["protocol_sha256"] == criteria["producer_protocol_sha256"] and status["physical_acceptance"] is None)
        check("current_source9f", source_digest(args.repo) == criteria["production_source_sha256"])
        complete_path = args.witness / "complete_receipt.json"
        check("original_complete", sha(complete_path) == criteria["original_complete_sha256"])
        original = json.loads(complete_path.read_text())
        for name, expected in protocol["input_artifacts_sha256"].items():
            old_pins[name] = sha(args.witness / name)
            check("old_input:" + name, old_pins[name] == expected == original["artifact_sha256"][name])
        rows = status["cases"]
        check("8_unique_cases", len(rows) == 8 and {r["case_index"] for r in rows} == set(range(8)))
        counters = dict(frequencies=0, stages=0, factorizations=0, rhs_columns=0, gradient_contracts=0, case_bands=0)
        declared_artifacts = set()
        for case_report in report["cases"]:
            index = case_report["case_index"]
            case_report["status"] = "attempted"
            write(args.output / "status.json", report)
            row = next(r for r in rows if r["case_index"] == index)
            check(f"case{index}:complete", row["status"] == "complete")
            frequency_rows = row["frequencies"]
            check(f"case{index}:8_unique_frequencies", len(frequency_rows) == 8
                and [r["frequency_hz"] for r in frequency_rows] == protocol["frequencies_hz"])
            for freq_report in case_report["frequencies"]:
                frequency = freq_report["frequency_hz"]
                native = next(r for r in frequency_rows if r["frequency_hz"] == frequency)
                freq_report.update(status=native["status"], operations=native["operations"])
                check(f"case{index}:f{frequency}:complete", native["status"] == "complete")
                counters["frequencies"] += 1
                for stage in ["truth", "initial", "tangent"]:
                    operation = native["operations"][stage]
                    factor_count = int(stage != "tangent")
                    check(f"case{index}:f{frequency}:{stage}:work", operation["status"] == "complete"
                        and operation["rhs_columns_planned"] == operation["rhs_columns_attempted"] == operation["rhs_columns_completed"] == 6
                        and operation["factorizations_attempted"] == operation["factorizations_completed"] == factor_count)
                    residuals = operation.get("true_relative_residuals", [])
                    check(f"case{index}:f{frequency}:{stage}:reported_residuals", len(residuals) == 6
                        and all(v is not None and math.isfinite(v) and v <= criteria["reported_linear_residual_limit"] for v in residuals))
                    counters["stages"] += 1
                    counters["factorizations"] += operation["factorizations_completed"]
                    counters["rhs_columns"] += operation["rhs_columns_completed"]
                artifact = args.run / native["artifact"]
                if artifact.resolve().parent != args.run.resolve():
                    raise ValueError("escaping native receiver artifact")
                declared_artifacts.add(artifact.name)
                check(f"case{index}:f{frequency}:artifact", sha(artifact) == native["artifact_sha256"])
                with np.load(artifact, allow_pickle=False) as bundle:
                    check(f"case{index}:f{frequency}:only_receivers", set(bundle.files) == {"truth_receiver", "initial_receiver", "tangent_receiver"})
                    values = tuple(np.array(bundle[key], copy=True) for key in ["truth_receiver", "initial_receiver", "tangent_receiver"])
                schema_ok = all(v.shape == (20, 6) and v.dtype == np.complex128 and np.isfinite(v).all() for v in values)
                check(f"case{index}:f{frequency}:array_schema", schema_ok)
                if not schema_ok:
                    raise ValueError("invalid/nonfinite receiver data; scalar evaluation unavailable")
                value = metrics([values])
                compare(f"case{index}:f{frequency}", native["diagnostics"], value)
                freq_report["scalar"] = value
                payload[index, frequency] = values
                write(args.output / "status.json", report)
            for band, frequencies in protocol["bands_hz"].items():
                value = full_band(payload, [index], frequencies)
                check(f"case{index}:{band}:full_band", value is not None and row["bands"][band]["status"] == "complete"
                    and row["bands"][band]["frequencies_hz"] == frequencies)
                case_report["bands"][band] = value
                if value is not None:
                    compare(f"case{index}:{band}", row["bands"][band], value)
                    counters["case_bands"] += 1
            old_case = json.loads((args.witness / f"case_{index:02d}_receipt.json").read_text())
            with np.load(args.witness / f"case_{index:02d}_arrays.npz", allow_pickle=False) as bundle:
                initial, truth, gradient = (np.array(bundle[key], copy=True) for key in
                    ["initial_model", "independent_truth_model", "independent_true_gradient"])
            for key, vector in zip(["initial_model", "independent_truth_model", "independent_true_gradient"],
                                   [initial, truth, gradient], strict=True):
                check(f"case{index}:old_vector:{key}", descriptor(vector) == old_case["arrays"][key])
            inherited = real_product(gradient, truth - initial)
            direct = case_report["bands"]["originalband"]["g_data_dot_delta"]
            scale = max(abs(inherited), abs(direct), math.sqrt(squared_norm(gradient) * squared_norm(truth - initial)), 1e-12)
            error, allowance = abs(inherited - direct), criteria["original_gradient_contract_relative_allowance"] * scale
            value = dict(saved_gradient_dot_delta=inherited, tangent_dot_residual=direct, absolute_error=error,
                         scale=scale, allowance=allowance, numerical_agreement=error <= allowance, formal_enclosure=False)
            native = row["original_band_saved_gradient_check"]
            for key in ["saved_gradient_dot_delta", "tangent_dot_residual", "absolute_error", "scale", "allowance"]:
                near(f"case{index}:saved_gradient:{key}", native[key], value[key], scale if key in ["saved_gradient_dot_delta", "tangent_dot_residual", "absolute_error"] else None)
            check(f"case{index}:saved_gradient:criterion", native["numerical_agreement"] == value["numerical_agreement"] and value["numerical_agreement"])
            check(f"case{index}:saved_gradient:not_formal", native["formal_enclosure"] is False)
            case_report.update(status="scalar_audited", saved_gradient_contract=value)
            counters["gradient_contracts"] += 1
            write(args.output / "status.json", report)
        check("no_extra_or_missing_receiver_artifacts", {p.name for p in args.run.glob("*.npz")} == declared_artifacts and len(declared_artifacts) == 64)
        aggregate_keys = ["data_change_norm", "tangent_norm", "remainder_norm", "remainder_relative_to_data_change", "real_alignment", "g_data_dot_delta"]
        for band, frequencies in protocol["bands_hz"].items():
            value = full_band(payload, range(8), frequencies)
            native = status["bands_all8"][band]
            check(band + ":all8_denominator", value is not None and native["status"] == "complete"
                and native["planned_cases"] == native["complete_cases"] == 8 and native["independent_sample_claim"] is False)
            report["all8_bands"][band] = value
            if value is not None:
                compare(band + ":all8", native, value, aggregate_keys)
        check("full_work_and_band_counts", counters == dict(frequencies=64, stages=192, factorizations=128,
              rhs_columns=1152, gradient_contracts=8, case_bands=16), actual=counters)
        check("native_files_immutable", manifest(args.run) == before)
        check("old_inputs_immutable", old_pins == {name: sha(args.witness / name) for name in old_pins})
        check("original_complete_immutable", sha(complete_path) == criteria["original_complete_sha256"])
        check("audit_source_criteria_immutable", sha(Path(__file__)) == args.expected_source_sha256
            and sha(args.criteria) == args.expected_criteria_sha256)
        check("producer_source_protocol_immutable", sha(args.producer) == criteria["producer_sha256"]
            and sha(args.producer.with_name("protocol.json")) == criteria["producer_protocol_sha256"])
        check("source9f_after", source_digest(args.repo) == criteria["production_source_sha256"])
        report.update(status="complete", checks_count=len(checks), issues=sum(not c["passed"] for c in checks),
            numerical_record_integrity_pass=all(c["passed"] for c in checks), counts=counters,
            files_before=before, old_input_sha256=old_pins, producer_complete_sha256=criteria["native_complete_sha256"],
            auditor_sha256=sha(Path(__file__)), criteria_sha256=sha(args.criteria))
        write(args.output / "status.json", report)
        write(args.output / "receipt.json", report)
        print(json.dumps(dict(status=report["status"], issues=report["issues"], checks=report["checks_count"], counts=counters)))
        return 0 if report["numerical_record_integrity_pass"] else 1
    except BaseException as exc:
        report.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        write(args.output / "status.json", report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criteria", type=Path, default=Path(__file__).with_name("criteria.json"))
    parser.add_argument("--execute-audit", action="store_true")
    for name in ["repo", "witness", "run", "producer", "output"]:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-criteria-sha256")
    args = parser.parse_args(argv)
    criteria = json.loads(args.criteria.read_text())
    if not args.execute_audit:
        print(json.dumps(dict(status="metadata_only_no_outcome_reads", planned_frequency_rows=64,
            source_sha256=sha(Path(__file__)), criteria_sha256=sha(args.criteria))))
        return 0
    if not all([args.repo, args.witness, args.run, args.producer, args.output, args.expected_source_sha256, args.expected_criteria_sha256]):
        parser.error("explicit paths and exact audit source/criteria SHA required")
    if sha(Path(__file__)) != args.expected_source_sha256 or sha(args.criteria) != args.expected_criteria_sha256:
        raise ValueError("frozen scalar auditor/criteria changed")
    return run(args, criteria)


if __name__ == "__main__":
    raise SystemExit(main())
