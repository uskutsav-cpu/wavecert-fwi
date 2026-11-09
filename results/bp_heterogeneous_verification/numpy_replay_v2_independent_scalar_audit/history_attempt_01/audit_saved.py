"""Independent saved-array BP V2 scalar audit; never imports project numerics."""

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def canonical_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def scalar_norm(value):
    return math.hypot(*(float(x) for x in value.flat))


def scalar_relative(left, right):
    numerator = math.hypot(*(float(x) - float(y) for x, y in zip(left.flat, right.flat, strict=True)))
    denominator = scalar_norm(right)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator > 0 else None,
        "finite": math.isfinite(numerator) and math.isfinite(denominator),
        "zero_denominator": denominator == 0,
    }


def scalar_dot(left, right):
    return math.fsum(float(x) * float(y) for x, y in zip(left.flat, right.flat, strict=True))


def physical_indices(index, size, width):
    if index == 0:
        return range(width + 1)
    if index == size - 1:
        return range(width + size - 1, size + 2 * width)
    return (index + width,)


def scalar_fold(padded, shape, width):
    result = np.empty(shape, dtype=np.float64)
    for x in range(shape[0]):
        for z in range(shape[1]):
            result[x, z] = math.fsum(
                float(padded[px, pz])
                for px in physical_indices(x, shape[0], width)
                for pz in physical_indices(z, shape[1], width)
            )
    return result


def scalar_padded_dot(padded, direction, width):
    rows, cols = direction.shape
    return math.fsum(
        float(padded[x, z])
        * float(direction[max(0, min(rows - 1, x - width)), max(0, min(cols - 1, z - width))])
        for x in range(padded.shape[0])
        for z in range(padded.shape[1])
    )


def audit(args):
    checks = []
    scalar_comparisons = 0
    opened = {}

    def check(label, condition):
        checks.append({"label": label, "pass": bool(condition)})

    def bind(path, expected=None):
        actual = file_sha(path)
        if expected is not None:
            check(str(path) + " SHA", actual == expected)
        opened[str(path)] = actual
        return path

    def read_json(path, expected=None):
        return json.loads(bind(path, expected).read_bytes())

    def arrays(path, expected):
        bind(path, expected)
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def same_scalar(label, actual, saved):
        nonlocal scalar_comparisons
        scalar_comparisons += 1
        check(label + " exact saved scalar", actual == saved)

    p = read_json(args.stage / "protocol.json")
    old = read_json(args.stage / "history/v1_c51_f0/protocol.json")
    core_path = bind(args.stage / "numpy_bp_tangent.py", p["core_sha256"])
    check("kernel bytes equal V1 core pin", file_sha(core_path) == old["core_sha256"])
    for key in ("criteria", "parameters", "inputs", "probe_scale", "directions", "physical_shape_x_z", "own_forward_solves", "direction_checks"):
        check("V1 numerical contract unchanged " + key, p[key] == old[key])
    check("fixed criteria", p["criteria"] == {
        "trace_relative_L2": 1e-9, "gradient_relative_error": 1e-3,
        "source_relative_L2": 1e-14, "direction_relative_L2": 1e-14,
        "fold_relative_L2": 1e-13, "near_zero_relative_normscale": 1e-8,
    })
    plan = read_json(args.stage / "EXPORT_SOURCE_PLAN.json", p["export_source_plan_sha256"])
    bind(args.stage / "canonical_inputs.py", p["canonical_input_source_sha256"])
    runner = bind(args.stage / "run_replay.py")
    receipt = read_json(args.run / "complete_receipt.json", args.expected_complete)
    check("status equals complete", (args.run / "status.json").read_bytes() == (args.run / "complete_receipt.json").read_bytes())
    check("runner source matches receipt", file_sha(runner) == receipt["source_sha256"])
    check("protocol matches receipt", file_sha(args.stage / "protocol.json") == receipt["protocol_sha256"])
    check("complete full denominator", receipt["status"] == "complete_comparison" and receipt["numerical_agreement_pass"] is True and receipt["full3_case_6_direction_denominators_retained"] is True)
    check("three solves six directions", receipt["own_solves_planned"] == 3 and receipt["directions_planned"] == 6 and len(receipt["cases"]) == 3)
    for key, expected in (("native_derived_numeric_inputs_used", True), ("native_solution_outputs_used_during_own_solves", False), ("all3_own_complete_before_native_solution_outputs", True), ("native_solution_output_arrays_opened", True), ("physical_acceptance", None)):
        check("root provenance " + key, receipt[key] is expected)
    manifest = read_json(args.inputs / "manifest.json", receipt["canonical_input_manifest_sha256"])
    check("manifest/plan provenance", manifest["plan_sha256"] == p["export_source_plan_sha256"] and manifest["exporter_sha256"] == p["canonical_input_source_sha256"] and manifest["native_complete_sha256"] == p["native_complete_sha256"])
    check("manifest inputs only", manifest["numeric_output_arrays_exported"] == 0 and manifest["no_value_regeneration_or_renormalization"] is True and manifest["physical_acceptance"] is None)
    native = read_json(args.native / "complete_receipt.json", p["native_complete_sha256"])
    check("native complete57 physical acceptance null", native["status"] == "complete_guard_attempts" and len(native["calls"]) == 57 and native["physical_acceptance"] is None)
    check("ordered cases", [row["case_id"] for row in receipt["cases"]] == [c["id"] for c in p["inputs"]] == [row["case"]["id"] for row in manifest["cases"]])
    summary = []
    keys = {"physical_probe_velocity", "direction_parameter_probe_m", "interior_direction", "edge_direction"}
    for case, row, inp, binding in zip(p["inputs"], receipt["cases"], manifest["cases"], plan["cases"], strict=True):
        name = case["id"]
        check(name + " manifest source binding", inp["case"] == case and inp["native_binding"] == binding and binding["case"] == case)
        own = arrays(args.run / row["artifact"], row["artifact_sha256"])
        canonical = arrays(args.inputs / inp["artifact"], inp["artifact_sha256"])
        with zipfile.ZipFile(args.inputs / inp["artifact"]) as archive:
            names = archive.namelist()
        check(name + " four input-only ZIP members", len(names) == 4 and set(names) == {key + ".npy" for key in keys})
        check(name + " four canonical input keys", set(canonical) == keys)
        record = row["record"]
        check(name + " complete own and comparison", row["status"] == "complete_own_solve" and row["comparison_status"] == "complete")
        check(name + " canonical SHA records", record["canonical_input_array_sha256"] == inp["array_sha256"] and record["canonical_input_artifact_sha256"] == inp["artifact_sha256"])
        for key, expected in (("kernel_opens_native_files", False), ("native_derived_numeric_inputs_used", True), ("native_solution_outputs_used_during_own_solve", False), ("full_wavefield_history_saved", False)):
            check(name + " scoped provenance " + key, record[key] is expected)
        check(name + " broad obsolete flag absent", "native_numeric_witnesses_used" not in record)
        for key, own_key in (("physical_probe_velocity", "physical_probe_velocity"), ("direction_parameter_probe_m", "physical_probe_m"), ("interior_direction", "interior_direction"), ("edge_direction", "edge_direction")):
            check(name + " exact input " + key, canonical[key].shape == (192, 96) and canonical[key].dtype == np.dtype("<f8") and np.all(np.isfinite(canonical[key])) and array_sha(canonical[key]) == inp["array_sha256"][key] and np.array_equal(canonical[key], own[own_key]))
        folder = args.native / name
        worker = read_json(folder / "status.json", binding["worker_status_sha256"])
        worker_complete = read_json(folder / "complete_receipt.json", binding["worker_complete_sha256"])
        check(name + " worker identical and witnesses bound", worker == worker_complete and worker["case"] == case and worker["direction_witnesses"] == binding["witnesses"])
        calls = {r["call_id"]: r for r in worker["calls"]}
        for call_id in ("probe_saved", "probe_adjoint", "coarse40"):
            matching = [r for r in native["calls"] if r["case_id"] == name and r["call_id"] == call_id]
            check(name + " aggregate call " + call_id, len(matching) == 1 and matching[0] == calls[call_id] and calls[call_id]["status"] == "complete")
        check(name + " source identity before-after", worker["source_before"] == worker["source_after"])
        check(name + " source digest", worker["source_before"]["production"] == p["production_source_sha256"])
        probe_record = calls["probe_saved"]["result"]
        check(name + " canonical probe call", canonical_sha(calls["probe_saved"]) == binding["probe_call_canonical_sha256"])
        probe = arrays(folder / probe_record["artifact"], probe_record["artifact_sha256"])
        gradient_record = calls["probe_adjoint"]["result"]
        gradient = arrays(folder / gradient_record["artifact"], gradient_record["artifact_sha256"])
        observed_record = calls["coarse40"]["result"]
        observed = arrays(folder / observed_record["artifact"], observed_record["artifact_sha256"])
        check(name + " exact native velocity/context", np.array_equal(own["physical_probe_velocity"], probe["actual_physical_velocity_kmps"]) and np.array_equal(own["physical_probe_m"], probe["actual_physical_m"]))
        check(name + " exact model SHA", record["physical_velocity_sha256"] == array_sha(probe["actual_physical_velocity_kmps"]) == binding["physical_velocity_sha256"] and record["physical_m_sha256"] == array_sha(probe["actual_physical_m"]) == binding["actual_physical_m_sha256"])
        mask = own["computed_mask"]
        check(name + " time/mask identity", np.array_equal(own["times_ms"], probe["times_ms"]) and np.array_equal(own["times_ms"], observed["times_ms"]) and np.array_equal(mask, probe["computed_mask"]) and np.array_equal(mask, observed["computed_mask"]))
        check(name + " fixed receiver/mask dimensions", own["receiver"].shape == (2001, 10) and mask.dtype == np.bool_ and mask.shape == (2001,) and not mask[0] and not mask[-1] and np.all(mask[1:-1]))
        check(name + " fixed actual bounds", record["time_bounds"] == probe_record["time_bounds"] == gradient_record["actual_time_bounds"] == {"time_m": 1, "time_M": 1999})
        check(name + " held source bitwise", np.array_equal(own["source"], probe["source_before"]) and np.array_equal(probe["source_before"], probe["source_after"]))
        residual = np.empty_like(gradient["residual"])
        for t in range(2001):
            for receiver in range(10):
                residual[t, receiver] = float(probe["receiver"][t, receiver]) - float(observed["receiver"][t, receiver]) if mask[t] else 0.0
        check(name + " residual definition exact", np.array_equal(residual, gradient["residual"]))
        trace = scalar_relative(own["receiver"][mask], probe["receiver"][mask])
        source = scalar_relative(own["source"], probe["source_before"])
        folded = scalar_fold(gradient["padded_gradient"], (192, 96), 40)
        fold = scalar_relative(folded, gradient["production_fold"])
        comparison = row["comparison"]
        for label, actual in (("trace", trace), ("source_relative_L2", source), ("padded_fold_relative_L2", fold)):
            for key, value in actual.items():
                same_scalar(name + " " + label + " " + key, value, comparison[label][key])
        check(name + " trace gate", trace["value"] is not None and trace["value"] <= p["criteria"]["trace_relative_L2"] and comparison["trace_agreement"] is True)
        check(name + " source/fold gates", source["value"] <= p["criteria"]["source_relative_L2"] and fold["value"] <= p["criteria"]["fold_relative_L2"])
        derivatives = []
        check(name + " two fixed direction labels", [d["direction"] for d in row["directions"]] == ["interior", "edge"])
        for index, label in enumerate(("interior", "edge")):
            saved = row["directions"][index]
            witness_record = binding["witnesses"][label]
            witness = arrays(folder / witness_record["artifact"], witness_record["artifact_sha256"])
            direction = own[label + "_direction"]
            check(name + " native direction/context exact " + label, np.array_equal(direction, witness["direction"]) and np.array_equal(own["physical_probe_m"], witness["probe_m"]) and array_sha(direction) == witness_record["direction_sha256"] and array_sha(witness["probe_m"]) == witness_record["probe_m_sha256"])
            direction_check = scalar_relative(direction, witness["direction"])
            for key, value in direction_check.items():
                same_scalar(name + " " + label + " direction " + key, value, saved["direction_relative_L2"][key])
            check(name + " unchanged direction gate " + label, direction_check["value"] is not None and direction_check["value"] <= p["criteria"]["direction_relative_L2"])
            tangent_dd = scalar_dot(residual[mask], own[label + "_receiver_tangent"][mask])
            native_dd = scalar_dot(gradient["production_fold"], direction)
            padded_dd = scalar_padded_dot(gradient["padded_gradient"], direction, 40)
            naive_dd = scalar_dot(gradient["naive_crop"], direction)
            normscale = scalar_norm(gradient["production_fold"]) * scalar_norm(direction)
            informative = normscale > 0 and abs(native_dd) > p["criteria"]["near_zero_relative_normscale"] * normscale
            error = abs(tangent_dd - native_dd)
            relative = error / max(abs(tangent_dd), abs(native_dd)) if informative else None
            computed = {"independent_tangent_DD": tangent_dd, "native_folded_gradient_DD": native_dd, "padded_gradient_DD": padded_dd, "naive_crop_DD": naive_dd, "gradient_direction_normscale": normscale, "informative": informative, "absolute_error": error, "relative_error": relative}
            for key, value in computed.items():
                same_scalar(name + " " + label + " " + key, value, saved[key])
            check(name + " fixed informative DD gate " + label, informative and relative is not None and relative <= p["criteria"]["gradient_relative_error"] and saved["agreement"] is True and saved["status"] == "complete")
            check(name + " repeated DD record consistent " + label, {k: v for k, v in saved.items() if k != "status"} == comparison["derivatives"][index])
            derivatives.append({"direction": label, **computed})
        check(name + " physical scope null", comparison["native_physical_acceptance"] is None and worker["physical_acceptance"] is None)
        summary.append({"case": name, "trace_relative_L2": trace["value"], "source_relative_L2": source["value"], "fold_relative_L2": fold["value"], "derivatives": derivatives})
    check("all bound files unchanged after audit", all(file_sha(Path(path)) == value for path, value in opened.items()))
    issues = [row for row in checks if not row["pass"]]
    report = {
        "status": "PASS" if not issues else "FAIL", "checks": len(checks), "exact_scalar_comparisons": scalar_comparisons,
        "issues": issues, "cases": summary, "bound_file_sha256": opened,
        "source_sha256": file_sha(Path(__file__)),
        "scope": "Saved-array/source-and-scalar audit only; no producer numeric imports, PDE/adjoint/tangent rerun, full-state-hash reconstruction, or independent operating-system I/O trace.",
        "phase_order_evidence": "Read reviewed runner/exporter source directly: four native-derived input arrays are loaded before propagation; target comparisons follow the all-three-complete gate. Saved flags agree. This audit alone does not prove runtime I/O chronology.",
        "independence_limits": "Shares NumPy decoding and actual native-derived prescribed inputs; scalar sqrt is not needed here. Scalar fsum/hypot arithmetic is independently written; not blind human reproduction, realistic physics acceptance, inverse recovery, new geology, or novelty.",
        "physical_acceptance": None,
        "prior_97f_failure_retroactively_passed": False,
        "checks_detail": checks,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(checks), "exact_scalar_comparisons": scalar_comparisons, "issues": len(issues), "report_sha256": file_sha(args.output)}))
    return 0 if not issues else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "run", "inputs", "native", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-complete", required=True)
    return audit(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
