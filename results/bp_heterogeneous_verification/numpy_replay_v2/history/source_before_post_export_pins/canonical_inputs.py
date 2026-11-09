"""Exact input-only export/load; no solver imports or target-output decoding."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

KEYS = ("physical_probe_velocity", "direction_parameter_probe_m", "interior_direction", "edge_direction")
SHAPE = (192, 96)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array, np):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing or escaping canonical input path")
    return path


def selected_member(path, key, allowed, np):
    if key not in allowed:
        raise ValueError("export target-output member access is forbidden")
    with np.load(path, allow_pickle=False) as archive:
        return archive[key].copy(order="C")


def validate_arrays(arrays, expected, np):
    if set(arrays) != set(KEYS) or len(arrays) != len(KEYS):
        raise ValueError("canonical array keys must be exactly four declared INPUTS")
    for key in KEYS:
        value = arrays[key]
        if value.shape != SHAPE or value.dtype != np.dtype("<f8") or not np.isfinite(value).all():
            raise ValueError("canonical input shape/dtype/finite guard: " + key)
        if key in KEYS[:2] and not (value > 0).all():
            raise ValueError("canonical velocity/context mass must be positive")
        if key in KEYS[2:] and not np.any(value != 0):
            raise ValueError("canonical direction must be nonzero")
        if array_sha(value, np) != expected[key]:
            raise ValueError("exact canonical input array SHA mismatch: " + key)
    # Values are copied unchanged; no normalization, clipping, or mapping.
    copied = {key:arrays[key].copy(order="C") for key in KEYS}
    if any(array_sha(copied[k], np) != expected[k] for k in KEYS):
        raise ValueError("C-copy altered canonical values")
    return copied


def expected_arrays(binding):
    return dict(physical_probe_velocity=binding["physical_velocity_sha256"],
        direction_parameter_probe_m=binding["actual_physical_m_sha256"],
        interior_direction=binding["witnesses"]["interior"]["direction_sha256"],
        edge_direction=binding["witnesses"]["edge"]["direction_sha256"])


def load_manifest(bundle, expected_sha, plan, expected_plan, exporter_sha):
    manifest_path = contained(bundle, "manifest.json")
    if sha(manifest_path) != expected_sha:
        raise ValueError("canonical manifest freeze changed")
    manifest = json.loads(manifest_path.read_bytes())
    if (manifest.get("schema_version") != 2 or manifest.get("plan_sha256") != expected_plan
            or manifest.get("exporter_sha256") != exporter_sha
            or manifest.get("native_complete_sha256") != plan["native_complete_sha256"]
            or manifest.get("numeric_output_arrays_exported") != 0
            or manifest.get("no_value_regeneration_or_renormalization") is not True
            or len(manifest.get("cases", [])) != len(plan["cases"])):
        raise ValueError("canonical input manifest provenance or scope changed")
    for row, binding in zip(manifest["cases"], plan["cases"], strict=True):
        if (row.get("status") != "complete" or row.get("case") != binding["case"]
                or row.get("native_binding") != binding
                or row.get("array_sha256") != expected_arrays(binding)
                or row.get("artifact") != binding["case"]["id"]+"_inputs.npz"):
            raise ValueError("canonical row differs from pinned export source plan")
    return manifest


def load_case(bundle, manifest, case, np):
    rows = manifest.get("cases", [])
    if [r["case"]["id"] for r in rows] != ["upper_left", "middle_center", "lower_right"]:
        raise ValueError("canonical manifest case order/denominator changed")
    row = next((r for r in rows if r["case"] == case), None)
    if row is None:
        raise ValueError("wrong canonical case/ancestry binding")
    path = contained(bundle, row["artifact"])
    if sha(path) != row["artifact_sha256"]:
        raise ValueError("canonical input container hash mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    if len(names) != len(KEYS) or set(names) != {k+".npy" for k in KEYS}:
        raise ValueError("canonical container has extra/missing/duplicate members")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key:archive[key] for key in KEYS}
    return validate_arrays(arrays, row["array_sha256"], np), row


def export(plan_path, native, output, expected_plan, expected_source):
    import numpy as np

    if sha(plan_path) != expected_plan or sha(__file__) != expected_source:
        raise ValueError("frozen exporter/source plan mismatch")
    plan = json.loads(plan_path.read_bytes())
    native, output = native.resolve(), output.resolve()
    if output.exists() or not output.parent.is_dir() or output.is_relative_to(native):
        raise ValueError("fresh input-only export outside original native run required")
    output.mkdir()
    rows = [dict(case=b["case"], status="unattempted") for b in plan["cases"]]
    write_json(output / "attempt_started.json", dict(plan_sha256=expected_plan,
        exporter_sha256=expected_source, cases=rows, native_solution_outputs_decoded=0))
    try:
        if sha(native / "complete_receipt.json") != plan["native_complete_sha256"]:
            raise ValueError("native aggregate completion changed")
        root = json.loads((native / "complete_receipt.json").read_bytes())
        for binding, row in zip(plan["cases"], rows, strict=True):
            row["status"] = "attempted"
            folder = native / binding["case"]["id"]
            for filename, field in (("status.json", "worker_status_sha256"), ("complete_receipt.json", "worker_complete_sha256")):
                if sha(folder / filename) != binding[field]:
                    raise ValueError("worker status/completion identity changed")
            status = json.loads((folder / "status.json").read_bytes())
            if status != json.loads((folder / "complete_receipt.json").read_bytes()):
                raise ValueError("worker status differs from completed worker")
            call = next(r for r in root["calls"] if r["case_id"] == binding["case"]["id"] and r["call_id"] == "probe_saved")
            if canonical_sha(call) != binding["probe_call_canonical_sha256"]:
                raise ValueError("aggregate-bound probe call changed")
            if next(r for r in status["calls"] if r["call_id"] == "probe_saved") != call:
                raise ValueError("worker probe call differs from aggregate")
            probe = folder / binding["probe_artifact"]
            if sha(probe) != binding["probe_artifact_sha256"]:
                raise ValueError("probe INPUT container changed")
            velocity = selected_member(probe, "actual_physical_velocity_kmps", {"actual_physical_velocity_kmps"}, np)
            arrays, contexts = {"physical_probe_velocity":velocity}, []
            for label in ("interior", "edge"):
                witness = binding["witnesses"][label]
                if status["direction_witnesses"][label] != witness:
                    raise ValueError("worker direction binding changed")
                path = folder / witness["artifact"]
                if sha(path) != witness["artifact_sha256"]:
                    raise ValueError("native input witness container changed")
                context = selected_member(path, "probe_m", {"probe_m", "direction"}, np)
                direction = selected_member(path, "direction", {"probe_m", "direction"}, np)
                if array_sha(context, np) != witness["probe_m_sha256"]:
                    raise ValueError("native context mass input SHA changed")
                if array_sha(direction, np) != witness["direction_sha256"]:
                    raise ValueError("native direction input SHA changed")
                contexts.append(context)
                arrays[label+"_direction"] = direction
            if not np.array_equal(*contexts):
                raise ValueError("two recorded context models differ")
            arrays["direction_parameter_probe_m"] = contexts[0]
            expected = expected_arrays(binding)
            arrays = validate_arrays(arrays, expected, np)
            filename = binding["case"]["id"]+"_inputs.npz"
            with (output / filename).open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            row.update(status="complete", artifact=filename, artifact_sha256=sha(output / filename),
                array_sha256=expected, native_binding=binding)
        result = dict(schema_version=2, role="exact native-derived numeric INPUTS only",
            plan_sha256=expected_plan, exporter_sha256=expected_source,
            native_complete_sha256=plan["native_complete_sha256"], cases=rows,
            units={"physical_probe_velocity":"km/s", "direction_parameter_probe_m":"s^2/km^2",
                "directions":"unit Euclidean norm vectors in physical-m coordinates"},
            array_axes="x,z", array_shape=list(SHAPE), array_dtype="<f8",
            numeric_output_arrays_exported=0, no_value_regeneration_or_renormalization=True,
            physical_acceptance=None)
        write_json(output / "manifest.json", result)
        write_json(output / "export_complete.json", dict(manifest_sha256=sha(output / "manifest.json"),
            cases=3, arrays=12, native_solution_outputs_decoded=0, PDE_calls=0))
        return result
    except BaseException as exc:
        write_json(output / "export_failure.json", dict(error=f"{type(exc).__name__}: {exc}",
            cases=rows, all_prefix_files_preserved=True, native_solution_outputs_decoded=0))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-pinned-inputs", action="store_true")
    for name in ("plan", "native-run", "output"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    if not args.export_pinned_inputs:
        print(json.dumps(dict(status="metadata_only_no_export_or_PDE", keys=KEYS, cases=3)))
        return 0
    if not all((args.plan, args.native_run, args.output, args.expected_plan_sha256, args.expected_source_sha256)):
        parser.error("explicit plan/native path/fresh output and source/plan hashes required")
    export(args.plan, args.native_run, args.output, args.expected_plan_sha256, args.expected_source_sha256)
    print(json.dumps(dict(status="exact_input_export_complete", cases=3, arrays=12, PDE_calls=0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
