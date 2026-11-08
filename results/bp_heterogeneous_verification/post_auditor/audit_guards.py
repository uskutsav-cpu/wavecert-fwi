"""Independent retained-record audit. Default CLI reads no BP/run arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def scalar_norm(values):
    return math.sqrt(math.fsum(float(v) * float(v) for v in np.asarray(values).ravel()))


def scalar_dot(a, b):
    return math.fsum(float(x) * float(y) for x, y in zip(np.asarray(a).ravel(), np.asarray(b).ravel(), strict=True))


def ratio(error, reference):
    a, b = scalar_norm(error), scalar_norm(reference)
    return dict(numerator=a, denominator=b, ratio=a / b if b else None, zero_denominator=b == 0,
                finite=math.isfinite(a) and math.isfinite(b))


def objective(candidate, observed, mask):
    return .5 * scalar_dot(np.asarray(candidate)[mask] - np.asarray(observed)[mask],
                           np.asarray(candidate)[mask] - np.asarray(observed)[mask])


def independent_fold(padded, shape, width):
    """Each physical edge bucket is summed with math.fsum, not producer np.add.at."""
    result = np.empty(shape, dtype=np.float64)
    def indices(index, size, total):
        if index == 0:
            return range(width + 1)
        if index == size - 1:
            return range(width + size - 1, total)
        return [width + index]
    for x in range(shape[0]):
        for z in range(shape[1]):
            result[x, z] = math.fsum(float(padded[i, j]) for i in indices(x, shape[0], padded.shape[0])
                                    for j in indices(z, shape[1], padded.shape[1]))
    return result


def bilinear_samples(coarse):
    """Independent scalar evaluation of the same coarse-node bilinear field."""
    out = np.empty(tuple(2 * n - 1 for n in coarse.shape), dtype=np.float64)
    for x in range(out.shape[0]):
        for z in range(out.shape[1]):
            ix, iz = x // 2, z // 2
            xs, zs = [ix] if x % 2 == 0 else [ix, ix + 1], [iz] if z % 2 == 0 else [iz, iz + 1]
            out[x, z] = math.fsum(float(coarse[i, j]) for i in xs for j in zs) / (len(xs) * len(zs))
    return out


def common_comparison(a, b):
    lookup = {float(t): i for i, t in enumerate(b["times_ms"]) if b["computed_mask"][i]}
    pairs = [(i, lookup[float(t)]) for i, t in enumerate(a["times_ms"])
             if a["computed_mask"][i] and float(t) in lookup]
    if not pairs:
        raise ValueError("no exact common computed times")
    ia, ib = np.asarray(pairs).T
    if not np.array_equal(a["times_ms"][ia], b["times_ms"][ib]):
        raise ValueError("time alignment changed")
    error, ref = a["receiver"][ia] - b["receiver"][ib], b["receiver"][ib]
    return dict(aggregate=ratio(error, ref), per_receiver=[ratio(error[:, i], ref[:, i]) for i in range(ref.shape[1])],
        exact_common_samples=len(pairs), first_time_ms=float(a["times_ms"][ia[0]]),
        last_time_ms=float(a["times_ms"][ia[-1]]), physical_acceptance=None)


def planned(protocol, case=None):
    rows = []
    for c in protocol["inputs"]:
        if case is not None and c["id"] != case:
            continue
        ids = [v["id"] for v in protocol["forward_calls"]] + ["probe_saved", "probe_adjoint"]
        ids += [f"fd_{d}_{s}_{sign}" for d in ["interior", "edge"]
                for s in protocol["gradient"]["relative_steps"] for sign in ["plus", "minus"]]
        rows.extend((c["id"], call_id) for call_id in ids)
    return rows


def source_digest(repo):
    files = [repo / "pyproject.toml"] + list((repo / "src").rglob("*.py"))
    files += [p for p in (repo / "configs").rglob("*") if p.is_file()]
    files += [p for p in (repo / "scripts").rglob("*") if p.suffix in {".py", ".sh"}]
    files += list((repo / ".github/workflows").glob("*.yml")) + list(repo.glob("requirements*.txt"))
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        if path.is_file():
            digest.update(path.relative_to(repo).as_posix().encode() + b"\0" + checksum(path).encode() + b"\0")
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(clean(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class Audit:
    def __init__(self, criteria):
        self.criteria, self.checks = criteria, []

    def check(self, label, condition, **detail):
        self.checks.append(dict(check=label, passed=bool(condition), **clean(detail)))
        return bool(condition)

    def near(self, label, actual, expected, scale=None):
        valid = actual is not None and expected is not None
        if valid:
            valid = math.isfinite(float(actual)) and math.isfinite(float(expected))
        error = abs(float(actual) - float(expected)) if valid else None
        bound = self.criteria["scalar_report_absolute_tolerance"]
        if valid:
            bound += self.criteria["scalar_report_relative_tolerance"] * (abs(expected) if scale is None else abs(scale))
        return self.check(label, valid and error <= bound, observed=actual, expected=expected, absolute_error=error, bound=bound)

    def numeric_record(self, label, reported, recomputed):
        for key in ["numerator", "denominator", "ratio"]:
            if reported.get(key) is None or recomputed.get(key) is None:
                self.check(label + ":" + key, reported.get(key) is recomputed.get(key), reported=reported.get(key), recomputed=recomputed.get(key))
            else:
                self.near(label + ":" + key, reported[key], recomputed[key])
        for key in ["finite", "zero_denominator"]:
            self.check(label + ":" + key, reported.get(key) == recomputed.get(key))


def decode(path):
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {key: np.array(bundle[key], copy=True) for key in bundle.files}
    if any(a.ndim > 2 for a in arrays.values()):
        raise ValueError("unexpected wavefield-history artifact")
    return arrays


def model_checks(audit, label, record, arrays, expected, nbl, spacing):
    vp, model = arrays["actual_physical_velocity_kmps"], arrays["actual_physical_m"]
    audit.check(label + ":model_shape_dtype_positive", vp.shape == expected.shape and vp.dtype == np.float64
        and model.shape == vp.shape and np.isfinite(vp).all() and (vp > 0).all())
    audit.check(label + ":declared_velocity_field", np.allclose(vp, expected, rtol=audit.criteria["model_interpolation_relative_tolerance"], atol=0))
    audit.check(label + ":native_m_mapping", np.array_equal(model, 1 / vp ** 2))
    padded = np.pad(vp, nbl, mode="edge")
    expected_hashes = dict(physical_velocity_sha256=array_sha(vp), padded_velocity_sha256=array_sha(padded),
                          actual_physical_m_sha256=array_sha(model), actual_padded_m_sha256=array_sha(1 / padded ** 2))
    for key, value in expected_hashes.items():
        audit.check(label + ":" + key, record[key] == value)
    audit.check(label + ":grid_metadata", record["shape_x_z"] == list(vp.shape)
        and record["padded_shape"] == list(padded.shape) and record["grid_origin_m"] == [-nbl * spacing] * 2)
    audit.check(label + ":CFL", record["maximum_velocity_kmps"] <= 5. and 0 < record["dt_ms"] <= record["critical_dt_ms"])


def forward_checks(audit, label, arrays, record, expected, config, protocol):
    dt, scale, h = config["dt_ms"], config["source_scale"], config["spacing_m"]
    times = np.arange(round(1000 / dt) + 1, dtype=np.float64) * dt
    mask = np.arange(len(times))
    mask = (mask >= 1) & (mask <= len(times) - 2)
    audit.check(label + ":sampling", np.array_equal(arrays["times_ms"], times)
        and np.array_equal(arrays["computed_mask"], mask) and arrays["computed_mask"].dtype == bool)
    values = arrays["receiver"]
    audit.check(label + ":finite_nonzero_raw_record", values.shape == (len(times), 10)
        and values.dtype == np.float64 and np.isfinite(values).all() and scalar_norm(values[mask]) > 0)
    audit.check(label + ":raw_endpoint_sentinels", np.all(values[[0, -1]] == 0)
        and record["time_bounds"] == dict(time_m=1, time_M=len(times) - 2))
    source = scale * (1 - 2 * (np.pi * .005 * (times - 200)) ** 2) * np.exp(-(np.pi * .005 * (times - 200)) ** 2)
    audit.check(label + ":source_arrays", arrays["source_before"].shape == (len(times), 1)
        and arrays["source_before"].dtype == np.float64
        and np.array_equal(arrays["source_before"], arrays["source_after"])
        and np.array_equal(arrays["source_times_ms"], times) and np.array_equal(arrays["source_times_after_ms"], times))
    source_difference = ratio(arrays["source_before"][:, 0] - source, source)
    audit.check(label + ":source_excitation", source_difference["ratio"] is not None
        and source_difference["ratio"] <= audit.criteria["source_relative_l2_tolerance"], diagnostic=source_difference)
    sc = np.asarray([protocol["geometry"]["source_local_x_z_m"]], dtype=np.float64)
    rc = np.column_stack((np.asarray(protocol["geometry"]["receiver_coarse_x_nodes"]) * 12.5, np.full(10, 200.)))
    audit.check(label + ":coordinates", np.array_equal(arrays["source_coordinates"], sc)
        and np.array_equal(arrays["source_coordinates_after"], sc) and np.array_equal(arrays["receiver_coordinates"], rc))
    hashes = dict(source_sha256=array_sha(arrays["source_before"]), time_sha256=array_sha(times), receiver_sha256=array_sha(values),
        source_time_after_sha256=array_sha(times), source_coordinates_sha256=array_sha(sc),
        source_coordinates_after_sha256=array_sha(sc), receiver_coordinates_sha256=array_sha(rc))
    for key, value in hashes.items():
        audit.check(label + ":" + key, record[key] == value)
    audit.check(label + ":forcing_strength", record["source_scale"] == scale
        and record["physical_point_strength_m2"] == scale * h * h == 156.25)
    origin = np.asarray([-config["nbl"] * h] * 2)
    audit.check(label + ":computational_nodes", record["computational_source_indices"] == ((sc - origin) / h).tolist()
        and record["computational_receiver_indices"] == ((rc - origin) / h).tolist())
    saved = label.endswith("probe_saved")
    expected_layout = [len(times) if saved else 3] + [n + 2 * config["nbl"] + 8 for n in expected.shape]
    audit.check(label + ":bounded_wavefield_metadata", record["save_wavefield"] is saved
        and record["allocated_wavefield_shape"] == record["materialized_wavefield_shape"] == expected_layout
        and record["allocated_wavefield_bytes"] == record["materialized_wavefield_bytes"]
        == math.prod(expected_layout) * 8
        and (record["allocated_wavefield_bytes"] == 824732160 if saved else record["allocated_wavefield_shape"][0] == 3))
    model_checks(audit, label, record["model"], arrays, expected, config["nbl"], h)


def execute(args, criteria):
    producer, protocol_path = args.producer, args.producer.with_name("protocol.json")
    if checksum(producer) != criteria["producer_sha256"] or checksum(protocol_path) != criteria["protocol_sha256"]:
        raise ValueError("frozen producer/protocol changed")
    protocol = json.loads(protocol_path.read_text())
    if checksum(producer.with_name("guard_math.py")) != criteria["helper_sha256"]:
        raise ValueError("frozen producer helper changed")
    if checksum(args.terminal_receipt) != args.expected_terminal_sha256:
        raise ValueError("explicit root terminal-evidence hash mismatch")
    if args.output.exists():
        raise FileExistsError("new audit output required")
    args.output.mkdir()
    audit = Audit(criteria)
    report = dict(status="running", physical_acceptance=None, one_ancestor=True,
        root_terminal_evidence_sha256=checksum(args.terminal_receipt), planned_calls=57,
        call_inventory=[dict(case_id=c, call_id=i, native_status="not_read") for c, i in planned(protocol)],
        cases=[dict(case_id=c["id"], status="unattempted", planned_derivatives=[
            dict(direction=d, relative_step=s, audit_status="not_read") for d in ["interior", "edge"]
            for s in protocol["gradient"]["relative_steps"]]) for c in protocol["inputs"]], checks=audit.checks)
    def snapshot():
        return {p.relative_to(args.run).as_posix(): checksum(p) for p in sorted(args.run.rglob("*")) if p.is_file()}
    before = snapshot()
    report["run_files_before"] = before
    write(args.output / "status.json", report)
    try:
        root = json.loads((args.run / "status.json").read_text())
        complete_path = args.run / "complete_receipt.json"
        audit.check("root_complete_receipt_exists", complete_path.is_file())
        if complete_path.is_file():
            audit.check("root_complete_receipt_matches_status", root == json.loads(complete_path.read_text()))
        for inventory in report["call_inventory"]:
            matching = [r for r in root["calls"] if r["case_id"] == inventory["case_id"] and r["call_id"] == inventory["call_id"]]
            inventory.update(native_status=matching[0]["status"] if len(matching) == 1 else "missing_or_duplicate",
                             error=matching[0].get("error") if len(matching) == 1 else None)
        expected = planned(protocol)
        actual = [(r["case_id"], r["call_id"]) for r in root["calls"]]
        audit.check("all57_unique_calls", len(actual) == len(set(actual)) == 57 and set(actual) == set(expected))
        audit.check("all57_completed_not_only_processes", all(r["status"] == "complete" for r in root["calls"]))
        audit.check("root_completion_status", root.get("status") == "complete_guard_attempts")
        audit.check("all3_workers_complete", len(root["case_processes"]) == 3
            and {r["case_id"] for r in root["case_processes"]} == {c["id"] for c in protocol["inputs"]}
            and all(r["status"] == "complete" and r["exit_code"] == 0 for r in root["case_processes"]))
        audit.check("root_worker_aggregate_binding", root.get("all_case_workers_complete") ==
            all(r["status"] == "complete" and r["exit_code"] == 0 for r in root["case_processes"]))
        audit.check("root_frozen_identity", root["source_sha256"] == criteria["producer_sha256"]
                    and root["protocol_sha256"] == criteria["protocol_sha256"] and root["physical_acceptance"] is None)
        current_source = source_digest(args.repo)
        audit.check("current_production9f", current_source == criteria["production_source_sha256"])
        primary = {"devito_backend.py": args.repo / "src/wavecert/physics/devito_backend.py"}
        primary.update({name: args.primary_source_root / relative for name, relative in
            {"model.py": "model.py", "operators.py": "acoustic/operators.py", "wavesolver.py": "acoustic/wavesolver.py",
             "source.py": "source.py", "utils.py": "utils.py"}.items()})
        for name, path in primary.items():
            audit.check("current_primary:" + name, checksum(path) == protocol["source_pins"][name])
        for case, case_report in zip(protocol["inputs"], report["cases"], strict=True):
            if not complete_path.is_file():
                case_report["status"] = "metadata_only_incomplete_native_receipt"
                child_status = args.run / case["id"] / "status.json"
                if child_status.is_file():
                    child_metadata = json.loads(child_status.read_text())
                    case_report["native_status"] = child_metadata.get("status")
                    case_report["native_call_states"] = [{k: row.get(k) for k in ["call_id", "status", "error", "restoration"]}
                                                         for row in child_metadata.get("calls", [])]
                case_report["retained_record_integrity_pass"] = False
                continue
            case_report["status"] = "attempted"
            write(args.output / "status.json", report)
            check_start = len(audit.checks)
            try:
                audit_case(args, protocol, criteria, audit, case, case_report, root)
                case_report["status"] = "records_examined"
            except Exception as exc:
                case_report.update(status="audit_failed", error=f"{type(exc).__name__}: {exc}")
                audit.check(case["id"] + ":audit_exception", False, error=case_report["error"])
            case_report["retained_record_integrity_pass"] = all(r["passed"] for r in audit.checks[check_start:])
            write(args.output / "status.json", report)
        report["run_files_after"] = snapshot()
        audit.check("immutable_native_artifacts", report["run_files_after"] == before)
        audit.check("production9f_after", source_digest(args.repo) == criteria["production_source_sha256"])
        report.update(status="complete", integrity_checks_pass=all(r["passed"] for r in audit.checks),
            checks_count=len(audit.checks), failed_checks=sum(not r["passed"] for r in audit.checks),
            scope=criteria["scope"])
        write(args.output / "status.json", report)
        write(args.output / "receipt.json", report)
        return 0 if report["integrity_checks_pass"] else 1
    except BaseException as exc:
        report.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        write(args.output / "status.json", report)
        raise


def audit_case(args, protocol, criteria, audit, case, report, root):
    cid, directory = case["id"], args.run / case["id"]
    status = json.loads((directory / "status.json").read_text())
    report["native_status"] = status["status"]
    report["native_call_states"] = [{k: r.get(k) for k in ["call_id", "status", "error", "restoration"]} for r in status["calls"]]
    complete_path = directory / "complete_receipt.json"
    audit.check(cid + ":worker_receipt_exists", complete_path.is_file())
    if complete_path.is_file():
        audit.check(cid + ":worker_receipt", json.loads(complete_path.read_text()) == status)
    audit.check(cid + ":scope_metadata", status["case"] == case and status["physical_acceptance"] is None
        and status.get("no_physical_acceptance_claim") is True and status.get("one_ancestor") is True)
    audit.check(cid + ":worker_completion_status", status["status"] == "complete_technical_diagnostics")
    ids = [r["call_id"] for r in status["calls"]]
    audit.check(cid + ":all19_call_denominator", len(ids) == len(set(ids)) == 19 and set(ids) == {v for _, v in planned(protocol, cid)})
    rows = {r["call_id"]: r for r in status["calls"]}
    report["all_native_calls_completed"] = all(r["status"] == "complete" for r in status["calls"])
    declared = {r.get("result", r.get("retained_gradient_record", {})).get("artifact") for r in status["calls"]}
    declared |= {r["artifact"] for r in status.get("direction_witnesses", {}).values()}
    declared.discard(None)
    audit.check(cid + ":complete_artifact_inventory", {p.name for p in directory.glob("*.npz")} == declared)
    for row in status["calls"]:
        audit.check(cid + ":call_status:" + row["call_id"], row["status"] == "complete")
        parent = next((r for r in root["calls"] if r["case_id"] == cid and r["call_id"] == row["call_id"]), None)
        audit.check(cid + ":parent_child_binding:" + row["call_id"], parent == row)
    raw_path = args.inputs / case["file"]
    input_before = checksum(raw_path)
    audit.check(cid + ":input_identity", input_before == case["sha256"])
    raw = np.load(raw_path, allow_pickle=False)
    audit.check(cid + ":input_schema", raw.shape == (96, 192) and raw.dtype == np.float32 and np.isfinite(raw).all() and (raw > 0).all())
    coarse = np.asarray(raw.T, dtype=np.float64) / 1000
    fine, probe = bilinear_samples(coarse), .99 * coarse
    for phase in ["source_before", "source_after"]:
        pin = status[phase]
        audit.check(cid + ":" + phase, pin["production"] == criteria["production_source_sha256"]
            and pin["input_sha256"] == case["sha256"] and pin["protocol_sha256"] == criteria["protocol_sha256"]
            and pin["driver_sha256"] == criteria["producer_sha256"] and pin["helper_sha256"] == criteria["helper_sha256"]
            and pin["installed"] == protocol["source_pins"])
    audit.check(cid + ":before_after_pins", status["source_before"] == status["source_after"])
    origin = np.asarray(case["global_origin_z_x_m"])[::-1]
    geom = status["geometry"]
    audit.check(cid + ":artificial_global_geometry", geom["global_crop_origin_x_z_m"] == origin.tolist()
        and geom["global_source_x_z_m"] == (np.asarray(geom["local_source_x_z_m"]) + origin).tolist()
        and geom["global_receivers_x_z_m"] == (np.asarray(geom["local_receivers_x_z_m"]) + origin).tolist())
    records, arrays = {}, {}
    configurations = {v["id"]: v for v in protocol["forward_calls"]}
    configurations["probe_saved"] = configurations["coarse40"]
    for call_id in [v["id"] for v in protocol["forward_calls"]] + ["probe_saved"]:
        row = rows[call_id]
        if row["status"] != "complete" or "result" not in row:
            continue
        record = row["result"]
        path = directory / record["artifact"]
        audit.check(cid + ":artifact:" + call_id, checksum(path) == record["artifact_sha256"])
        values = decode(path)
        target = probe if call_id == "probe_saved" else fine if call_id == "fine160" else coarse
        forward_checks(audit, cid + ":" + call_id, values, record, target, configurations[call_id], protocol)
        records[call_id], arrays[call_id] = record, values
    comparison_rows = []
    for a, b in protocol["forward_comparisons"]["boundary"] + [protocol["forward_comparisons"]["space"], protocol["forward_comparisons"]["time"]]:
        key = a + "__" + b
        if a not in arrays or b not in arrays:
            comparison_rows.append(dict(pair=key, status="unavailable"))
            continue
        value = common_comparison(arrays[a], arrays[b])
        reported = status["forward_comparisons"][key]
        audit.numeric_record(cid + ":comparison:" + key, reported["aggregate"], value["aggregate"])
        for i in range(10):
            audit.numeric_record(cid + f":comparison:{key}:receiver{i}", reported["per_receiver"][i], value["per_receiver"][i])
        audit.check(cid + ":comparison_mask:" + key, all(reported[k] == value[k] for k in
            ["exact_common_samples", "first_time_ms", "last_time_ms", "physical_acceptance"]))
        comparison_rows.append(dict(pair=key, status="recomputed", **value))
    report["forward_comparisons_scalar"] = comparison_rows
    audit_gradient(directory, cid, rows, status, arrays, probe, configurations["coarse40"], protocol, criteria, audit, report)
    audit.check(cid + ":input_immutable_after", checksum(raw_path) == input_before)


def audit_gradient(directory, cid, rows, status, arrays, probe, config, protocol, criteria, audit, report):
    expected_steps = {(d, s) for d in ["interior", "edge"] for s in protocol["gradient"]["relative_steps"]}
    actual_steps = [(r["direction"], r["relative_step"]) for r in status["derivative_checks"]]
    report["native_derivative_rows"] = status["derivative_checks"]
    audit.check(cid + ":all6_direction_steps", len(actual_steps) == len(set(actual_steps)) == 6 and set(actual_steps) == expected_steps)
    observed, predicted = arrays["coarse40"], arrays["probe_saved"]
    mask = predicted["computed_mask"]
    audit.check(cid + ":same_observed_probe_source_time", np.array_equal(observed["source_before"], predicted["source_before"])
        and np.array_equal(observed["times_ms"], predicted["times_ms"]) and np.array_equal(observed["computed_mask"], mask))
    row = rows["probe_adjoint"]
    gradient_record = row.get("result", row.get("retained_gradient_record"))
    path = directory / gradient_record["artifact"]
    audit.check(cid + ":gradient_artifact", checksum(path) == gradient_record["artifact_sha256"])
    g = decode(path)
    residual = predicted["receiver"] - observed["receiver"]
    residual[~mask] = 0
    audit.check(cid + ":residual_binding", np.array_equal(residual, g["residual"]) and scalar_norm(residual) > 0)
    padded = g["padded_gradient"]
    audit.check(cid + ":padded_gradient_binding", padded.shape == (272, 176) and np.isfinite(padded).all()
        and array_sha(padded) == gradient_record["padded_gradient_sha256"])
    audit.check(cid + ":actual_adjoint_bounds", gradient_record["actual_time_bounds"] == row["actual_time_bounds"]
        == dict(time_m=1, time_M=len(mask) - 2) and gradient_record["adjoint_rolling_allocated_shape"] == [3, 280, 184])
    independent = independent_fold(padded, probe.shape, 40)
    for key in ["production_fold", "independent_fold"]:
        error = ratio(g[key] - independent, independent)
        agrees = error["numerator"] == 0 if error["denominator"] == 0 else error["ratio"] <= criteria["pullback_relative_l2_tolerance"]
        audit.check(cid + ":scalar_padding_pullback:" + key, agrees, diagnostic=error)
    audit.check(cid + ":naive_crop_binding", np.array_equal(g["naive_crop"], padded[40:-40, 40:-40]))
    model, derivative_rows, middle = 1 / probe ** 2, [], []
    for direction in ["interior", "edge"]:
        witness = status["direction_witnesses"][direction]
        path = directory / witness["artifact"]
        audit.check(cid + ":direction_artifact:" + direction, checksum(path) == witness["artifact_sha256"])
        d = decode(path)
        active = np.zeros(model.shape, dtype=bool)
        if direction == "interior":
            active[4:-4, 4:-4] = True
        else:
            active[0] = active[-1] = True
            active[:, 0] = active[:, -1] = True
        scale = scalar_norm(model * active)
        audit.check(cid + ":direction_binding:" + direction, np.array_equal(d["mask"], active)
            and np.array_equal(d["probe_m"], model) and array_sha(d["direction"]) == witness["direction_sha256"]
            and array_sha(model) == witness["probe_m_sha256"]
            and np.allclose(d["direction"], model * active / scale, rtol=1e-13, atol=0))
        audit.near(cid + ":direction_normscale:" + direction, witness["norm_scale"], scale)
        gradient_dd = scalar_dot(g["production_fold"], d["direction"])
        naive_dd = scalar_dot(g["naive_crop"], d["direction"])
        normscale = scalar_norm(g["production_fold"]) * scalar_norm(d["direction"])
        informative = normscale > 0 and abs(gradient_dd) > criteria["near_zero_directional_fraction"] * normscale
        for step in protocol["gradient"]["relative_steps"]:
            native = next(r for r in status["derivative_checks"] if r["direction"] == direction and r["relative_step"] == step)
            item = dict(direction=direction, relative_step=step, native_status=native["status"], informative=informative,
                        adjoint_derivative=gradient_dd, naive_crop_derivative=naive_dd, finite_difference=None, relative_error=None)
            audit.near(cid + f":epsilon:{direction}:{step}", native.get("epsilon"), step * scale)
            objectives = []
            for sign, sign_value in [("plus", 1), ("minus", -1)]:
                call_id = f"fd_{direction}_{step}_{sign}"
                call = rows[call_id]
                restore = call.get("restoration", {})
                audit.check(cid + ":restored_actual_model:" + call_id, restore.get("status") == "restored"
                    and restore.get("model", {}).get("actual_physical_m_sha256") == array_sha(model)
                    and restore.get("model", {}).get("physical_velocity_sha256") == array_sha(probe))
                if call["status"] != "complete" or "result" not in call:
                    objectives.append(None)
                    continue
                rec = call["result"]
                path = directory / rec["artifact"]
                audit.check(cid + ":artifact:" + call_id, checksum(path) == rec["artifact_sha256"])
                trial = decode(path)
                requested = model + sign_value * native["epsilon"] * d["direction"]
                audit.check(cid + ":requested_m:" + call_id, np.array_equal(trial["requested_m"], requested)
                    and array_sha(requested) == rec["requested_m_sha256"]
                    and np.allclose(requested, model * (1 + sign_value * step * active), rtol=1e-13, atol=0))
                vp = 1 / np.sqrt(requested)
                audit.check(cid + ":trial_vp:" + call_id, np.array_equal(trial["requested_trial_velocity_kmps"], vp)
                    and np.array_equal(trial["actual_physical_velocity_kmps"], vp))
                forward_checks(audit, cid + ":" + call_id, trial, rec, vp, config, protocol)
                audit.check(cid + ":FD_source_time_mask:" + call_id,
                    np.array_equal(trial["source_before"], observed["source_before"])
                    and np.array_equal(trial["times_ms"], observed["times_ms"])
                    and np.array_equal(trial["computed_mask"], mask))
                value = objective(trial["receiver"], observed["receiver"], mask)
                audit.near(cid + ":objective:" + call_id, rec["objective"], value)
                audit.check(cid + ":row_result_binding:" + call_id, native.get(sign) == rec
                    and rec["restored_probe_model"] == restore["model"])
                objectives.append(value)
            if all(v is not None for v in objectives):
                fd = (objectives[0] - objectives[1]) / (2 * native["epsilon"])
                absolute_error = abs(fd - gradient_dd)
                relative_error = absolute_error / max(abs(fd), abs(gradient_dd)) if informative else None
                item.update(finite_difference=fd, absolute_error=absolute_error, relative_error=relative_error,
                            objectives=objectives, normalized_absolute_error=absolute_error / normscale if normscale else None)
                audit.near(cid + f":gradient_dot:{direction}:{step}", native["adjoint_derivative"], gradient_dd, normscale)
                audit.near(cid + f":naive_dot:{direction}:{step}", native["naive_crop_derivative"], naive_dd, normscale)
                audit.near(cid + f":FD_scalar:{direction}:{step}", native["finite_difference"], fd, max(abs(fd), normscale))
                audit.check(cid + f":informative_binding:{direction}:{step}", native["informative"] == informative)
                if informative:
                    audit.check(cid + f":relative_DD_error:{direction}:{step}", math.isclose(native["relative_error"], relative_error,
                        rel_tol=criteria["scalar_report_relative_tolerance"], abs_tol=criteria["derived_relative_error_absolute_tolerance"]),
                        producer=native["relative_error"], independently_recomputed=relative_error)
                else:
                    audit.check(cid + f":uninformative_not_pass:{direction}:{step}", native["relative_error"] is None)
            if step == criteria["fixed_middle_relative_step"]:
                middle.append(informative and item["relative_error"] is not None
                              and item["relative_error"] <= criteria["middle_step_directional_relative_error_limit"])
            derivative_rows.append(item)
            planned_row = next(r for r in report["planned_derivatives"] if r["direction"] == direction and r["relative_step"] == step)
            planned_row["audit_status"] = "recomputed" if item["finite_difference"] is not None else "unavailable"
    report["derivatives_scalar"] = derivative_rows
    report["fixed_middle_gradient_numerical_assessment"] = len(middle) == 2 and all(middle)
    audit.check(cid + ":middle_assessment_binding", status["gradient_numerical_gate"] == report["fixed_middle_gradient_numerical_assessment"])
    report["probe_objective_scalar"] = objective(predicted["receiver"], observed["receiver"], mask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criteria", type=Path, default=Path(__file__).with_name("criteria.json"))
    parser.add_argument("--execute-audit", action="store_true")
    for name in ["repo", "inputs", "run", "producer", "primary-source-root", "terminal-receipt", "output"]:
        parser.add_argument("--" + name, type=Path)
    for name in ["expected-source-sha256", "expected-criteria-sha256", "expected-terminal-sha256"]:
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    criteria = json.loads(args.criteria.read_text())
    if not args.execute_audit:
        print(json.dumps(dict(status="metadata_only_no_BP_or_native_output_reads", planned_calls=57,
            planned_cases=3, physical_acceptance=None, source_sha256=checksum(Path(__file__)), criteria_sha256=checksum(args.criteria))))
        return 0
    required = [args.repo, args.inputs, args.run, args.producer, args.primary_source_root, args.terminal_receipt,
                args.output, args.expected_source_sha256, args.expected_criteria_sha256, args.expected_terminal_sha256]
    if not all(required):
        parser.error("explicit audit requires all paths and source/criteria/root-terminal SHA gates")
    if checksum(Path(__file__)) != args.expected_source_sha256 or checksum(args.criteria) != args.expected_criteria_sha256:
        raise ValueError("reviewed auditor/criteria changed")
    return execute(args, criteria)


if __name__ == "__main__":
    raise SystemExit(main())
