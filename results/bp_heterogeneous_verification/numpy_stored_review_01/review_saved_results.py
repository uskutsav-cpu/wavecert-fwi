"""Post-terminal stored-array arithmetic review; no propagation or source imports."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(a):
    return math.hypot(*(float(v) for v in a.ravel()))


def inner(a, b):
    return math.fsum(float(x) * float(y) for x, y in zip(a.ravel(), b.ravel(), strict=True))


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    if digest(Path(__file__)) != args.expected_source_sha256 or args.output.exists():
        raise ValueError("exact pre-execution source pin and fresh output required")
    work = args.outer / "work"
    stage = work / "new-phase-staging/numpy_bp_tangent_replay"
    native = work / "bp_heterogeneous_guards_run_01"
    replay = work / "numpy_bp_tangent_replay_run_01"
    checks, issues, hashes, case_results = [], [], {}, []

    def check(label, value):
        checks.append(label)
        if not bool(value):
            issues.append(label)

    def read_json(path):
        hashes[str(path.relative_to(args.outer))] = digest(path)
        return json.loads(path.read_bytes())

    def arrays(path, expected):
        h = digest(path)
        hashes[str(path.relative_to(args.outer))] = h
        check("artifact hash: " + str(path), h == expected)
        with np.load(path, allow_pickle=False) as data:
            result = {k: data[k].copy() for k in data.files}
        check("bounded arrays: " + str(path), all(a.ndim <= 2 and not a.dtype.hasobject for a in result.values()))
        return result

    p = read_json(stage / "protocol.json")
    freeze = read_json(work / "numpy_bp_tangent_replay_prefreeze_01.json")
    terminal = read_json(work / "numpy_bp_tangent_replay_terminal_observation_01.json")
    native_terminal = read_json(work / "bp_heterogeneous_guards_terminal_observation_01.json")
    result = read_json(replay / "complete_receipt.json")
    status = read_json(replay / "status.json")
    original = read_json(native / "complete_receipt.json")
    check("expected complete replay", digest(replay / "complete_receipt.json") == "207bc209e59e6e07d8a1da1d26afd15ab1a92fcb01feef7793bd29d66438e89a")
    check("terminal exit and completion", terminal["exit_code"] == 0 and terminal["complete_receipt_sha256"] == digest(replay / "complete_receipt.json"))
    check("native terminal gate", native_terminal["exit_code"] == 0 and native_terminal["complete_receipt_sha256"] == digest(native / "complete_receipt.json") == result["expected_native_complete_sha256"] == freeze["native_complete_sha256"])
    check("root terminal semantics manually checked", freeze["terminal_semantics_manually_checked"] is True)
    check("same status and complete", result == status)
    check("fixed completed replay result", result["status"] == "complete_comparison" and result["result"] == "ALL_FIXED_NUMERICAL_CHECKS_AGREE" and result["numerical_agreement_pass"] is True)
    check("full denominators and phase record", result["own_solves_planned"] == 3 and result["directions_planned"] == 6 and result["full3_case_6_direction_denominators_retained"] is True and result["all3_own_complete_before_native_arrays"] is True)
    check("original full call denominator", len(original["calls"]) == 57 and all(r["status"] == "complete" for r in original["calls"]))
    check("native status complete", original["status"] == "complete_guard_attempts")
    check("null physical acceptance", result["physical_acceptance"] is None and original["physical_acceptance"] is None)
    check("source/protocol tied to prefreeze", result["source_sha256"] == freeze["files_sha256"]["run_replay.py"] and result["protocol_sha256"] == freeze["files_sha256"]["protocol.json"])
    check("native terminal observation tied to prefreeze", result["terminal_observation_sha256"] == digest(work / "bp_heterogeneous_guards_terminal_observation_01.json") == freeze["terminal_observation_sha256"])
    pins = dict(freeze["files_sha256"])
    pins.update(p["copied_primary_python_sha256"])
    pins.update({"primary_metadata/" + k: v for k, v in p["primary_C_sha256"].items()})
    pins.update({"query_primary_metadata.py": p["metadata_query_source_sha256"], "primary_metadata/operator_metadata.json": p["metadata_sha256"], "original_native_protocol.json": p["original_native_protocol_sha256"]})
    for filename, expected in pins.items():
        check("frozen source: " + filename, digest(stage / filename) == expected)
    check("native source and protocol pinned", original["source_sha256"] == p["native_producer_sha256"] and original["protocol_sha256"] == p["original_native_protocol_sha256"])
    check("frozen ordinary-FP criteria", p["criteria"] == dict(trace_relative_L2=1e-9, gradient_relative_error=1e-3, source_relative_L2=1e-14, direction_relative_L2=1e-14, fold_relative_L2=1e-13, near_zero_relative_normscale=1e-8))
    check("exact case schedule", [r["case_id"] for r in result["cases"]] == [c["id"] for c in p["inputs"]])
    calls = {(r["case_id"], r["call_id"]): r for r in original["calls"]}
    for row, case in zip(result["cases"], p["inputs"], strict=True):
        cid = case["id"]
        own = arrays(replay / row["artifact"], row["artifact_sha256"])
        child = read_json(native / cid / "complete_receipt.json")
        check(cid + " worker receipt and aggregate", child == read_json(native / cid / "status.json") and child["calls"] == [r for r in original["calls"] if r["case_id"] == cid])
        check(cid + " source before/after", child["source_before"] == child["source_after"] and child["source_before"]["production"] == p["production_source_sha256"] and child["source_before"]["input_sha256"] == case["sha256"])
        check(cid + " complete trace and two directions", row["status"] == "complete_own_solve" and row["comparison_status"] == "complete" and [d["direction"] for d in row["directions"]] == ["interior", "edge"] and all(d["status"] == "complete" for d in row["directions"]))
        loaded = {}
        for call in ["probe_saved", "probe_adjoint", "coarse40"]:
            record = calls[cid, call]["result"]
            loaded[call] = arrays(native / cid / record["artifact"], record["artifact_sha256"])
        probe, grad, obs = loaded["probe_saved"], loaded["probe_adjoint"], loaded["coarse40"]
        mask = np.arange(2001)
        mask = (mask >= 1) & (mask <= 1999)
        times = np.arange(2001, dtype=np.float64) * .5
        check(cid + " exact time and computed mask", np.array_equal(own["computed_mask"], mask) and np.array_equal(probe["computed_mask"], mask) and np.array_equal(obs["computed_mask"], mask) and all(np.array_equal(a["times_ms"], times) for a in [own, probe, obs]))
        check(cid + " receiver shape and sentinel", own["receiver"].shape == (2001, 10) and not np.any(own["receiver"][[0, 2000]]) and not np.any(probe["receiver"][[0, 2000]]))
        check(cid + " probe bytes", np.array_equal(own["physical_probe_m"], probe["actual_physical_m"]) and np.array_equal(own["physical_probe_velocity"], probe["actual_physical_velocity_kmps"]))
        check(cid + " declared own model hashes", array_digest(own["physical_probe_m"]) == row["record"]["physical_m_sha256"] and array_digest(own["physical_probe_velocity"]) == row["record"]["physical_velocity_sha256"])
        check(cid + " held source", np.array_equal(own["source"], probe["source_before"]) and np.array_equal(probe["source_before"], probe["source_after"]))
        q = np.array([(1 - 2 * (math.pi * .005 * (t - 200)) ** 2) * math.exp(-(math.pi * .005 * (t - 200)) ** 2) for t in times])[:, None]
        source_error = norm(q - own["source"]) / norm(q)
        check(cid + " scalar analytic source", source_error <= p["criteria"]["source_relative_L2"])
        check(cid + " physical coordinates", np.array_equal(probe["source_coordinates"], [[1200., 200.]]) and np.array_equal(probe["receiver_coordinates"], np.asarray(p["parameters"]["receiver_nodes"]) * 12.5))
        residual = probe["receiver"] - obs["receiver"]
        residual[~mask] = 0
        check(cid + " residual objective binding", np.array_equal(residual, grad["residual"]))
        trace_rel = norm(own["receiver"][mask] - probe["receiver"][mask]) / norm(probe["receiver"][mask])
        check(cid + " trace metric and criterion", trace_rel == row["comparison"]["trace"]["value"] and trace_rel <= p["criteria"]["trace_relative_L2"])
        folded = np.empty((192, 96))
        for x in range(192):
            xs = range(0, 41) if x == 0 else range(231, 272) if x == 191 else [x + 40]
            for z in range(96):
                zs = range(0, 41) if z == 0 else range(135, 176) if z == 95 else [z + 40]
                folded[x, z] = math.fsum(float(grad["padded_gradient"][i, j]) for i in xs for j in zs)
        fold_rel = norm(folded - grad["production_fold"]) / norm(grad["production_fold"])
        check(cid + " independent edge and corner pullback", fold_rel <= p["criteria"]["fold_relative_L2"])
        ds = []
        for d in row["directions"]:
            label = d["direction"]
            wrec = child["direction_witnesses"][label]
            witness = arrays(native / cid / wrec["artifact"], wrec["artifact_sha256"])
            direction = own[label + "_direction"]
            tangent = own[label + "_receiver_tangent"]
            check(cid + "/" + label + " direction/model bound", np.array_equal(direction, witness["direction"]) and np.array_equal(own["physical_probe_m"], witness["probe_m"]))
            check(cid + "/" + label + " tangent shape/sentinel", tangent.shape == (2001, 10) and not np.any(tangent[[0, 2000]]))
            dd = inner(residual[mask], tangent[mask])
            native_dd = inner(grad["production_fold"], direction)
            crop_dd = inner(grad["naive_crop"], direction)
            scale = norm(grad["production_fold"]) * norm(direction)
            informative = scale > 0 and abs(native_dd) > 1e-8 * scale
            relative = abs(dd - native_dd) / max(abs(dd), abs(native_dd)) if informative else None
            check(cid + "/" + label + " exact DD stored arithmetic", dd == d["independent_tangent_DD"] and native_dd == d["native_folded_gradient_DD"] and crop_dd == d["naive_crop_DD"] and scale == d["gradient_direction_normscale"] and relative == d["relative_error"])
            check(cid + "/" + label + " informative and fixed criterion", informative and d["informative"] is True and d["agreement"] is True and relative <= 1e-3)
            ds.append(dict(direction=label, tangent_DD=dd, native_folded_DD=native_dd, naive_crop_DD=crop_dd, relative_error=relative, informative=informative))
        case_results.append(dict(case_id=cid, trace_relative_L2=trace_rel, scalar_source_relative_L2=source_error, fold_relative_L2=fold_rel, directions=ds))
    repo = work / "wavecert-fwi"
    files = [repo / "pyproject.toml", *repo.glob("requirements*.txt"), *(repo / "src").rglob("*.py"), *(q for q in (repo / "configs").rglob("*") if q.is_file()), *(q for q in (repo / "scripts").rglob("*") if q.suffix in {".py", ".sh"}), *(repo / ".github/workflows").glob("*.yml")]
    source = hashlib.sha256()
    for path in sorted(set(files)):
        source.update(path.relative_to(repo).as_posix().encode() + b"\0" + digest(path).encode() + b"\0")
    check("all161 production manifest unchanged", len(set(files)) == 161 and source.hexdigest() == p["production_source_sha256"])
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), result="CLEAR" if not issues else "ISSUES_RETAINED", checks=len(checks), issues=issues, reviewed_source_sha256=args.expected_source_sha256, artifact_hashes=hashes, source_files=len(set(files)), production_source_sha256=source.hexdigest(), cases=case_results, full_case_denominator=3, full_direction_denominator=6, native_call_denominator=57, physical_acceptance=None, propagation_executed_by_this_review=False, production_imports=False,
        execution_evidence="Root observed native session88942 exit0 and replay session28294 exit0, corroborated by terminal JSON, complete hashes and final CLI log. Reviewer executed stored-array/hash/scalar checks only.",
        phase_limit="All-own-before-native-decode follows the frozen reviewed source path and stored phase flags, not independent syscall tracing. Final rolling states/full histories are only hashed or unsaved, so this postcheck does not recreate trajectories.",
        scope="Source-informed ordinary-FP numerical implementation and derivative verification, all3 correlated crops of one BP ancestor. No continuum/field accuracy, original acquisition, inverse recovery, certificate transfer, student final design, human/blind independence or full-goal completion. Native physical differences/failures remain separate; physical acceptance null.")
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(result=report["result"], checks=report["checks"], issues=issues, report_sha256=digest(args.output), max_trace_relative_L2=max(r["trace_relative_L2"] for r in case_results), max_direction_relative_error=max(d["relative_error"] for r in case_results for d in r["directions"]))))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
