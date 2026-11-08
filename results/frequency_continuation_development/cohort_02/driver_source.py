"""Four-arm paid frequency-schedule development study; default is metadata only.

Source/configuration review and an exact-hash root freeze precede project work.
The student final research design and scientific interpretation are not supplied
by this engineering runner. No retained BP or initial-probe array is an input.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import gzip
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import shutil
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

METHODS = ["joint8", "low4_then8", "restart8", "original4"]
FREQUENCIES = [.25, .5, 1., 2., 2.5, 3., 3.5, 4.]
PRODUCTION_SHA = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
FINAL_RECEIPT_RESERVE = 2 * 1024**2


class StaleFreeze(RuntimeError):
    pass


class NumericalAuditFailure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def exclusive(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    try:
        exclusive(temporary, json_bytes(value))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def retained_bytes(directory):
    return sum(p.stat().st_size for p in directory.iterdir() if p.is_file())


def checked_exclusive(path, data, limit=None, reserve=0):
    if limit is not None and retained_bytes(path.parent) + len(data) + reserve > limit:
        raise OSError("retained-artifact envelope would be exceeded")
    exclusive(path, data)


def gzip_json(path, value, *, limit=None, reserve=0):
    compressed = gzip.compress(json_bytes(value), mtime=0)
    checked_exclusive(path, compressed, limit, reserve)


def schedule(protocol):
    rng, rows = random.Random(protocol["schedule"]["seed"]), []
    for repetition in range(3):
        for case in range(8):
            order = list(METHODS)
            rng.shuffle(order)
            for position, arm in enumerate(order):
                rows.append(dict(run_index=len(rows), case_index=case, repetition=repetition,
                    order_position=position, method=arm, identity=protocol["identities"][case],
                    status="unattempted", final_study_eligible=False, primary_quality=None))
    return rows


def load_protocol(path, *, executable=False):
    protocol = json.loads(path.read_bytes())
    original_path = path.with_name("original_protocol.json")
    if sha(original_path) != protocol["original_protocol_sha256"]:
        raise StaleFreeze("original protocol changed")
    original = json.loads(original_path.read_bytes())
    if (protocol["methods"] != METHODS or protocol["available_frequencies_hz"] != FREQUENCIES
            or protocol["schedule"]["total_runs"] != 96 or protocol["schedule"]["cases"] != 8
            or protocol["schedule"]["repetitions"] != 3 or protocol["schedule"]["seed"] != 2026110802
            or protocol["controller_settings"]["single_stage_maximum_steps"] != 4096
            or protocol["controller_settings"]["per_stage_maximum_steps"] != 2048
            or protocol["controller_settings"]["engine_deadline_seconds"] != 7.5
            or protocol["controller_settings"]["first_stage_deadline_seconds"] != 3.75
            or protocol["controller_settings"]["api_return_cap_seconds"] != 8.
            or protocol["production_source_sha256"] != PRODUCTION_SHA
            or protocol["student_final_research_design"] is not False
            or protocol["competition_evidence"] is not False):
        raise StaleFreeze("fixed development scope, full denominator or settings changed")
    for key in ["identities", "inputs_sha256", "physics", "geometry", "squared_slowness_bounds", "lbfgsb", "threads"]:
        if protocol[key] != original[key]:
            raise StaleFreeze("old raw-input/operator setting changed: " + key)
    for name, digest in protocol["local_dependency_sha256"].items():
        candidate = (path.parent / name).resolve()
        if not candidate.is_relative_to(path.parent.resolve()) or sha(candidate) != digest:
            raise StaleFreeze("local dependency changed: " + name)
    if executable and protocol["controller_dependencies_pending_review"]:
        raise StaleFreeze("controller dependencies/review are not frozen")
    return protocol, original


def import_file(name, path):
    if name in sys.modules:
        raise StaleFreeze("explicit import name already occupied: " + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise StaleFreeze("wrong explicit module origin")
    return module


def load_runtime(repo, bundle, protocol):
    sys.path.insert(0, str(repo / "src"))
    directory = repo / protocol["original_candidate_relative_directory"]
    for name, digest in protocol["candidate_files_sha256"].items():
        if sha(directory / name) != digest:
            raise StaleFreeze("old helper package changed: " + name)
    name = "_frequency_study_original_helpers"
    if name in sys.modules:
        raise StaleFreeze("helper package import name occupied")
    spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
        submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    helpers = importlib.import_module(name + ".verification")
    helpers._read_protocol(directory / "protocol.json")
    engine = import_file("_frequency_study_bounded_exact", bundle / "bounded_exact.py")
    controller = import_file("_frequency_study_controller", bundle / "controller.py")
    api = controller.load_native_api(repo, engine_run=engine.run_exact_lbfgsb,
        engine_source_path=bundle / "bounded_exact.py",
        engine_source_sha256=protocol["local_dependency_sha256"]["bounded_exact.py"])
    audit = import_file("_frequency_study_independent_graph", bundle / "independent_saved_models.py")
    # The immutable807 source initializes these globals only in its standalone
    # CLI. Imported API users must initialize the same five aliases explicitly.
    for alias, module in [("np", "numpy"), ("scipy", "scipy"), ("sp", "scipy.sparse"),
                          ("spla", "scipy.sparse.linalg"), ("ndi", "scipy.ndimage")]:
        setattr(audit, alias, importlib.import_module(module))
    return SimpleNamespace(helpers=helpers, controller=controller, engine=engine, api=api,
        audit=audit, np=importlib.import_module("numpy"))


def static_receipt(repo, path, protocol, runtime):
    original_raw = path.with_name("original_protocol.json").read_bytes()
    production = runtime.helpers.source_receipt(repo, original_raw)
    if production["repository_source_sha256"] != PRODUCTION_SHA:
        raise StaleFreeze("production source changed")
    local = {name: sha(path.parent / name) for name in protocol["local_dependency_sha256"]}
    if local != protocol["local_dependency_sha256"]:
        raise StaleFreeze("frozen local dependencies changed")
    return dict(production=production, local=local, driver_sha256=sha(Path(__file__)), protocol_sha256=sha(path))


def verify_inputs(repo, protocol):
    result = {}
    for name, digest in protocol["inputs_sha256"].items():
        path = (repo / name).resolve()
        if not path.is_relative_to(repo.resolve()) or sha(path) != digest:
            raise StaleFreeze("pinned input changed: " + name)
        result[name] = digest
    return result


def preparation_protocol(original, protocol):
    copied = copy.deepcopy(original)
    copied["geometry"]["frequencies_hz"] = list(protocol["available_frequencies_hz"])
    copied["geometry"]["blocks_per_model"] = protocol["observation_block_count"]
    return copied


def data_binding(prepared, runtime):
    np = runtime.np
    h = hashlib.sha256(json_bytes(list(prepared["receivers"])))
    for block in prepared["blocks"]:
        h.update(json_bytes([int(block.source_index), float(block.frequency_hz), str(block.label)]))
        h.update(np.ascontiguousarray(block.observed, dtype="<c16").tobytes())
    return dict(initial_native_m_sha256=runtime.controller.model_sha(prepared["initial"]),
        truth_sha256=prepared["truth_sha256"], raw_model_sha256=prepared["raw_model_sha256"],
        available8_observations_sha256=h.hexdigest())


def prepare_for_run(repo, original, preparation, row, runtime, context, expected, check_static, cost_context):
    check_static()  # True global start has already occurred in the controller caller.
    try:
        cases = runtime.helpers.load_raw_inputs(repo, original)
    except (ValueError, OSError) as exc:
        raise StaleFreeze("pinned raw inputs/schema changed") from exc
    raw = cases[row["case_index"]]
    context["raw_case"] = raw
    # No provider exists inside the old helper; interruption before return is
    # explicitly unknown preparation work, never an assumed zero counter.
    prepared = runtime.helpers.prepare_case(raw, preparation)
    context["prepared"] = prepared
    costs = copy.deepcopy(prepared["preparation_costs"])
    cost_context.register_preparation_costs("all8_observation_generation", lambda costs=costs: copy.deepcopy(costs))
    binding = data_binding(prepared, runtime)
    if expected is not None and binding != expected:
        raise StaleFreeze("repeated common native initial/data bytes differ")
    context["common_binding"] = binding
    return runtime.controller.PreparedProblem(dict(preparation["physics"]), prepared["initial"],
        prepared["blocks"], prepared["receivers"])


def outer_timing(measurement, returned, cap):
    old = measurement["timing"]
    wall = float(returned) - old["global_started"]
    inner = old["actual_api_return_seconds"]
    if isinstance(returned, bool) or not math.isfinite(wall) or not 0 <= inner <= wall + 1e-9:
        raise ValueError("invalid nested caller-observed return timing")
    valid = bool(measurement["runtime_gate"]["eligible"] is True
        and measurement["payload"]["method_valid_before_return_gate"] and wall <= cap)
    return dict(actual_api_return_seconds=wall, inner_execute_timed_return_seconds=inner,
        measured_helper_return_tail_seconds=max(0., wall-inner), api_return_cap_seconds=cap,
        overshoot_seconds=max(0., wall-cap), runtime_eligible=valid,
        event_seconds=wall if valid else None,
        inner_runtime_gate=dict(measurement["runtime_gate"]),
        scope="actual measure_controller_return return; all algorithm/data/output work charged, artifact IO and offline audits afterward",
        absolute_ages_and_preparation_are_subsets_not_summed=True)


def scalar_trace(payload):
    trace = {k: v for k, v in payload.items() if k not in {"initial_model", "final_model"}}
    json_bytes(trace)  # Unexpected array/field exports fail rather than silently flattening them.
    return trace


def save_core(output, row, measurement, runtime, *, suffix="", limit=None, reserve=0):
    payload, np = measurement["payload"], runtime.np
    prefix = f"run_{row['run_index']:03d}" + suffix
    arrays = {key: payload[key] for key in ["initial_model", "final_model"] if payload.get(key) is not None}
    if any(a.ndim != 1 or a.size != 2401 or a.dtype != np.float64 for a in arrays.values()):
        raise ValueError("only native2401-vector snapshots may be saved")
    if arrays:
        p = output / (prefix + "_models.npz")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        checked_exclusive(p, buffer.getvalue(), limit, reserve)
        row.update(models_artifact=p.name, models_sha256=sha(p))
    p = output / (prefix + "_trajectory.json.gz")
    gzip_json(p, scalar_trace(payload), limit=limit, reserve=reserve)
    row.update(trajectory_artifact=p.name, trajectory_sha256=sha(p))


def failure_witness(output, row, attempt, context, runtime, *, limit=None):
    """Retain an available paid vector prefix without inventing a return event."""
    payload = context.get("measurement", {}).get("payload") or attempt.get("payload")
    if not isinstance(payload, dict):
        row["failure_vector_prefix_available"] = False
        return
    row["failure_vector_prefix_available"] = any(payload.get(k) is not None for k in ["initial_model", "final_model"])
    existing = {k: row[k] for k in ["models_artifact", "models_sha256", "trajectory_artifact", "trajectory_sha256"] if k in row}
    try:
        save_core(output, row, dict(payload=payload), runtime, suffix="_failure_prefix", limit=limit)
        row["failure_prefix_saved"] = True
    except Exception as exc:
        row.update(failure_prefix_saved=False, failure_prefix_save_error=f"{type(exc).__name__}: {exc}")
    if existing:
        row["earlier_returned_core_artifacts"] = existing


def offline_audit(row, context, payload, protocol, original, runtime):
    np, module = runtime.np, runtime.audit
    p = preparation_protocol(original, protocol)
    checks = module.Checks()
    independent = module.IndependentCase(context["raw_case"]["raw_model"], p, checks, row["case_index"])
    prepared = context["prepared"]
    actual = {f: np.column_stack([b.observed for b in prepared["blocks"] if float(b.frequency_hz) == f])
              for f in protocol["available_frequencies_hz"]}
    observation_errors = {}
    for f, reference in independent.observations.items():
        error = module.scalar_norm(actual[f]-reference) / max(module.scalar_norm(reference), 1e-300)
        observation_errors[str(f)] = error
        checks.require(math.isfinite(error) and error <= protocol["offline_audit"]["observation_relative_L2_maximum"],
            f"observations/f{f}", error)
    records, own = {}, {}
    for label, key, receipt in [("initial", "initial_model", payload.get("common_initial_full8")),
                                ("final", "final_model", payload.get("common_final_full8"))]:
        model = payload.get(key)
        if model is None:
            continue
        record = independent.evaluate(model)
        records[label] = record
        truth = prepared["truth"]
        own[label] = dict(relative_squared_slowness_L2=float(np.linalg.norm(model-truth)/np.linalg.norm(truth)),
            relative_physical_velocity_L2=float(np.linalg.norm(1/np.sqrt(model)-1/np.sqrt(truth))/np.linalg.norm(1/np.sqrt(truth))))
        independent.quality_check(record, own[label], label+"/quality")
        if receipt is not None:
            independent.objective_check(record["objective"], receipt["objective"], label+"/paid_common8")
            checks.require(receipt["model_sha256"] == runtime.controller.model_sha(model), label+"/model_binding")
    return dict(records=records, own_quality=own, checks=checks.count, numeric_comparisons=checks.numeric_count,
        failures=checks.failures, observation_relative_L2=observation_errors,
        factorizations=independent.factorizations, forward_rhs=independent.forward_rhs,
        maximum_relative_linear_residual=independent.maximum_relative_linear_residual,
        scope="post-return independently assembled graph; shared NumPy/SciPy/SuperLU; no policy feedback or physical acceptance")


def run(repo, output, path, expected_driver, expected_protocol):
    bootstrap = perf_counter()
    repo, output, path = repo.resolve(), output.resolve(), path.resolve()
    protocol, original = load_protocol(path, executable=True)
    if sha(path) != expected_protocol or sha(Path(__file__)) != expected_driver:
        raise StaleFreeze("explicit reviewed driver/protocol hash mismatch")
    if output.exists() or output.is_relative_to(repo) or output.is_relative_to(path.parent):
        raise ValueError("fresh output outside repository/frozen bundle required")
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < protocol["disk"]["initial_free_bytes"]:
        raise OSError("fixed initial disk reserve required")
    if any(os.environ.get(k) != v for k, v in protocol["threads"].items()):
        raise ValueError("fixed single-thread environment required")
    runtime = load_runtime(repo, path.parent, protocol)
    freeze = static_receipt(repo, path, protocol, runtime)
    rows = schedule(protocol)
    output.mkdir()
    for name in ["original_protocol.json", *protocol["local_dependency_sha256"]]:
        if not (output / name).exists():
            exclusive(output / name, (path.parent / name).read_bytes())
    exclusive(output / "driver_source.py", Path(__file__).read_bytes())
    exclusive(output / "protocol.json", path.read_bytes())
    exclusive(output / "schedule.json", json_bytes(rows))
    status = dict(status="running", runs=rows, started_utc=utc(), source_before=freeze,
        all96_scheduled_before_project_arrays=True, shared_bootstrap_seconds=perf_counter()-bootstrap,
        environment=dict(python=sys.version, platform=platform.platform(), numpy=runtime.np.__version__,
                         scipy=importlib.import_module("scipy").__version__, threads=protocol["threads"]),
        scientific_goal_complete=False, physical_acceptance=None)
    atomic(output / "status.json", status)
    bindings = {}
    settings = runtime.controller.ControllerSettings(**protocol["controller_settings"])
    preparation = preparation_protocol(original, protocol)
    output_limit = protocol["disk"]["maximum_output_bytes"]

    def check_static():
        if static_receipt(repo, path, protocol, runtime) != freeze:
            raise StaleFreeze("source/protocol freeze changed")

    try:
        for row in rows:
            attempt, context = {}, {}
            row.update(status="attempted", started_utc=utc(), progress_artifact=f"run_{row['run_index']:03d}_progress.json")
            atomic(output / "status.json", status)
            atomic(output / row["progress_artifact"], dict(run_index=row["run_index"], phase="before_timed_call"))

            def progress(event, row=row):
                row["phase"] = event["phase"]
                atomic(output / row["progress_artifact"], dict(run_index=row["run_index"], method=row["method"], **event))

            def prepare(cost_context, row=row, context=context):
                return prepare_for_run(repo, original, preparation, row, runtime, context,
                    bindings.get(row["case_index"]), check_static, cost_context)

            try:
                if shutil.disk_usage(output).free < protocol["disk"]["per_run_reserve_bytes"]:
                    raise OSError("per-run disk reserve exhausted")
                if retained_bytes(output) + FINAL_RECEIPT_RESERVE >= output_limit:
                    raise OSError("final receipt reservation exhausted before next run")
                measurement = runtime.controller.measure_controller_return(arm=row["method"], prepare=prepare,
                    api=runtime.api, settings=settings, attempt=attempt, progress=progress)
                returned = perf_counter()  # Caller observes the complete measurement/controller return.
                context["measurement"] = measurement
                payload = measurement["payload"]
                row["caller_observed_return_stamp"] = returned
                # Returned model/trace evidence survives even a later timing or
                # integrity failure. These writes occur after the fixed stamp.
                save_core(output, row, measurement, runtime, limit=output_limit, reserve=FINAL_RECEIPT_RESERVE)
                row.update(timing=outer_timing(measurement, returned, settings.api_return_cap_seconds),
                    controller_status=payload["status"], common_binding=context.get("common_binding"),
                    stage_summaries=[{k: s.get(k) for k in ["index", "status", "absolute_deadline_seconds",
                        "maximum_steps", "native_callback_events", "retained_native_callback_events",
                        "callback_count_scope", "eligible_retained_checkpoint_count",
                        "eligible_checkpoint_count_scope"]} for s in payload["stages"]])
                if payload["status"] == "failed_controller":
                    raise StaleFreeze("controller infrastructure failure: " + str(payload.get("error")))
                if "common_binding" in context:
                    bindings.setdefault(row["case_index"], context["common_binding"])
                row["phase"] = "post_return_integrity"
                check_static()
                row["input_hashes_after"] = verify_inputs(repo, protocol)
                row["phase"] = "offline_independent_audit"
                atomic(output / row["progress_artifact"], dict(run_index=row["run_index"],
                    method=row["method"], phase=row["phase"]))
                audit = offline_audit(row, context, payload, protocol, original, runtime)
                p = output / f"run_{row['run_index']:03d}_audit.json.gz"
                gzip_json(p, audit, limit=output_limit, reserve=FINAL_RECEIPT_RESERVE)
                row.update(audit_artifact=p.name, audit_sha256=sha(p))
                if audit["failures"]:
                    raise NumericalAuditFailure("independent common8 objective/quality audit failed")
                row.update(status="complete", phase="finished", integrity_valid=True,
                    final_study_eligible=bool(row["timing"]["runtime_eligible"]),
                    primary_quality=audit["own_quality"].get("final") if row["timing"]["runtime_eligible"] else None)
                ratio = payload["common_full8_ratio"]
                row["returned_targets"] = [dict(ratio=r, reached=bool(ratio is not None and ratio <= r),
                    event_seconds=row["timing"]["actual_api_return_seconds"]
                        if row["primary_quality"] is not None and ratio is not None and ratio <= r else None)
                    for r in protocol["targets"]["ratios"]]
                row["common_full8_ratio"] = ratio
            except BaseException as exc:
                row.update(status="failed" if isinstance(exc, Exception) else "interrupted",
                    error=f"{type(exc).__name__}: {exc}", final_study_eligible=False, primary_quality=None)
                failure_witness(output, row, attempt, context, runtime, limit=output_limit)
                started = attempt.get("global_started_perf_counter")
                if isinstance(started, (int, float)) and not isinstance(started, bool) and math.isfinite(started):
                    row["wall_until_failure_or_interrupt_seconds"] = perf_counter() - started
                failure = dict(row=row, available_attempt={k: v for k, v in attempt.items() if k != "payload"},
                    preparation_work_unknown_if_helper_did_not_return="prepared" not in context,
                    successful_API_return="measurement" in context)
                p = output / f"run_{row['run_index']:03d}_failure.json.gz"
                try:
                    gzip_json(p, failure, limit=output_limit)
                    row.update(failure_artifact=p.name, failure_sha256=sha(p))
                except Exception as save_exc:
                    row["failure_persistence_error"] = f"{type(save_exc).__name__}: {save_exc}"
                atomic(output / "status.json", status)
                if not isinstance(exc, NumericalAuditFailure):
                    raise  # Unexpected infrastructure errors never consume the other95 runs blindly.
            row["finished_utc"] = utc()
            atomic(output / row["progress_artifact"], dict(run_index=row["run_index"],
                phase="finished", status=row["status"], method=row["method"]))
            atomic(output / "status.json", status)
            if retained_bytes(output) > output_limit:
                raise OSError("fixed compact-output envelope exhausted")
            print(json.dumps(dict(run_index=row["run_index"], method=row["method"], status=row["status"],
                actual_api_return=row.get("timing", {}).get("actual_api_return_seconds"),
                primary_quality_eligible=row["primary_quality"] is not None)), flush=True)
        check_static()
        status.update(status="complete_engineering_attempts", finished_utc=utc(), source_after=freeze,
            input_hashes_after=verify_inputs(repo, protocol),
            rows_terminal=sum(r["status"] in {"complete", "failed"} for r in rows),
            numerical_audits_all_pass=all(r["status"] == "complete" for r in rows))
        status_bytes = json_bytes(status)
        hashes = {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != "status.json"}
        hashes["status.json"] = hashlib.sha256(status_bytes).hexdigest()
        complete = dict(status, artifact_sha256=hashes)
        complete_bytes = json_bytes(complete)
        projected = retained_bytes(output) - (output / "status.json").stat().st_size + len(status_bytes) + len(complete_bytes)
        if projected > output_limit:
            raise OSError("complete receipt would exceed the retained-output envelope")
        atomic(output / "status.json", status)
        checked_exclusive(output / "complete_receipt.json", complete_bytes, output_limit)
        return complete
    except BaseException as exc:
        status.update(status="failed_or_interrupted", finished_utc=utc(), error=f"{type(exc).__name__}: {exc}")
        atomic(output / "status.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--execute-study", action="store_true")
    for key in ["repo", "output"]:
        parser.add_argument("--"+key, type=Path)
    parser.add_argument("--expected-driver-sha256")
    parser.add_argument("--expected-protocol-sha256")
    args = parser.parse_args(argv)
    protocol, _ = load_protocol(args.protocol)
    if not args.execute_study:
        print(json.dumps(dict(status="metadata_only_no_project_arrays_or_solves", planned_runs=len(schedule(protocol)),
            driver_sha256=sha(Path(__file__)), protocol_sha256=sha(args.protocol))))
        return 0
    if not all([args.repo, args.output, args.expected_driver_sha256, args.expected_protocol_sha256]):
        parser.error("explicit reviewed source/protocol hashes, repository and fresh outer output required")
    result = run(args.repo, args.output, args.protocol, args.expected_driver_sha256, args.expected_protocol_sha256)
    return 0 if result["numerical_audits_all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
