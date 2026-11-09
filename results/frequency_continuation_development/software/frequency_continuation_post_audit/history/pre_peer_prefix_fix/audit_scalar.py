"""Independent stored-scalar audit; default never opens project numeric data.

No producer, controller, engine, WaveCert or807 imports and no PDE/optimizer
calls. Source/criteria and manufactured controls precede outcome inspection.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

METHODS = ("joint8", "low4_then8", "restart8", "original4")
ALL8 = (.25, .5, 1., 2., 2.5, 3., 3.5, 4.)
STAGES = {"joint8": (ALL8,), "low4_then8": (ALL8[:4], ALL8),
          "restart8": (ALL8, ALL8), "original4": (ALL8[4:],)}
CHAIN_SEED = b"bounded-exact-record-stream-v1\0"
QUALITY_KEYS = ("relative_squared_slowness_L2", "relative_physical_velocity_L2")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def json_load(path):
    if Path(path).suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(Path(path).read_bytes())


def number(v, *, nonnegative=True):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and (not nonnegative or v >= 0))


def count(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def sha_text(v):
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


class Checks:
    def __init__(self):
        self.count = self.numeric = 0
        self.issues = []

    def require(self, ok, where, detail=None):
        self.count += 1
        if not bool(ok):
            self.issues.append(dict(where=where, detail=detail))
        return bool(ok)

    def near(self, actual, recorded, rtol, atol, where):
        self.numeric += 1
        ok = number(actual, nonnegative=False) and number(recorded, nonnegative=False)
        return self.require(ok and abs(actual-recorded) <= atol+rtol*abs(actual), where,
            dict(independent=actual, recorded=recorded, rtol=rtol, atol=atol))


def safe_file(root, name, checks, where):
    try:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("missing or escaping artifact")
        return path
    except (OSError, TypeError, ValueError) as exc:
        checks.require(False, where, str(exc))
        return None


def verified_json(root, name, expected, checks, where):
    path = safe_file(root, name, checks, where)
    if path is None:
        return None
    if not checks.require(sha_text(expected) and file_sha(path) == expected, where+"/sha256"):
        return None
    try:
        return json_load(path)
    except (OSError, ValueError) as exc:
        checks.require(False, where+"/decode", str(exc))
        return None


def source_digest(repo):
    files = [repo/"pyproject.toml", *repo.glob("requirements*.txt"), *(repo/"src").rglob("*.py"),
        *(p for p in (repo/"configs").rglob("*") if p.is_file()),
        *(p for p in (repo/"scripts").rglob("*") if p.suffix in {".py", ".sh"}),
        *(repo/".github/workflows").glob("*.yml")]
    h = hashlib.sha256()
    for path in sorted(set(files)):
        h.update(path.relative_to(repo).as_posix().encode()+b"\0"+file_sha(path).encode()+b"\0")
    return dict(sha256=h.hexdigest(), files=len(set(files)))


def expected_schedule(protocol):
    rng, rows = random.Random(2026110802), []
    for repetition in range(3):
        for case in range(8):
            order = list(METHODS)
            rng.shuffle(order)
            for position, method in enumerate(order):
                rows.append(dict(run_index=len(rows), case_index=case, repetition=repetition,
                    order_position=position, method=method, identity=protocol["identities"][case]))
    return rows


def schedule_check(rows, planned, checks, label):
    checks.require(len(rows) == 96, label+"/full96")
    checks.require(all(all(count(row.get(k)) for k in ("run_index", "case_index", "repetition", "order_position"))
        for row in rows), label+"/identity_indices_are_nonboolean_integers")
    for i, expected in enumerate(planned):
        if i >= len(rows):
            continue
        checks.require(all(rows[i].get(k) == v for k, v in expected.items()), label+f"/{i}/identity")
    keys = [(r.get("case_index"), r.get("repetition"), r.get("method")) for r in rows]
    checks.require(len(set(keys)) == 96, label+"/unique96")
    quads, groups = Counter(), Counter()
    for case, repetition, method in keys:
        quads[case, repetition] += 1
        groups[method, case] += 1
    checks.require(len(quads) == 24 and all(v == 4 for v in quads.values()), label+"/24quadruples")
    checks.require(len(groups) == 32 and all(v == 3 for v in groups.values()), label+"/32three-rep-groups")


def count_tree(value):
    if not isinstance(value, dict):
        raise ValueError("counter dictionary required; unknown is not zero")
    out = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("counter keys must be strings")
        if isinstance(item, dict):
            out[key] = count_tree(item)
        elif count(item):
            out[key] = item
        else:
            raise ValueError("invalid counter: "+key)
    return out


def sum_trees(items):
    out = {}
    for value in items:
        for k, v in count_tree(value).items():
            if isinstance(v, dict):
                out[k] = sum_trees([out.get(k, {}), v])
            elif isinstance(out.get(k), dict):
                raise ValueError("counter tree type conflict")
            else:
                out[k] = out.get(k, 0)+v
    return out


def monotone(before, after):
    for key in set(before) | set(after):
        a, b = before.get(key, 0), after.get(key, 0)
        if isinstance(a, dict) or isinstance(b, dict):
            if not isinstance(a, dict) and a != 0 or not isinstance(b, dict) and b != 0:
                return False
            if not monotone(a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {}):
                return False
        elif b < a:
            return False
    return True


def record_chain(records):
    digest = hashlib.sha256(CHAIN_SEED).digest()
    for i, record in enumerate(records, 1):
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        digest = hashlib.sha256(digest+i.to_bytes(8, "big")+len(raw).to_bytes(8, "big")+raw).digest()
    return digest.hex()


def stream_check(records, summary, checks, label):
    if not checks.require(isinstance(records, list) and isinstance(summary, dict), label+"/schema"):
        return False
    fields = ("total_count", "retained_count", "omitted_count", "finalized_count")
    if not checks.require(all(count(summary.get(k)) for k in fields), label+"/integer_counts"):
        return False
    pending = summary.get("pending_record_unsealed")
    checks.require(type(pending) is bool, label+"/pending_flag")
    p = int(pending is True)
    checks.require(summary.get("head_limit") == summary.get("tail_limit") == 8, label+"/8plus8")
    checks.require(summary["retained_count"] <= 16 and len(records) == summary["retained_count"]+p,
        label+"/bounded_samples")
    checks.require(summary["total_count"] == summary["finalized_count"]+p
        and summary["omitted_count"] == summary["finalized_count"]-summary["retained_count"], label+"/count_math")
    checks.require(sha_text(summary.get("chain_sha256")), label+"/chain_shape")
    checks.require((summary.get("pending_record") is not None) is bool(p), label+"/pending_record")
    if p:
        checks.require(bool(records) and records[-1] == summary["pending_record"], label+"/pending_sample_binding")
    finalized = records[:-1] if p else records
    if summary["omitted_count"] == 0:
        checks.require(record_chain(finalized) == summary["chain_sha256"], label+"/all_retained_chain_recomputed")
    # A chain with an omitted middle cannot be independently reconstructed.
    return not p


def manual_refinement(raw):
    """49 half-index velocity samples, explicit edge extension and scalar math."""
    if getattr(raw, "shape", None) != (24, 24):
        raise ValueError("raw24x24 native-m required")
    velocity = [[1/math.sqrt(float(raw[z, x])) for x in range(24)] for z in range(24)]
    fine = []
    for z in range(49):
        zz = .5*(z+1)
        iz, fz = math.floor(zz), zz-math.floor(zz)
        for x in range(49):
            xx = .5*(x+1)
            ix, fx = math.floor(xx), xx-math.floor(xx)
            fine.append(math.fsum(wz*wx*velocity[max(0, min(23, rz-1))][max(0, min(23, rx-1))]
                for rz, wz in [(iz, 1-fz), (iz+1, fz)]
                for rx, wx in [(ix, 1-fx), (ix+1, fx)]))
    return fine, [1/(v*v) for v in fine]


def scalar_quality(model, truth):
    m, t = [float(x) for x in model], [float(x) for x in truth]
    if len(m) != len(t) or not m or any(not number(x) or x <= 0 for x in m+t):
        raise ValueError("matching finite positive native-m vectors required")
    em = math.sqrt(math.fsum((a-b)**2 for a, b in zip(m, t, strict=True)) / math.fsum(b*b for b in t))
    v, vt = [1/math.sqrt(x) for x in m], [1/math.sqrt(x) for x in t]
    ev = math.sqrt(math.fsum((a-b)**2 for a, b in zip(v, vt, strict=True)) / math.fsum(b*b for b in vt))
    return dict(zip(QUALITY_KEYS, (em, ev), strict=True))


def runtime_check(row, trace, checks, label):
    timing = row.get("timing", {})
    started, returned = trace.get("global_started_perf_counter"), row.get("caller_observed_return_stamp")
    valid_numbers = number(started, nonnegative=False) and number(returned, nonnegative=False)
    wall = returned-started if valid_numbers else None
    inner = timing.get("inner_execute_timed_return_seconds")
    checks.require(valid_numbers and number(wall) and number(inner), label+"/finite_return_stamps")
    if not valid_numbers or not number(wall) or not number(inner):
        return False, wall
    checks.near(wall, timing.get("actual_api_return_seconds"), 0., 1e-8, label+"/outer_wall")
    checks.near(max(0., wall-inner), timing.get("measured_helper_return_tail_seconds"), 0., 1e-8, label+"/tail_subset")
    checks.require(inner <= wall+1e-9 and timing.get("api_return_cap_seconds") == 8., label+"/nesting_and_cap")
    checks.near(max(0., wall-8.), timing.get("overshoot_seconds"), 0., 1e-8, label+"/overshoot")
    method_valid = trace.get("method_valid_before_return_gate") is True
    last_seen = trace.get("last_observed_absolute_clock")
    expected_inner = (method_valid and inner <= 8. and number(last_seen, nonnegative=False)
                      and started+inner >= last_seen)
    inner_gate = timing.get("inner_runtime_gate", {})
    checks.require(inner_gate.get("eligible") is expected_inner, label+"/inner_gate")
    checks.require(inner_gate.get("event_seconds") == (inner if expected_inner else None), label+"/inner_event")
    expected = bool(expected_inner and inner_gate.get("eligible") is True and wall <= 8.
                    and inner <= wall)  # Strict8: NO timing tolerance permits credit.
    checks.require(timing.get("runtime_eligible") is expected, label+"/outer_gate_cannot_rescue")
    checks.require(timing.get("event_seconds") == (wall if expected else None), label+"/outer_event")
    return expected, wall


def cost_check(trace, checks, label):
    owners, providers = trace.get("owners", []), trace.get("preparation_costs", {})
    labels = [r.get("label") for r in owners]
    checks.require(len(labels) == len(set(labels)), label+"/unique_serialized_owner_labels")
    owner_complete = not trace.get("in_flight_work_unknown") and all(r.get("costs_complete") is True for r in owners)
    prep_complete = (bool(providers) and not trace.get("preparation_in_flight_counts_unknown")
                     and all(r.get("complete") is True for r in providers.values()))
    checks.require(all(isinstance(r.get("latest_costs"), dict) for r in owners if r.get("costs_complete") is True),
        label+"/completed_owner_requires_snapshot")
    checks.require(all(isinstance(r.get("latest_counts"), dict) for r in providers.values() if r.get("complete") is True),
        label+"/completed_preparation_requires_snapshot")
    try:
        owner_sum = sum_trees([r["latest_costs"] for r in owners if r.get("latest_costs") is not None])
        prep_sum = sum_trees([r["latest_counts"] for r in providers.values() if r.get("latest_counts") is not None])
        checks.require(trace.get("owner_costs_summed_once") == (owner_sum if owner_complete else None), label+"/owner_sum_once")
        checks.require(trace.get("preparation_costs_separate") == (prep_sum if prep_complete else None), label+"/prep_separate")
        checks.require(trace.get("owner_known_counts_partial_sum") == (None if owner_complete else owner_sum), label+"/owner_unknown_not_zero")
        checks.require(trace.get("preparation_known_counts_partial_sum") == (None if prep_complete else prep_sum), label+"/prep_unknown_not_zero")
        checks.require(trace.get("costs_complete") is bool(owner_complete and prep_complete), label+"/cost_completeness")
        if prep_complete:
            expected = dict(factorizations=8, forward_rhs=48, tangent_rhs=0, adjoint_rhs=0,
                stability_rhs=0, dense_inverse_rhs=0, svd_evaluations=0, cache_hits=0, analytic_bound_evaluations=0)
            checks.require(providers.get("all8_observation_generation", {}).get("latest_counts") == expected,
                label+"/actual8LU48RHS_preparation")
        return owner_sum, prep_sum
    except (TypeError, ValueError, KeyError) as exc:
        checks.require(False, label+"/invalid_counter_tree", str(exc))
        return None, None


def target_check(row, trace, eligible, wall, checks, label):
    initial, final = trace.get("common_initial_full8"), trace.get("common_final_full8")
    a, b = (None if initial is None else initial.get("objective")), (None if final is None else final.get("objective"))
    ratio = b/a if number(a) and a > 0 and number(b) else None
    ratio = ratio if ratio is not None and math.isfinite(ratio) else None
    checks.require(trace.get("common_full8_ratio") == ratio, label+"/common8_ratio")
    if row.get("status") == "complete":
        checks.require(row.get("common_full8_ratio") == ratio, label+"/row_ratio")
        targets = row.get("returned_targets", [])
        checks.require(len(targets) == 2, label+"/two_returned_targets")
        for target, recorded in zip((.5, .25), targets, strict=False):
            reached = ratio is not None and ratio <= target
            checks.require(recorded.get("ratio") == target and recorded.get("reached") is reached,
                label+f"/raw_returned_ratio{target}")
            checks.require(recorded.get("event_seconds") == (wall if eligible and reached else None),
                label+f"/eligible_return_event{target}_not_first_hit")
    return ratio


def trace_check(trace, method, protocol, checks, label):
    """Check available descriptors; never invent omitted intermediate states."""
    expected = STAGES[method]
    stages, owners = trace.get("stages", []), trace.get("owners", [])
    checks.require(trace.get("arm") == method and trace.get("settings") == protocol["controller_settings"], label+"/arm_and_settings")
    checks.require(sha_text(trace.get("available8_data_sha256")), label+"/controller_data_encoding_sha")
    checks.require(len(stages) == len(expected), label+"/stage_denominator")
    expected_labels = {"stage1": ("policy", expected[0])}
    if len(expected) == 2:
        expected_labels["stage2"] = ("policy", expected[1])
    if method in {"low4_then8", "original4"}:
        expected_labels["common8_metric"] = ("metric_only", ALL8)
    by_label = {r.get("label"): r for r in owners}
    for owner in owners:
        name = owner.get("label")
        checks.require(name in expected_labels, label+"/owner_role/"+str(name))
        if name not in expected_labels:
            continue
        role, frequencies = expected_labels[name]
        checks.require(owner.get("role") == role and owner.get("frequencies_hz") == list(frequencies), label+"/owner_band/"+name)
        if owner.get("status") == "constructed_cold":
            contract = owner.get("contract", {})
            checks.require(contract.get("operator_base_record", {}).get("configuration") == protocol["physics"], label+"/owner_config/"+name)
            checks.require(contract.get("reference_source_sha256") == "caf51b58da24ed6d3633bda4eaaab804714f5c7932d9d5f343644763bc158131", label+"/native_source/"+name)
            checks.require(all(sha_text(contract.get(k)) for k in ("operator_base_sha256", "geometry_sha256", "data_sha256", "survey_sha256")), label+"/owner_fingerprints/"+name)
    constructed = [r["contract"] for r in owners if r.get("status") == "constructed_cold"]
    checks.require(len({c.get("operator_base_sha256") for c in constructed}) <= 1, label+"/same_operator_base")
    source = trace.get("source_binding", {})
    checks.require(source.get("production_sha256") == protocol["production_source_sha256"]
        and source.get("engine_source_sha256") == protocol["local_dependency_sha256"]["bounded_exact.py"], label+"/runtime_source_binding")
    checks.require(trace.get("available_frequencies_hz") == list(ALL8) and trace.get("bounds_m") == [1/36, 1.], label+"/all8_native_m")
    checks.require(trace.get("truth_input") is False and trace.get("checkpoint_vectors_retained") is False, label+"/scope_flags")
    last = None
    all_invocations = all_points = 0
    method_failure = False
    unsealed = False
    for index, frequencies in enumerate(expected):
        if index >= len(stages):
            break
        stage, key = stages[index], label+f"/stage{index+1}"
        deadline = 3.75 if len(expected) == 2 and index == 0 else 7.5
        cap = 2048 if len(expected) == 2 else 4096
        checks.require(stage.get("index") == index and stage.get("frequencies_hz") == list(frequencies)
            and stage.get("absolute_deadline_seconds") == deadline and stage.get("maximum_steps") == cap, key+"/contract")
        engine = stage.get("engine_result")
        if engine is None:
            checks.require(stage.get("status") in {"unattempted", "unattempted_after_failure", "unattempted_no_prior_paid_state", "unattempted_phase_deadline", "failed_controller", "interrupted"}, key+"/missing_engine_status")
            continue
        checkpoints, steps = engine.get("checkpoints", []), engine.get("steps", [])
        finalized_cp = stream_check(checkpoints, engine.get("checkpoint_trace"), checks, key+"/checkpoint_stream")
        finalized_steps = stream_check(steps, engine.get("step_trace"), checks, key+"/step_stream")
        unsealed = unsealed or not finalized_cp or not finalized_steps or engine.get("no_completed_return_claim") is True
        inv, points, alias = (engine.get(k) for k in ("native_callback_invocations", "native_callback_point_records", "native_callback_count"))
        if checks.require(count(inv) and count(points) and count(alias) and points <= inv == alias <= cap, key+"/all_callback_counts"):
            all_invocations += inv
            all_points += points
            if "native_callback_invocations" in stage:
                checks.require(stage["native_callback_invocations"] == inv and stage.get("native_callback_events") == inv
                    and stage.get("native_callback_point_records") == points, key+"/stage_aggregate_counts")
        retained_points = sum("native_callback" in s for s in steps)
        checks.require(not count(points) or retained_points <= points, key+"/retained_points_are_subset")
        if "retained_native_callback_events" in stage:
            checks.require(stage["retained_native_callback_events"] == retained_points, key+"/retained_point_count")
        last_attempt = engine.get("last_native_callback_attempt")
        if count(inv):
            checks.require(last_attempt is None if inv == 0 else isinstance(last_attempt, dict)
                and last_attempt.get("sequence") == inv and sha_text(last_attempt.get("candidate_sha256")), key+"/last_invocation")
        eligible = []
        previous_time = -1.
        for cp in checkpoints:
            age = cp.get("elapsed_seconds")
            checks.require(number(age) and age >= previous_time, key+"/chronological_retained_checkpoint")
            if number(age):
                previous_time = age
            checks.require(type(cp.get("within_budget")) is bool and cp.get("within_budget") is bool(number(age) and age <= deadline), key+"/within_phase_flag")
            checks.require("model" not in cp and sha_text(cp.get("model_sha256")) and number(cp.get("objective")), key+"/checkpoint_descriptor_only")
            if cp.get("within_budget") is True and number(age) and age <= deadline:
                eligible.append(cp)
        if "eligible_retained_checkpoint_count" in stage:
            checks.require(stage.get("eligible_retained_checkpoint_count") == len(eligible)
                and stage.get("eligible_checkpoint_count") == len(eligible)
                and "retained" in stage.get("eligible_checkpoint_count_scope", ""), key+"/retained_scope")
        if checkpoints:
            checks.require(checkpoints[0].get("kind") == "initial" and checkpoints[0].get("model_sha256") == stage.get("start_model_sha256")
                == engine.get("initial_model_sha256"), key+"/initial_model_binding")
        expected_start = trace.get("initial_model_sha256") if index == 0 else None if last is None else last["model_sha256"]
        if "start_model_sha256" in stage:
            checks.require(stage["start_model_sha256"] == expected_start, key+"/model_only_handoff")
        if eligible:
            checks.require(engine.get("final_model_sha256") == eligible[-1]["model_sha256"]
                and engine.get("final_objective") == eligible[-1]["objective"], key+"/last_paid_point_not_best_objective")
            last = dict(stage_index=index, model_sha256=eligible[-1]["model_sha256"], elapsed_seconds=eligible[-1]["elapsed_seconds"])
        policy_owner = by_label.get(f"stage{index+1}", {})
        checks.require(policy_owner.get("status") == "constructed_cold", key+"/actual_policy_owner_required")
        snapshot = engine.get("total_online_costs", {}).get("oracle")
        latest = policy_owner.get("latest_costs")
        if snapshot is not None and latest is not None:
            try:
                checks.require(monotone(count_tree(snapshot), count_tree(latest)), key+"/engine_snapshot_subset_of_final_owner")
            except ValueError as exc:
                checks.require(False, key+"/snapshot_counter_schema", str(exc))
        if engine.get("status") == "failed" or stage.get("method_failed") is True or engine.get("costs_complete") is not True:
            method_failure = True
        for step_index, step in enumerate(steps):
            library_finalized = stream_check(step.get("library_evaluations", []), step.get("library_evaluations_trace"),
                checks, key+f"/step{step_index}/library_stream")
            unsealed = unsealed or not library_finalized
        global_library = engine.get("all_library_evaluation_trace", {})
        checks.require(count(global_library.get("total_count")) and sha_text(global_library.get("chain_sha256")), key+"/all_library_commitment")
        if engine.get("status") != "unfinalized_engine_prefix":
            checks.require(engine.get("scipy_maxfun_default") == 15000 and engine.get("scipy_maxfun_passed_explicitly") is False, key+"/unpassed_maxfun_limit")
    origin = trace.get("final_model_origin")
    checks.require(origin == last, label+"/last_chronological_origin")
    checks.require(trace.get("final_model_sha256") == (None if last is None else last["model_sha256"]), label+"/final_model_sha")
    common_a, common_b = trace.get("common_initial_full8"), trace.get("common_final_full8")
    common_valid = isinstance(common_a, dict) and isinstance(common_b, dict)
    if common_valid:
        checks.require(common_a.get("model_sha256") == trace.get("initial_model_sha256")
            and common_b.get("model_sha256") == trace.get("final_model_sha256"), label+"/common8_model_headers")
        checks.require(all(common_a.get(k) == common_b.get(k) and sha_text(common_a.get(k))
            for k in ("geometry_sha256", "data_sha256")), label+"/common8_same_survey_data")
        common_valid = number(common_a.get("objective")) and number(common_b.get("objective"))
        initial_label = "stage1" if expected[0] == ALL8 else "common8_metric"
        initial_contract = by_label.get(initial_label, {}).get("contract", {})
        full8_contracts = [r.get("contract", {}) for r in owners
            if r.get("status") == "constructed_cold" and r.get("frequencies_hz") == list(ALL8)]
        checks.require(all(common_a.get(k) == initial_contract.get(k) for k in ("geometry_sha256", "data_sha256")),
            label+"/initial_common8_actual_owner_binding")
        checks.require(bool(full8_contracts) and all(common_b.get(k) == full8_contracts[-1].get(k)
            for k in ("geometry_sha256", "data_sha256")), label+"/final_common8_actual_owner_binding")
    expected_valid = (trace.get("status") == "completed_paid_state" and last is not None and common_valid
        and trace.get("costs_complete") is True and not method_failure and not unsealed
        and trace.get("in_flight_work_unknown") is False)
    checks.require(trace.get("method_valid_before_return_gate") is bool(expected_valid), label+"/method_valid_not_unsealed_or_failed")
    return dict(method_valid=bool(expected_valid), invocations=all_invocations, point_records=all_points,
        unsealed_prefix=unsealed, method_failure=method_failure, last_origin=last)


def load_references(repo, protocol, checks):
    """Execute only after terminal GO: pinned validation inputs, not holdouts."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    blobs = {}
    for name, expected in protocol["inputs_sha256"].items():
        path = safe_file(repo, name, checks, "input/"+name)
        if path is None:
            raise ValueError("required raw input missing")
        raw = path.read_bytes()
        if not checks.require(hashlib.sha256(raw).hexdigest() == expected, "input/"+name+"/sha256"):
            raise ValueError("required raw input changed; no numeric decode")
        blobs[name] = raw
    prefix = "data/reference/subsurfacegen_parent_split/"
    metadata = list(csv.DictReader(io.StringIO(blobs[prefix+"validation_metadata.csv"].decode())))
    manifest = list(csv.DictReader(io.StringIO(blobs[prefix+"split_manifest.csv"].decode())))
    with np.load(io.BytesIO(blobs[prefix+"validation.npz"]), allow_pickle=False) as d:
        models, sources, frequencies = d["models"], d["source_indices"], d["frequencies_hz"]
        checks.require(models.shape == (192, 24, 24) and models.dtype == np.float32 and len(metadata) == 192
            and d["shape"].tolist() == [24, 24] and float(d["spacing"]) == .05
            and np.array_equal(d["receiver_indices"], protocol["geometry"]["original_receiver_indices"]), "input/schema")
        refs = []
        pairs = {(s, f) for s in protocol["geometry"]["original_source_indices"] for f in (2.5, 3., 3.5, 4.)}
        for case, identity in enumerate(protocol["identities"]):
            positions = [i for i, row in enumerate(metadata) if row["slice_id"] == identity["slice_id"]]
            if not checks.require(len(positions) == 24 and positions[0] == identity["array_index"], f"input/case{case}/positions"):
                raise ValueError("raw case identity missing")
            checks.require({(int(sources[i]), float(frequencies[i])) for i in positions} == pairs, f"input/case{case}/old4_source_coverage")
            for i in positions:
                checks.require(all(metadata[i].get(k) == v for k, v in dict(split="validation", model_id=identity["parent_id"],
                    source_file=identity["source_file"], source_sha256=identity["source_sha256"]).items())
                    and np.array_equal(models[i], models[positions[0]]), f"input/case{case}/repeat_identity{i}")
            selected = [r for r in manifest if r["slice_id"] == identity["slice_id"]]
            checks.require(len(selected) == 1 and selected[0]["split"] == "validation" and selected[0]["model_id"] == identity["parent_id"], f"input/case{case}/manifest")
            raw_model = np.asarray(models[positions[0]], dtype=np.float64)
            if not np.isfinite(raw_model).all() or (raw_model <= 0).any():
                raise ValueError("nonpositive raw model")
            velocity, truth = manual_refinement(raw_model)
            smoothed = gaussian_filter(np.asarray(velocity).reshape(49, 49), sigma=3.2, mode="reflect")
            initial = [1/(float(v)*float(v)) for v in smoothed.ravel()]
            refs.append(dict(truth=truth, initial=initial,
                raw_model_sha256=hashlib.sha256(raw_model.astype("<f8").tobytes()).hexdigest(),
                manual_truth_sha256=hashlib.sha256(np.asarray(truth, dtype="<f8").tobytes()).hexdigest()))
    return refs


def array_sha(a, np):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


def quality_check(models, reference, recorded, criteria, checks, label):
    own = {}
    for key, model in models.items():
        if not checks.require(model.shape == (2401,) and str(model.dtype) == "float64"
            and all(number(float(v)) and 1/36 <= float(v) <= 1. for v in model), label+"/"+key+"/native_vector_domain"):
            continue
        if key == "initial_model":
            fraction = max(abs(float(a)-b)/(criteria["numeric"]["initial_model_atol"]
                +criteria["numeric"]["initial_model_rtol"]*abs(b)) for a, b in zip(model, reference["initial"], strict=True))
            checks.require(fraction <= 1., label+"/direct_prepared_initial_reference", fraction)
        value = scalar_quality(model, reference["truth"])
        out_key = "initial" if key == "initial_model" else "final"
        own[out_key] = value
        if out_key in recorded:
            for q in QUALITY_KEYS:
                checks.near(value[q], recorded[out_key].get(q), criteria["numeric"]["quality_rtol"], criteria["numeric"]["quality_atol"], label+"/"+out_key+"/"+q)
    return own


def embedded_check(audit, trace, models, own_quality, criteria, checks, label):
    """Audit recorded scalar arithmetic/coverage, not the independent PDE again."""
    errors = audit.get("observation_relative_L2", {})
    checks.require(set(errors) == {str(f) for f in ALL8}, label+"/all8_observations")
    numerical_pass = not audit.get("failures")
    checks.require(numerical_pass, label+"/reported_independent_numerical_failures_disable_credit")
    if numerical_pass:
        checks.require(all(number(v) and v <= 1e-9 for v in errors.values())
            and number(audit.get("maximum_relative_linear_residual"))
            and audit["maximum_relative_linear_residual"] <= 1e-9, label+"/recorded_residual_and_observation_gates")
    records = audit.get("records", {})
    baseline = records.get("initial", {}).get("objective")
    scale = max(baseline, 1e-12) if number(baseline) else 1e-12
    for key in ("initial", "final"):
        model_key = key+"_model"
        if model_key not in models:
            continue
        record = records.get(key, {})
        checks.require(record.get("model_sha256") == trace.get(model_key+"_sha256"), label+"/"+key+"/model_sha")
        values = record.get("per_frequency_objectives", [])
        checks.require(len(values) == 8 and all(number(v) for v in values), label+"/"+key+"/8Phi_terms")
        if len(values) == 8 and all(number(v) for v in values):
            checks.near(math.fsum(values), record.get("objective"), 2e-9, 2e-12*scale, label+"/"+key+"/Phi_scalar_sum")
        if key in own_quality:
            for q in QUALITY_KEYS:
                checks.near(own_quality[key][q], record.get(q), criteria["numeric"]["quality_rtol"],
                    criteria["numeric"]["quality_atol"], label+"/"+key+"/independent807_"+q)
        receipt = trace.get("common_"+key+"_full8")
        if receipt is not None and numerical_pass:
            checks.near(record.get("objective"), receipt.get("objective"), 2e-9, 2e-12*scale, label+"/"+key+"/recorded807_vs_nativePhi")
    #807 precomputes its independently rounded initial once; that cache entry
    #may or may not coincide bitwise with a saved initial/final. Do not guess.
    distinct = len({r.get("model_sha256") for r in records.values()})
    factors, rhs = audit.get("factorizations"), audit.get("forward_rhs")
    checks.require(count(factors) and count(rhs) and 8*(1+max(1, distinct)) <= factors <= 8*(2+distinct)
        and factors % 8 == 0 and rhs == 6*factors, label+"/offline_costs_separate_807_cache_range")
    return numerical_pass


def row_audit(row, planned, run, reference, protocol, criteria, checks, np):
    label = f"run{planned['run_index']:03d}"
    issues_before = len(checks.issues)
    record = dict(**planned, recorded_status=row.get("status", "missing"), recorded_final_study_eligible=row.get("final_study_eligible"),
        recorded_primary_quality=row.get("primary_quality"), verified_credit_eligible=False,
        verified_quality=None, own_quality={}, common_full8_ratio=None, actual_api_return_seconds=None,
        method_failure=False, unsealed_prefix=False, targets=[])
    if row.get("status") == "unattempted":
        checks.require(row.get("final_study_eligible") is False and row.get("primary_quality") is None, label+"/unattempted_not_credited")
        return record
    trace = verified_json(run, row.get("trajectory_artifact"), row.get("trajectory_sha256"), checks, label+"/trajectory")
    if not isinstance(trace, dict):
        checks.require(row.get("final_study_eligible") is not True and row.get("primary_quality") is None, label+"/missing_trace_no_credit")
        return record
    record["controller_status"] = trace.get("status")
    record["controller_available8_data_sha256"] = trace.get("available8_data_sha256")
    is_returned = isinstance(row.get("timing"), dict)
    models = {}
    if row.get("models_artifact") is not None:
        path = safe_file(run, row["models_artifact"], checks, label+"/models")
        if path is not None and checks.require(file_sha(path) == row.get("models_sha256"), label+"/models/sha256"):
            with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
                checks.require(set(archive.files) <= {"initial_model", "final_model"}, label+"/only2native_vectors")
                models = {k: archive[k].copy() for k in archive.files}
    for key, value in models.items():
        digest = array_sha(value, np)
        checks.require(digest == trace.get(key+"_sha256"), label+"/"+key+"/trace_bytes")
    binding = row.get("common_binding")
    record["common_binding"] = binding
    if binding is not None:
        checks.require(binding.get("raw_model_sha256") == reference["raw_model_sha256"], label+"/raw_model_binding")
        checks.require(binding.get("initial_native_m_sha256") == trace.get("initial_model_sha256"), label+"/direct_initial_binding")
        checks.require(all(sha_text(binding.get(k)) for k in ("truth_sha256", "available8_observations_sha256")), label+"/data_fingerprint_shapes")
    if is_returned:
        state = trace_check(trace, planned["method"], protocol, checks, label+"/trace")
        record.update(method_failure=state["method_failure"], unsealed_prefix=state["unsealed_prefix"],
            callback_invocations=state["invocations"], callback_point_records=state["point_records"])
        eligible_time, wall = runtime_check(row, trace, checks, label+"/timing")
        record["actual_api_return_seconds"] = wall
        record["owner_counts"], record["preparation_counts"] = cost_check(trace, checks, label+"/costs")
        checks.require(row.get("controller_status") == trace.get("status"), label+"/controller_status")
        checks.require(row.get("input_hashes_after") == protocol["inputs_sha256"] if row.get("status") == "complete" else True,
            label+"/per_run_input_integrity")
        for summary, stage in zip(row.get("stage_summaries", []), trace.get("stages", []), strict=False):
            checks.require(all(summary[k] == stage.get(k) for k in summary), label+"/stage_summary_subset")
    else:
        eligible_time, state = False, dict(method_valid=False)
        record["prefix_only_not_API_return"] = True
        checks.require(row.get("status") in {"failed", "interrupted", "attempted"}, label+"/nonreturn_status")
    audit = None
    if row.get("audit_artifact"):
        audit = verified_json(run, row["audit_artifact"], row.get("audit_sha256"), checks, label+"/embedded807")
    record["own_quality"] = quality_check(models, reference, {} if audit is None else audit.get("own_quality", {}),
        criteria, checks, label+"/scalar_quality")
    audit_pass = embedded_check(audit, trace, models, record["own_quality"], criteria, checks, label+"/embedded807") if isinstance(audit, dict) else False
    row_complete = row.get("status") == "complete" and row.get("integrity_valid") is True
    expected_eligible = bool(row_complete and eligible_time and state["method_valid"] and audit_pass)
    checks.require(row.get("final_study_eligible") is expected_eligible, label+"/joint_terminal_eligibility")
    if expected_eligible:
        primary = row.get("primary_quality", {})
        for q in QUALITY_KEYS:
            actual = record["own_quality"].get("final", {}).get(q)
            checks.near(actual, primary.get(q), criteria["numeric"]["quality_rtol"], criteria["numeric"]["quality_atol"], label+"/primary_"+q)
    else:
        checks.require(row.get("primary_quality") is None, label+"/ineligible_quality_null")
    ratio = target_check(row, trace, expected_eligible, record["actual_api_return_seconds"], checks, label+"/targets")
    record["common_full8_ratio"] = ratio
    record["local_credit_conditions_pass"] = bool(expected_eligible and len(checks.issues) == issues_before)
    record["local_audit_issue_count"] = len(checks.issues)-issues_before
    record["targets"] = [dict(ratio=t, returned_ratio_reached=ratio is not None and ratio <= t,
        verified_event_seconds=None) for t in (.5, .25)]
    return record


def median(values):
    values = sorted(values)
    n = len(values)
    return None if not n else values[n//2] if n % 2 else .5*(values[n//2-1]+values[n//2])


def summarize(records, global_credit):
    groups, arms = [], {}
    for record in records:
        record["verified_credit_eligible"] = bool(global_credit and record.get("local_credit_conditions_pass"))
        if record["verified_credit_eligible"]:
            record["verified_quality"] = record["own_quality"].get("final")
            for target in record.get("targets", []):
                if target["returned_ratio_reached"]:
                    target["verified_event_seconds"] = record["actual_api_return_seconds"]
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        arms[method] = dict(planned_runs=24, recorded_status_counts=dict(Counter(r["recorded_status"] for r in rows)),
            verified_eligible=sum(r["verified_credit_eligible"] for r in rows),
            returned_targets={str(t): dict(events=sum(r["verified_credit_eligible"] and r["common_full8_ratio"] is not None
                and r["common_full8_ratio"] <= t for r in rows), planned_denominator=24) for t in (.5, .25)})
        for case in range(8):
            selected = [r for r in rows if r["case_index"] == case]
            good = [r for r in selected if r["verified_credit_eligible"] and r["verified_quality"] is not None]
            full = len(selected) == len(good) == 3 and {r["repetition"] for r in selected} == {0, 1, 2}
            groups.append(dict(method=method, case_index=case, planned_repetitions=3, retained_runs=len(selected),
                verified_repetitions=len(good), median_quality={q: median([r["verified_quality"][q] for r in good]) if full else None for q in QUALITY_KEYS},
                all3_required=full))
    return dict(full_run_denominator=96, per_method_denominator=24, case_repetition_quadruples=24,
        case_method_groups=32, reused_cases=8, arms=arms, groups=groups,
        no_successful_only_winner=True, independent_site_count=None, pvalues=None, physical_acceptance=None)


def protocol_check(protocol, criteria, checks):
    checks.require(protocol.get("methods") == list(METHODS) and protocol.get("available_frequencies_hz") == list(ALL8), "protocol/fixed4arms8frequencies")
    checks.require(protocol.get("schedule", {}).get("total_runs") == 96 and len(protocol.get("identities", [])) == 8
        and protocol["schedule"].get("seed") == 2026110802, "protocol/fixed96schedule")
    checks.require(protocol.get("controller_dependencies_pending_review") is False, "protocol/activated_source_review")
    for filename, digest in criteria["source_pins"].items():
        if filename != "production":
            checks.require(protocol.get("local_dependency_sha256", {}).get(filename) == digest, "protocol/pin/"+filename)
    checks.require(protocol.get("production_source_sha256") == criteria["source_pins"]["production"], "protocol/production9f")
    checks.require(protocol.get("physics") == dict(shape=[49, 49], spacing=.025, damping_width=6, damping_strength=2.), "protocol/native_physics")
    checks.require(protocol.get("controller_settings", {}).get("single_stage_maximum_steps") == 4096
        and protocol["controller_settings"].get("per_stage_maximum_steps") == 2048, "protocol/safety_caps")
    checks.require(protocol.get("targets", {}).get("ratios") == [.5, .25] and protocol.get("competition_evidence") is False
        and protocol.get("student_final_research_design") is False, "protocol/software_scope")


def manifest_check(run, complete, checks):
    manifest = complete.get("artifact_sha256", {})
    actual = {p.name: file_sha(p) for p in run.iterdir() if p.is_file() and p.name != "complete_receipt.json"}
    checks.require(manifest == actual, "run/full_artifact_manifest", dict(expected_files=len(manifest), actual_files=len(actual)))
    for name, digest in manifest.items():
        p = safe_file(run, name, checks, "manifest/"+name)
        if p is not None:
            checks.require(file_sha(p) == digest, "manifest/"+name+"/sha256")
    return actual


def complete_exit_check(exit_code, complete, checks):
    """The pinned producer main returns0 for all audits passed, otherwise2."""
    all_passed = complete.get("numerical_audits_all_pass")
    expected = 0 if all_passed is True else 2 if all_passed is False else None
    return checks.require(expected is not None and isinstance(exit_code, int)
        and not isinstance(exit_code, bool) and exit_code == expected,
        "terminal/actual_exit_matches_pinned_driver_contract",
        dict(actual_exit_code=exit_code, expected_exit_code=expected))


def collect_records(planned, rows, run, repo, protocol, criteria, checks, numeric_authorized):
    """Fail closed before even importing NumPy or decoding reference/model arrays."""
    if not numeric_authorized:
        records = []
        for i, expected in enumerate(planned):
            row = rows[i] if i < len(rows) else {}
            records.append(dict(expected, recorded_status=row.get("status", "missing"),
                recorded_final_study_eligible=row.get("final_study_eligible"),
                metadata_only=True, numeric_data_decoded=False,
                verified_credit_eligible=False, verified_quality=None, own_quality={},
                targets=[], common_full8_ratio=None, local_credit_conditions_pass=False,
                noncredit_reason="incomplete_or_invalid_terminal_source_manifest_preflight"))
        return [], records
    import numpy as np
    references = load_references(repo, protocol, checks)
    records, common = [], {}
    for i, expected in enumerate(planned):
        row = rows[i] if i < len(rows) else dict(expected, status="missing", final_study_eligible=False, primary_quality=None)
        try:
            record = row_audit(row, expected, run, references[expected["case_index"]], protocol, criteria, checks, np)
        except Exception as exc:
            checks.require(False, f"run{i:03d}/audit_exception", f"{type(exc).__name__}: {exc}")
            record = dict(expected, recorded_status=row.get("status", "missing"), verified_credit_eligible=False,
                verified_quality=None, own_quality={}, targets=[], common_full8_ratio=None,
                local_credit_conditions_pass=False, error=f"{type(exc).__name__}: {exc}")
        binding = record.get("common_binding")
        if binding is not None:
            case = expected["case_index"]
            if case in common:
                checks.require(common[case] == (binding, record.get("controller_available8_data_sha256")),
                    f"run{i:03d}/same_case_common_initial_truth_data")
            else:
                common[case] = (binding, record.get("controller_available8_data_sha256"))
        records.append(record)
    return references, records


def audit_execute(args, criteria):
    """The sole project-data entry point; root terminal GO must precede it."""
    run, repo, output = args.run.resolve(), args.repo.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(run) or output.is_relative_to(repo) or output.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("fresh OUTER output outside run/repository/frozen auditor required")
    protocol = json_load(args.protocol)
    if file_sha(args.protocol) != args.expected_protocol_sha256 or file_sha(args.terminal_observation) != args.expected_terminal_sha256:
        raise ValueError("explicit frozen protocol/root-terminal evidence hash mismatch")
    terminal = json_load(args.terminal_observation)
    if not isinstance(terminal.get("exit_code"), int) or isinstance(terminal["exit_code"], bool):
        raise ValueError("root terminal evidence must record an actual terminal exit, not a running session")
    checks = Checks()
    protocol_check(protocol, criteria, checks)
    before = {p.name: file_sha(p) for p in run.iterdir() if p.is_file()}
    source_before = source_digest(repo)
    checks.require(source_before["sha256"] == criteria["source_pins"]["production"] and source_before["files"] == 161, "current_source/9f161")
    checks.require(before.get("driver_source.py") == args.expected_driver_sha256, "producer/exact_source")
    checks.require(before.get("protocol.json") == args.expected_protocol_sha256, "producer/exact_protocol")
    for name, expected in protocol["local_dependency_sha256"].items():
        checks.require(before.get(name) == expected, "producer/dependency/"+name)
    status = json_load(run/"status.json") if (run/"status.json").is_file() else {}
    complete = json_load(run/"complete_receipt.json") if (run/"complete_receipt.json").is_file() else None
    complete_producer = isinstance(complete, dict) and complete.get("status") == "complete_engineering_attempts"
    checks.require(complete_producer, "producer/terminal_complete_receipt_required_for_verified_credit")
    if complete_producer:
        complete_exit_check(terminal["exit_code"], complete, checks)
        checks.require({k: v for k, v in complete.items() if k != "artifact_sha256"} == status, "producer/status_complete_identity")
        checks.require(terminal.get("complete_receipt_sha256") == before["complete_receipt.json"], "terminal/complete_binding")
        manifest_check(run, complete, checks)
        checks.require(complete.get("source_before") == complete.get("source_after"), "producer/source_before_after")
        checks.require(complete.get("input_hashes_after") == protocol["inputs_sha256"], "producer/input_before_after")
        checks.require(complete.get("physical_acceptance") is None and complete.get("scientific_goal_complete") is False, "producer/scope")
    binding = status.get("source_before", {})
    checks.require(binding.get("driver_sha256") == args.expected_driver_sha256 and binding.get("protocol_sha256") == args.expected_protocol_sha256
        and binding.get("local") == protocol["local_dependency_sha256"], "source_before/frozen_driver_protocol_dependencies")
    production = binding.get("production", {})
    checks.require(production.get("repository_source_sha256") == criteria["source_pins"]["production"]
        and production.get("protocol_sha256") == criteria["source_pins"]["original_protocol.json"], "source_before/native_original_protocol")
    expected_backend = {"wavecert.physics.exact_merit", "wavecert.physics.coarse_dst_reference",
        "wavecert.certificates.cached_merit_certificate", "wavecert.optimization.paid_descent",
        "wavecert.physics.helmholtz", "wavecert.physics.coarse_stability", "wavecert.physics.coarse_batch"}
    checks.require(set(production.get("backend_modules", {})) == expected_backend, "source_before/all7_backend_modules")
    for module, digest in production.get("backend_modules", {}).items():
        path = repo/"src"/Path(*module.split(".")).with_suffix(".py")
        checks.require(path.is_file() and file_sha(path) == digest, "source_before/imported_backend/"+module)
    candidate_root = repo/protocol["original_candidate_relative_directory"]
    for name, digest in protocol["candidate_files_sha256"].items():
        path = safe_file(candidate_root, name, checks, "candidate/"+name)
        if path is not None:
            checks.require(file_sha(path) == digest, "candidate/"+name+"/sha256")
    checks.require(production.get("adapter_sha256") == protocol["candidate_files_sha256"]["adapter.py"]
        and production.get("verification_helpers_sha256") == protocol["candidate_files_sha256"]["verification.py"],
        "source_before/candidate_adapter_helpers")
    checks.require(status.get("all96_scheduled_before_project_arrays") is True, "producer/prepopulated96_source_flag")
    planned = expected_schedule(protocol)
    rows = status.get("runs", [])
    scheduled = json_load(run/"schedule.json") if (run/"schedule.json").is_file() else []
    schedule_check(scheduled, planned, checks, "schedule_snapshot")
    schedule_check(rows, planned, checks, "terminal_rows")
    checks.require(all(r.get("status") == "unattempted" and r.get("final_study_eligible") is False
        and r.get("primary_quality") is None for r in scheduled), "schedule_snapshot/initial_noncredit")
    if complete_producer:
        checks.require(complete.get("rows_terminal") == sum(r.get("status") in {"complete", "failed"} for r in rows) == 96, "producer/all96_terminal")
        checks.require(complete.get("numerical_audits_all_pass") is all(r.get("status") == "complete" for r in rows), "producer/numerical_audit_status_scope")
    numeric_authorized = bool(complete_producer and not checks.issues)
    references, records = collect_records(planned, rows, run, repo, protocol, criteria, checks, numeric_authorized)
    after = {p.name: file_sha(p) for p in run.iterdir() if p.is_file()}
    checks.require(before == after, "audit/read_only_run_artifacts_unchanged")
    checks.require(source_digest(repo) == source_before, "audit/read_only_native_source_unchanged")
    for name, expected in protocol["inputs_sha256"].items():
        checks.require(file_sha(repo/name) == expected, "audit/input_unchanged/"+name)
    global_credit = complete_producer and not checks.issues
    summary = summarize(records, global_credit)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), result="CLEAR_SCALAR_AUDIT" if not checks.issues else "INCONSISTENCY_OR_INCOMPLETE_RETAINED",
        checks=checks.count, numeric_comparisons=checks.numeric, issues=checks.issues, verified_study_credit_enabled=global_credit,
        source_sha256=file_sha(__file__), criteria_sha256=file_sha(args.criteria),
        producer_source_sha256=args.expected_driver_sha256, protocol_sha256=args.expected_protocol_sha256,
        terminal_observation_sha256=args.expected_terminal_sha256, complete_receipt_sha256=before.get("complete_receipt.json"),
        numeric_decode_authorized=numeric_authorized,
        source_manifest=source_before, run_files_before=before, run_files_after=after,
        references=[{k: v for k, v in r.items() if k not in {"truth", "initial"}} for r in references],
        records=records, summary=summary, physical_acceptance=None,
        new_PDE_or_optimizer_calls=0, producer_or_engine_imports=False,
        limitations=["Shared NumPy input decoding and generic SciPy reflect-Gaussian initial filter; manual scalar bilinear truth and error reductions are independent.",
            "Manual truth SHA may differ from production rounding; its raw-model binding and scalar quality are checked with fixed inherited tolerances.",
            "Serialized labels/cold status and pinned source enforce distinct owners; opaque native owner tokens and per-stage first/current native receipt headers are not retained.",
            "Only retained scalar samples can rebuild their chain when no middle is omitted. Omitted scalar steps, intermediate vectors and wavefields are not replayed.",
            "Embedded807 objective/PDE checks are retained producer post-return evidence, not independently rerun here. Its unrecorded internal initial-Phi scale is approximated by the serialized saved-initial Phi for this scalar comparison only.",
            "Actual wall is shared-machine evidence. Eight reused inputs and three repetitions are correlated; no isolated timing, IID, pvalue, independent-site, continuum, recovery guarantee or student-authorship claim."])
    output.mkdir()
    with (output/"report.json").open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(result=report["result"], checks=checks.count, issues=len(checks.issues), verified_credit=global_credit)))
    return 0 if not checks.issues else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criteria", type=Path, default=Path(__file__).with_name("criteria.json"))
    parser.add_argument("--execute-after-terminal-confirmation", action="store_true")
    for name in ("run", "repo", "output", "protocol", "terminal-observation"):
        parser.add_argument("--"+name, type=Path)
    for name in ("source", "criteria", "driver", "protocol", "terminal"):
        parser.add_argument("--expected-"+name+"-sha256")
    args = parser.parse_args(argv)
    criteria = json_load(args.criteria)
    if not args.execute_after_terminal_confirmation:
        print(json.dumps(dict(status="metadata_only_no_project_arrays_or_solves", planned_runs=96,
            source_sha256=file_sha(__file__), criteria_sha256=file_sha(args.criteria))))
        return 0
    required = (args.run, args.repo, args.output, args.protocol, args.terminal_observation,
        args.expected_source_sha256, args.expected_criteria_sha256, args.expected_driver_sha256,
        args.expected_protocol_sha256, args.expected_terminal_sha256)
    if not all(required):
        parser.error("explicit terminal GO, paths and all exact reviewed hashes required")
    if file_sha(__file__) != args.expected_source_sha256 or file_sha(args.criteria) != args.expected_criteria_sha256:
        raise ValueError("auditor source/criteria changed after review")
    return audit_execute(args, criteria)


if __name__ == "__main__":
    raise SystemExit(main())
