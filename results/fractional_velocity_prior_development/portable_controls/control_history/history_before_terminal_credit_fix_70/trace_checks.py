"""Independent scalar inspection of retained composed-F engine metadata.

No producer, controller, model decoder, solver, or project data import occurs.
A SHA commitment to omitted records is not replayable from retained samples.
"""

from __future__ import annotations

import hashlib
import json
import math

SEED = b"bounded-exact-record-stream-v1\0"
ENGINE_SHA = "d5fb6326bcd252efc3a7672dba993311fb0ba724b90039e7fda317bcf8bb770c"
PRIOR_SHA = "4aa248fd3862533390ad918f4ad3f70d02fcdebcdc8d784494d375c935536163"
VALID_STATUSES = {"solver_success", "solver_terminated", "budget_exhausted", "maximum_steps"}
LIMITS = "Only retained head8/tail8 scalar records are inspected. Omitted middle vectors/events and opaque commitment preimages are not replayed; no PDE or receipt-issuer execution occurs. Main auditor must bind the returned final vector hash to the selected checkpoint."


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def count(value):
    return type(value) is int and value >= 0


def digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def counter_tree(value):
    return isinstance(value, dict) and all(
        isinstance(key, str) and (counter_tree(item) if isinstance(item, dict) else count(item))
        for key, item in value.items()
    )


def counter_leq(before, after):
    if not counter_tree(before) or not counter_tree(after):
        return False
    for key, item in before.items():
        other = after.get(key)
        if isinstance(item, dict):
            if not isinstance(other, dict) or not counter_leq(item, other):
                return False
        elif not count(other) or item > other:
            return False
    return True


def record_bytes(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def chain(records):
    state = hashlib.sha256(SEED).digest()
    for index, record in enumerate(records, 1):
        raw = record_bytes(record)
        state = hashlib.sha256(
            state + index.to_bytes(8, "big") + len(raw).to_bytes(8, "big") + raw
        ).digest()
    return state.hex()


def stream(records, summary, checks, label):
    if not isinstance(records, list) or not isinstance(summary, dict):
        raise ValueError("stream records/summary types")
    total, retained, omitted, finalized = (
        summary[k] for k in ("total_count", "retained_count", "omitted_count", "finalized_count")
    )
    if not all(count(v) for v in (total, retained, omitted, finalized)):
        raise ValueError("stream nonnegative integer counts")
    checks.require(summary["head_limit"] == summary["tail_limit"] == 8, label + "/fixed_head_tail")
    checks.require(
        finalized == total
        and summary["pending_record"] is None
        and summary["pending_record_unsealed"] is False,
        label + "/fully_sealed",
    )
    checks.require(
        retained == len(records) == min(total, 16) and omitted == total - retained,
        label + "/full_retained_denominator",
    )
    checks.require(digest(summary["chain_sha256"]), label + "/commitment_shape")
    if omitted == 0:
        checks.require(chain(records) == summary["chain_sha256"], label + "/full_chain_recomputed")
    positions = list(range(min(total, 8))) + list(range(max(8, total - 8), total))
    checks.require(len(positions) == len(records), label + "/head_tail_positions")
    return dict(
        total=total,
        omitted=omitted,
        positions=positions,
        full_chain_recomputed=omitted == 0,
        commitment_sha256=summary["chain_sha256"],
    )


def header(value, checks, criteria, label):
    if not isinstance(value, dict):
        raise ValueError("composed header must be object")
    checks.require(value["kind"] == "classical_composed_objective_point", label + "/kind")
    checks.require(
        value["native_receipt_fabricated"] is False and value["certificate_claim"] is False,
        label + "/no_fabrication_or_certificate",
    )
    native = value["native_data_receipt_header"]
    if not isinstance(native, dict):
        raise ValueError("native header must be object")
    for key in ("model_sha256", "operator_sha256", "geometry_sha256", "data_sha256", "key"):
        checks.require(digest(native[key]), label + "/native_" + key)
    checks.require(
        value["model_sha256"] == native["model_sha256"], label + "/native_model_identity"
    )
    checks.require(type(native["gradient_cached"]) is bool, label + "/native_cache_flag")
    checks.require(
        finite(native["objective"]) and native["objective"] >= 0, label + "/raw_Phi_finite"
    )
    components = [value[k] for k in ("objective", "normalized_data_value", "weighted_prior_value")]
    checks.require(
        all(finite(x) and x >= 0 for x in components), label + "/finite_nonnegative_F_components"
    )
    checks.near(
        value["normalized_data_value"] + value["weighted_prior_value"],
        value["objective"],
        rtol=criteria.get("composed_F_rtol", 2e-9),
        atol=criteria.get("composed_F_atol", 2e-11),
        label=label + "/F_component_sum",
    )
    return native


def audit_engine(engine, checks, criteria, label="engine"):
    """Fail closed on unexpected scalar schema; never import the producer."""
    try:
        return _audit_engine(engine, checks, criteria, label)
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        AttributeError,
        ZeroDivisionError,
        IndexError,
    ) as exc:
        checks.require(False, label + "/unexpected_schema", f"{type(exc).__name__}: {exc}")
        return dict(
            method_valid=False,
            last_eligible=None,
            checkpoint_omitted=None,
            step_omitted=None,
            omission_limits=LIMITS,
            schema_failed=True,
        )


def _audit_engine(engine, checks, criteria, label):
    if not isinstance(engine, dict):
        raise ValueError("engine must be object")
    deadline = criteria["engine_deadline_seconds"]
    checks.require(
        deadline == 7.5 and criteria["complete_return_cap_seconds"] == 8.0,
        label + "/fixed_deadlines",
    )
    checks.require(engine["budget_seconds"] == deadline, label + "/engine_budget")
    checks.require(type(engine["costs_complete"]) is bool, label + "/costs_complete_boolean")
    online = engine["total_online_seconds"]
    checks.require(finite(online) and online >= 0, label + "/online_time")
    checks.require(
        engine["budget_overshoot_seconds"] == max(0.0, online - deadline),
        label + "/overshoot_exact",
    )
    checks.require(engine["status"] in VALID_STATUSES | {"failed"}, label + "/known_status")
    checks.require(
        engine["certificate_claim"] is False and engine["certificate_scope"] is None,
        label + "/composed_not_certificate",
    )
    checks.require(
        engine["truth_used_for_policy"] is False
        and engine["offline_validation_work_performed"] is False,
        label + "/policy_truth_boundary",
    )
    checks.require(
        engine["checkpoint_vectors_retained"] is False, label + "/no_full_checkpoint_vectors"
    )
    checks.require(
        engine["actual_complete_return_must_be_stamped_by_caller"] is True,
        label + "/outer_return_required",
    )
    checks.require(
        count(engine["global_clock_anchor_calls"]) and engine["global_clock_anchor_calls"] > 0,
        label + "/global_anchor",
    )
    checks.require(
        engine["maximum_steps_safety_cap"] == 4096
        and engine["scipy_maxfun_default"] == 15000
        and engine["scipy_maxfun_passed_explicitly"] is False,
        label + "/library_limits",
    )
    checks.require(
        engine["dependency_sources"]
        == dict(engine_sha256=ENGINE_SHA, prior_sha256=PRIOR_SHA, byte_identical_copies=True),
        label + "/frozen_components",
    )
    checks.require(
        engine["trace_retention"]["head"] == engine["trace_retention"]["tail"] == 8,
        label + "/trace_retention",
    )
    setup, total = engine["setup_costs"], engine["total_online_costs"]
    checks.require(
        counter_tree(setup) and counter_tree(total), label + "/nonnegative_snapshot_trees"
    )
    checks.require(counter_leq(setup, total), label + "/setup_to_final_snapshot_monotonicity")
    checkpoint_records, step_records = engine["checkpoints"], engine["steps"]
    cp = stream(checkpoint_records, engine["checkpoint_trace"], checks, label + "/checkpoints")
    st = stream(step_records, engine["step_trace"], checks, label + "/steps")
    previous_time, previous_F, previous_iteration = -1.0, None, -2
    selected = None
    previous_costs = setup
    for index, point in enumerate(checkpoint_records):
        prefix = label + f"/checkpoint_retained{index}"
        position = cp["positions"][index]
        elapsed, F, iteration = point["elapsed_seconds"], point["objective"], point["iteration"]
        checks.require(
            finite(elapsed) and 0 <= elapsed <= online and elapsed >= previous_time,
            prefix + "/chronology",
        )
        checks.require(finite(F) and F >= 0, prefix + "/finite_F")
        checks.require(
            type(iteration) is int and iteration > previous_iteration,
            prefix + "/callback_iteration_order",
        )
        checks.require(digest(point["model_sha256"]), prefix + "/model_digest")
        checks.require(
            type(point["within_budget"]) is bool
            and point["within_budget"] is (elapsed <= deadline),
            prefix + "/strict_timing_gate",
        )
        checks.require(
            counter_tree(point["online_costs"]) and counter_leq(point["online_costs"], total),
            prefix + "/counter_snapshot",
        )
        checks.require(
            counter_leq(previous_costs, point["online_costs"]),
            prefix + "/chronological_counter_monotonicity",
        )
        previous_costs = point["online_costs"]
        if position == 0:
            checks.require(point["kind"] == "initial" and iteration == -1, prefix + "/initial_kind")
            checks.require(F == engine["initial_objective"], prefix + "/initial_F")
        else:
            checks.require(
                point["kind"] == "accepted" and iteration >= 0, prefix + "/accepted_kind"
            )
        if previous_F is not None:
            checks.require(F < previous_F, prefix + "/strict_composed_F_decrease")
        if point["within_budget"] and elapsed <= deadline:
            selected = point
        previous_time, previous_F, previous_iteration = elapsed, F, iteration
    cp_by_iteration = {
        point["iteration"]: point for point in checkpoint_records if point["kind"] == "accepted"
    }
    previous_end, retained_callbacks, retained_library, accepted_retained = -1.0, 0, [], 0
    all_library_known = st["omitted"] == 0
    library_omitted = 0
    for index, step in enumerate(step_records):
        prefix = label + f"/step_retained{index}"
        checks.require(step["iteration"] == st["positions"][index], prefix + "/full_step_position")
        checks.require(step["method"] == "exact_lbfgsb_search", prefix + "/native_method")
        start, end = step["elapsed_start"], step["elapsed_end"]
        checks.require(
            finite(start) and finite(end) and 0 <= start <= end <= online and start >= previous_end,
            prefix + "/chronology",
        )
        checks.require(step["seconds"] == end - start, prefix + "/duration_exact")
        checks.require(
            finite(step["objective_before"]) and step["objective_before"] >= 0,
            prefix + "/finite_F_before",
        )
        checks.require(digest(step["model_sha256"]), prefix + "/start_model_digest")
        checks.require(type(step["accepted"]) is bool, prefix + "/accepted_boolean")
        checks.require(counter_tree(step["online_cost_delta"]), prefix + "/nonnegative_delta_only")
        lib = stream(
            step["library_evaluations"],
            step["library_evaluations_summary"],
            checks,
            prefix + "/library",
        )
        library_omitted += lib["omitted"]
        all_library_known = all_library_known and lib["omitted"] == 0
        last_event_end = start
        for event_index, event in enumerate(step["library_evaluations"]):
            event_prefix = prefix + f"/event_retained{event_index}"
            checks.require(event["kind"] in ("merit", "gradient"), event_prefix + "/kind")
            checks.require(digest(event["candidate_sha256"]), event_prefix + "/candidate_digest")
            event_start = event["elapsed_start"]
            checks.require(
                finite(event_start) and last_event_end <= event_start <= end,
                event_prefix + "/start",
            )
            checks.require(type(event["completed"]) is bool, event_prefix + "/completed_boolean")
            if event["completed"]:
                event_end = event["elapsed_end"]
                checks.require(
                    finite(event_end) and event_start <= event_end <= end,
                    event_prefix + "/completion_time",
                )
                checks.require(
                    counter_tree(event["costs"]) and counter_leq(event["costs"], total),
                    event_prefix + "/counter_snapshot",
                )
                value_key = "objective" if event["kind"] == "merit" else "gradient_infinity_norm"
                checks.require(
                    finite(event[value_key]) and event[value_key] >= 0,
                    event_prefix + "/finite_scalar",
                )
                last_event_end = event_end
            else:
                last_event_end = event_start
            retained_library.append(event)
        callback = step.get("native_callback")
        if callback is not None:
            retained_callbacks += 1
            elapsed, F = callback["elapsed_seconds"], callback["objective"]
            checks.require(
                finite(elapsed) and last_event_end <= elapsed <= end, prefix + "/callback_time"
            )
            checks.require(
                finite(F) and F >= 0 and F <= step["objective_before"],
                prefix + "/callback_nonincreasing_F",
            )
            checks.require(digest(callback["candidate_sha256"]), prefix + "/callback_model_digest")
            strict = F < step["objective_before"] and elapsed <= deadline
            checks.require(
                type(callback["strict_decrease_endpoint_eligible"]) is bool
                and callback["strict_decrease_endpoint_eligible"] is strict,
                prefix + "/callback_strict_endpoint_flag",
            )
        if step["accepted"]:
            accepted_retained += 1
            checks.require(
                callback is not None and callback["strict_decrease_endpoint_eligible"] is True,
                prefix + "/accepted_paid_callback",
            )
            checks.require(
                step["objective_after"] == callback["objective"] < step["objective_before"],
                prefix + "/accepted_F",
            )
            checks.require(
                callback["elapsed_seconds"] <= step["accepted_elapsed_seconds"] <= deadline,
                prefix + "/accepted_deadline",
            )
        if step["iteration"] in cp_by_iteration:
            point = cp_by_iteration[step["iteration"]]
            checks.require(
                step["accepted"] is True, prefix + "/retained_checkpoint_requires_accepted_step"
            )
            if step["accepted"]:
                checks.require(
                    point["model_sha256"] == callback["candidate_sha256"]
                    and point["objective"] == step["objective_after"]
                    and point["elapsed_seconds"] == step["accepted_elapsed_seconds"],
                    prefix + "/retained_checkpoint_step_binding",
                )
        previous_end = end
    all_lib = engine["all_library_evaluation_trace"]
    if (
        not isinstance(all_lib, dict)
        or not count(all_lib["total_count"])
        or not digest(all_lib["chain_sha256"])
    ):
        raise ValueError("global library commitment schema")
    checks.require(
        all_lib["total_count"] >= len(retained_library) + library_omitted,
        label + "/global_library_denominator",
    )
    if all_library_known:
        checks.require(
            all_lib["total_count"] == len(retained_library)
            and all_lib["chain_sha256"] == chain(retained_library),
            label + "/global_library_chain_recomputed",
        )
    callbacks, point_records = (
        engine["native_callback_invocations"],
        engine["native_callback_point_records"],
    )
    checks.require(
        count(callbacks) and count(point_records) and point_records <= callbacks <= 4096,
        label + "/callback_counts",
    )
    checks.require(engine["native_callback_count"] == callbacks, label + "/callback_alias")
    checks.require(callbacks - point_records in (0, 1), label + "/possible_entry_guard_callback")
    checks.require(
        retained_callbacks <= point_records <= st["total"], label + "/retained_callback_denominator"
    )
    if st["omitted"] == 0:
        checks.require(retained_callbacks == point_records, label + "/all_callback_records_counted")
        checks.require(
            cp["total"] == accepted_retained + (1 if checkpoint_records else 0),
            label + "/all_accepts_match_checkpoints",
        )
    last_attempt = engine["last_native_callback_attempt"]
    if callbacks:
        checks.require(
            isinstance(last_attempt, dict)
            and last_attempt["sequence"] == callbacks
            and digest(last_attempt["candidate_sha256"]),
            label + "/last_callback_attempt",
        )
    else:
        checks.require(last_attempt is None, label + "/no_callback_attempt")
    adapter = engine["adapter"]
    current = None
    if adapter is not None:
        checks.require(counter_tree(adapter["adapter_counts"]), label + "/adapter_counter_tree")
        checks.require(
            adapter["weight"] in (0.0, 100.0) and type(adapter["weight"]) is not bool,
            label + "/fixed_weight",
        )
        checks.require(digest(adapter["initial_native_m_sha256"]), label + "/initial_model_digest")
        phi0 = adapter["initial_paid_Phi0"]
        if adapter["initial_accepted"] is not None:
            initial_native = header(
                adapter["initial_accepted"], checks, criteria, label + "/initial_header"
            )
            checks.require(
                finite(phi0) and phi0 > 0 and initial_native["objective"] == phi0,
                label + "/paid_positive_Phi0",
            )
            checks.require(
                adapter["initial_accepted"]["model_sha256"] == adapter["initial_native_m_sha256"],
                label + "/direct_initial_header",
            )
            checks.require(
                bool(checkpoint_records)
                and checkpoint_records[0]["kind"] == "initial"
                and checkpoint_records[0]["model_sha256"] == adapter["initial_native_m_sha256"],
                label + "/initial_checkpoint_header_binding",
            )
            checks.require(
                adapter["initial_accepted"]["objective"] == engine["initial_objective"] == 1.0,
                label + "/normalized_initial_F",
            )
        if adapter["current_accepted"] is not None:
            current = adapter["current_accepted"]
            native = header(current, checks, criteria, label + "/current_header")
            checks.require(
                current["objective"] == engine["final_objective"], label + "/final_F_header"
            )
            checks.require(
                adapter["initial_accepted"] is not None, label + "/current_requires_initial"
            )
            initial_native = adapter["initial_accepted"]["native_data_receipt_header"]
            checks.require(
                all(native[k] == initial_native[k] for k in ("geometry_sha256", "data_sha256")),
                label + "/same_data_geometry",
            )
            checks.near(
                native["objective"] / phi0,
                current["normalized_data_value"],
                rtol=criteria.get("composed_F_rtol", 2e-9),
                atol=criteria.get("composed_F_atol", 2e-11),
                label=label + "/current_normalization",
            )
        if adapter["latest_native_snapshot"] is not None:
            checks.require(
                counter_tree(adapter["latest_native_snapshot"]), label + "/latest_native_snapshot"
            )
            checks.require(
                adapter["latest_native_snapshot"] == total["oracle"]["native"],
                label + "/latest_native_snapshot_exact_end",
            )
        checks.require(
            adapter["adapter_counts"] == total["oracle"]["adapter"],
            label + "/adapter_snapshot_exact_end",
        )
    method_valid = bool(
        engine["status"] in VALID_STATUSES
        and engine["costs_complete"] is True
        and selected is not None
        and current is not None
        and selected["model_sha256"] == current["model_sha256"]
        and selected["objective"] == engine["final_objective"] == current["objective"]
    )
    if selected is not None and current is not None:
        checks.require(
            selected["model_sha256"] == current["model_sha256"]
            and selected["objective"] == current["objective"],
            label + "/last_chronological_eligible_header",
        )
    return dict(
        method_valid=method_valid,
        last_eligible=selected,
        checkpoint_omitted=cp["omitted"],
        step_omitted=st["omitted"],
        retained_library_events=len(retained_library),
        retained_steps_library_omitted=library_omitted,
        global_library_chain_recomputed=all_library_known,
        omission_limits=LIMITS,
        schema_failed=False,
    )
