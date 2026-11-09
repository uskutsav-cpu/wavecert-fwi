"""Tiny independently fabricated scalar trace metadata; no model/data imports."""

import hashlib

import pytest
import trace_checks as trace

CRITERIA = dict(
    engine_deadline_seconds=7.5,
    complete_return_cap_seconds=8.0,
    composed_F_rtol=2e-9,
    composed_F_atol=2e-11,
)


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []

    def require(self, condition, label, detail=None):
        self.count += 1
        if not condition:
            self.failures.append(dict(label=label, detail=detail))

    def near(self, a, b, *, rtol, atol, label):
        self.require(abs(a - b) <= atol + rtol * abs(a), label, dict(a=a, b=b))


def h(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def snapshot(n):
    return dict(oracle=dict(native=dict(queries=n), adapter=dict(queries=n)), proposal={})


def summary(records):
    return dict(
        total_count=len(records),
        chain_sha256=trace.chain(records),
        retained_count=len(records),
        omitted_count=0,
        head_limit=8,
        tail_limit=8,
        finalized_count=len(records),
        pending_record=None,
        pending_record_unsealed=False,
    )


def header(model, F, raw, weighted):
    native = dict(
        objective=raw,
        key=h("key" + model),
        model_sha256=model,
        operator_sha256=h("operator"),
        geometry_sha256=h("geometry"),
        data_sha256=h("data"),
        gradient_cached=False,
    )
    return dict(
        kind="classical_composed_objective_point",
        model_sha256=model,
        objective=F,
        normalized_data_value=raw / 5.0,
        weighted_prior_value=weighted,
        native_data_receipt_header=native,
        native_receipt_fabricated=False,
        certificate_claim=False,
    )


def engine():
    cps = [
        dict(
            kind="initial",
            iteration=-1,
            elapsed_seconds=0.1,
            within_budget=True,
            objective=1.0,
            model_sha256=h("m0"),
            online_costs=snapshot(1),
        )
    ]
    steps = []
    events = []
    for i, F in enumerate((0.85, 0.83)):
        start = 0.2 + i * 1.2
        end = start + 0.9
        accepted = start + 0.6
        model = h("m" + str(i + 1))
        before = 1.0 if i == 0 else 0.85
        local = [
            dict(
                kind="merit",
                candidate_sha256=model,
                elapsed_start=start + 0.1,
                completed=True,
                objective=F,
                elapsed_end=start + 0.2,
                costs=snapshot(2 + i * 3),
            ),
            dict(
                kind="gradient",
                candidate_sha256=model,
                elapsed_start=start + 0.2,
                completed=True,
                gradient_infinity_norm=0.01,
                elapsed_end=start + 0.3,
                costs=snapshot(3 + i * 3),
            ),
        ]
        cb = dict(
            candidate_sha256=model,
            objective=F,
            strict_decrease_endpoint_eligible=True,
            elapsed_seconds=start + 0.4,
        )
        steps.append(
            dict(
                iteration=i,
                method="exact_lbfgsb_search",
                model_sha256=cps[-1]["model_sha256"],
                objective_before=before,
                elapsed_start=start,
                accepted=True,
                trials=[],
                exact_recoveries=[],
                library_evaluations=local,
                elapsed_end=end,
                seconds=end - start,
                online_cost_delta=snapshot(3),
                library_evaluations_summary=summary(local),
                native_callback=cb,
                objective_after=F,
                accepted_elapsed_seconds=accepted,
            )
        )
        cps.append(
            dict(
                kind="accepted",
                iteration=i,
                elapsed_seconds=accepted,
                within_budget=True,
                objective=F,
                model_sha256=model,
                online_costs=snapshot(4 + i * 3),
            )
        )
        events.extend(local)
    return dict(
        status="budget_exhausted",
        costs_complete=True,
        budget_seconds=7.5,
        total_online_seconds=7.6,
        budget_overshoot_seconds=7.6 - 7.5,
        certificate_claim=False,
        certificate_scope=None,
        truth_used_for_policy=False,
        offline_validation_work_performed=False,
        checkpoint_vectors_retained=False,
        actual_complete_return_must_be_stamped_by_caller=True,
        global_clock_anchor_calls=20,
        maximum_steps_safety_cap=4096,
        scipy_maxfun_default=15000,
        scipy_maxfun_passed_explicitly=False,
        dependency_sources=dict(
            engine_sha256=trace.ENGINE_SHA, prior_sha256=trace.PRIOR_SHA, byte_identical_copies=True
        ),
        trace_retention=dict(head=8, tail=8),
        setup_costs=snapshot(0),
        total_online_costs=snapshot(8),
        checkpoints=cps,
        checkpoint_trace=summary(cps),
        steps=steps,
        step_trace=summary(steps),
        all_library_evaluation_trace=dict(
            total_count=len(events), chain_sha256=trace.chain(events)
        ),
        native_callback_count=2,
        native_callback_invocations=2,
        native_callback_point_records=2,
        last_native_callback_attempt=dict(sequence=2, candidate_sha256=h("m2")),
        initial_objective=1.0,
        final_objective=0.83,
        adapter=dict(
            adapter_counts=dict(queries=8),
            weight=100.0,
            initial_native_m_sha256=h("m0"),
            initial_paid_Phi0=5.0,
            initial_accepted=header(h("m0"), 1.0, 5.0, 0.0),
            current_accepted=header(h("m2"), 0.83, 4.1, 0.01),
            latest_native_snapshot=dict(queries=8),
        ),
    )


def seal(e):
    for s in e["steps"]:
        s["library_evaluations_summary"] = summary(s["library_evaluations"])
    e["checkpoint_trace"] = summary(e["checkpoints"])
    e["step_trace"] = summary(e["steps"])
    values = [v for s in e["steps"] for v in s["library_evaluations"]]
    e["all_library_evaluation_trace"] = dict(
        total_count=len(values), chain_sha256=trace.chain(values)
    )


def audit(e):
    checks = Checks()
    result = trace.audit_engine(e, checks, CRITERIA)
    return checks, result


def test_valid_composed_metadata_and_late_bookkeeping():
    e = engine()
    e["budget_overshoot_seconds"] = max(0.0, e["total_online_seconds"] - 7.5)
    checks, result = audit(e)
    assert not checks.failures, checks.failures
    assert result["method_valid"] is True and result["last_eligible"]["model_sha256"] == h("m2")
    assert result["global_library_chain_recomputed"] is True
    assert e["total_online_seconds"] > 7.5


@pytest.mark.parametrize(
    "kind",
    [
        "rawPhi_only_acceptance",
        "wrong_final_F",
        "wrong_current_model",
        "late_checkpoint_credit",
        "unsealed",
        "chain",
        "native_fabrication",
        "counts_boolean",
        "negative_delta",
        "library_legacy_field",
        "global_library_count",
        "callback_count",
        "checkpoint_step_binding",
        "wrong_native_normalization",
        "data_geometry_change",
        "wrong_dependency",
        "boolean_clock",
        "decreasing_checkpoint_counter",
        "wrong_final_native_snapshot",
        "wrong_adapter_snapshot",
        "initial_header_checkpoint_binding",
    ],
)
def test_corruption_rejected(kind):
    e = engine()
    e["budget_overshoot_seconds"] = e["total_online_seconds"] - 7.5
    if kind == "rawPhi_only_acceptance":
        e["checkpoints"][-1]["objective"] = 0.86
    elif kind == "wrong_final_F":
        e["final_objective"] += 0.01
    elif kind == "wrong_current_model":
        e["adapter"]["current_accepted"]["model_sha256"] = h("wrong")
    elif kind == "late_checkpoint_credit":
        e["checkpoints"][-1]["elapsed_seconds"] = 7.50001
    elif kind == "unsealed":
        e["checkpoint_trace"]["pending_record_unsealed"] = True
    elif kind == "chain":
        e["checkpoint_trace"]["chain_sha256"] = "0" * 64
    elif kind == "native_fabrication":
        e["adapter"]["current_accepted"]["native_receipt_fabricated"] = True
    elif kind == "counts_boolean":
        e["total_online_costs"]["oracle"]["native"]["queries"] = True
    elif kind == "negative_delta":
        e["steps"][0]["online_cost_delta"]["oracle"]["native"]["queries"] = -1
    elif kind == "library_legacy_field":
        e["steps"][0]["library_evaluations_trace"] = e["steps"][0].pop(
            "library_evaluations_summary"
        )
    elif kind == "global_library_count":
        e["all_library_evaluation_trace"]["total_count"] = 3
    elif kind == "callback_count":
        e["native_callback_point_records"] = 3
    elif kind == "checkpoint_step_binding":
        e["checkpoints"][1]["model_sha256"] = h("another")
    elif kind == "wrong_native_normalization":
        e["adapter"]["current_accepted"]["normalized_data_value"] += 0.01
    elif kind == "data_geometry_change":
        e["adapter"]["current_accepted"]["native_data_receipt_header"]["data_sha256"] = h(
            "otherdata"
        )
    elif kind == "wrong_dependency":
        e["dependency_sources"]["engine_sha256"] = h("otherengine")
    elif kind == "boolean_clock":
        e["checkpoints"][0]["elapsed_seconds"] = True
    elif kind == "decreasing_checkpoint_counter":
        e["checkpoints"][-1]["online_costs"] = snapshot(0)
    elif kind == "wrong_final_native_snapshot":
        e["adapter"]["latest_native_snapshot"] = {"queries": 7}
    elif kind == "wrong_adapter_snapshot":
        e["adapter"]["adapter_counts"] = {"queries": 7}
    elif kind == "initial_header_checkpoint_binding":
        e["checkpoints"][0]["model_sha256"] = h("badinitial")
    if kind not in ("unsealed", "chain", "global_library_count", "library_legacy_field"):
        seal(e)
    checks, _ = audit(e)
    assert checks.failures, kind


def test_no_raw_Phi_decrease_assumption():
    e = engine()
    e["budget_overshoot_seconds"] = e["total_online_seconds"] - 7.5
    # Final rawPhi4.1 may exceed an earlier rawPhi4.0: retained checkpoints
    # contain F only; no raw-Phi monotonic gate is invented by this checker.
    assert e["adapter"]["current_accepted"]["native_data_receipt_header"]["objective"] > 4.0
    checks, result = audit(e)
    assert not checks.failures and result["method_valid"] is True


def test_retained_omitted_stream_is_explicitly_not_replayed():
    points = [
        dict(
            kind="accepted",
            iteration=i,
            elapsed_seconds=i / 10,
            within_budget=True,
            objective=1 / (i + 1),
            model_sha256=h(i),
            online_costs={},
        )
        for i in range(25)
    ]
    kept = points[:8] + points[-8:]
    s = summary(points)
    s.update(retained_count=16, omitted_count=9)
    checks = Checks()
    result = trace.stream(kept, s, checks, "omitted")
    assert (
        not checks.failures and result["full_chain_recomputed"] is False and result["omitted"] == 9
    )
    s["omitted_count"] = 8
    checks = Checks()
    trace.stream(kept, s, checks, "badcount")
    assert checks.failures


def test_schema_failure_never_gains_method_credit():
    checks, result = audit({"unexpected": "schema"})
    assert checks.failures and result["schema_failed"] is True and result["method_valid"] is False


def test_nonnegative_nested_snapshot_rules():
    assert trace.counter_leq(dict(a=dict(b=1)), dict(a=dict(b=2)))
    assert not trace.counter_leq(dict(a=dict(b=2)), dict(a=dict(b=1)))
    assert not trace.counter_tree(dict(a=True))
    assert not trace.counter_tree(dict(a=-1))


def test_finalized_schema_not_pending_prefix_schema():
    e = engine()
    e["status"] = "unfinalized_engine_prefix"
    e["costs_complete"] = False
    e["budget_overshoot_seconds"] = e["total_online_seconds"] - 7.5
    checks, result = audit(e)
    assert checks.failures and result["method_valid"] is False


@pytest.mark.parametrize("kind", ["zeroPhi0", "retained_position_overrun"])
def test_unexpected_numeric_or_count_schema_is_fail_closed(kind):
    e = engine()
    e["budget_overshoot_seconds"] = e["total_online_seconds"] - 7.5
    if kind == "zeroPhi0":
        e["adapter"]["initial_paid_Phi0"] = 0.0
    else:
        e["checkpoint_trace"]["total_count"] = 0
    checks, result = audit(e)
    assert checks.failures and result["schema_failed"] is True
    assert result["method_valid"] is False
