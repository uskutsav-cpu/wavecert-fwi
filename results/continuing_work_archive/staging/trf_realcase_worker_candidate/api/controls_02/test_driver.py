"""Tiny manufactured optimizer and lifecycle controls, never native/project PDEs."""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import matched_driver as D
from cached_jacobian import CachedJacobianCallbacks, Layout
from component_controls import DenseToyMapPhysics, DenseToyNative, manual_jacobian
from normalized_exact import NormalizedExactOracle
from public_map_builder import PublicReceiverMapBuilder

HERE = Path(__file__).resolve().parent


def admission():
    return D.source_admission(HERE, json.loads((HERE/"protocol.json").read_bytes()))


class Clock:
    def __init__(self):
        self.time = 100.

    def __call__(self):
        return self.time

    def advance(self, seconds):
        self.time += seconds


class ToyNative(DenseToyNative):
    """Known tiny dense issuer; distinct from actual9f and no native function patch."""
    def __init__(self, layout, initial):
        super().__init__(layout, initial)
        self.current = None
        self.stop_gradient = False
        self.accept_hook = None

    def accept(self, receipt):
        if receipt not in self.receipts:
            raise ValueError("foreign toy receipt")
        self.counts["accepts"] = self.counts.get("accepts", 0) + 1
        self.current = receipt
        if self.accept_hook is not None:
            self.accept_hook(self.counts["accepts"])

    def gradient(self, receipt):
        if receipt not in self.receipts:
            raise ValueError("foreign toy receipt")
        self.counts["gradient_calls"] = self.counts.get("gradient_calls", 0) + 1
        if self.stop_gradient:
            raise KeyboardInterrupt("manufactured interrupted gradient after counted dispatch")
        residual = np.concatenate([f.residual_slots.T.reshape(-1) for f in receipt.frames])
        gradient = (manual_jacobian(self, receipt.model).conj().T @ residual).real
        return SimpleNamespace(gradient=gradient)


def fixture():
    initial = np.array([.25, .3, .35], dtype=np.float64)
    layout = Layout(3, (0, 1, 1, 2), ((.23, (2, 0), (0, 1)),))
    owners = []

    def prepare(attempt):
        attempt["preparation_prefix"] = {"manufactured_only": True, "PDE_calls": 0}
        return D.PolicyInput(initial_model=initial, preparation_costs={"manufactured_setup_calls": 1})

    def native_factory(problem):
        native = ToyNative(layout, problem.initial_model)
        owners.append(native)
        return native

    def callbacks_factory(native, initial_model):
        return CachedJacobianCallbacks(native,
            PublicReceiverMapBuilder(lambda config: DenseToyMapPhysics(native)),
            initial_model=initial_model, layout=layout)

    return initial, prepare, native_factory, callbacks_factory, owners


def run(arm, *, timing=None, prepare_override=None, serializer=D.encoded):
    initial, prepare, factory, callbacks, owners = fixture()
    attempt = {}
    kwargs = {} if timing is None else {"clock": timing}
    value = D.execute_one(arm=arm, admit=admission, prepare=prepare if prepare_override is None else prepare_override,
        native_factory=factory, callbacks_factory=callbacks, attempt=attempt, serialize=serializer, **kwargs)
    return value, attempt, initial, owners


def callback_replay(fun, x0, jac, bounds, callback, **options):
    """Declared manufactured SciPy callback protocol replay; not optimizer evidence."""
    residual = fun(x0)
    jac(x0).rmatvec(residual)
    accepted = x0 * .97
    residual = fun(accepted)
    jac(accepted).rmatvec(residual)
    callback(SimpleNamespace(x=accepted, cost=.5*residual@residual))
    rejected = x0 * .93  # Better fit still must not become a returned endpoint.
    fun(rejected)
    return SimpleNamespace(x=rejected, status=1, success=True, message="manufactured result ignored",
        nfev=3, njev=2)


def test_exact_prospective_sources_and48_slot_schedule_no_project_loader():
    protocol = json.loads((HERE/"protocol.json").read_bytes())
    receipt = D.source_admission(HERE, protocol)
    assert receipt["project_execution_available"] is False
    changed = copy.deepcopy(protocol)
    changed["optimizer_deadline_seconds"] = 9.
    with pytest.raises(ValueError, match="protocol"):
        D.source_admission(HERE, changed)
    rows = json.loads((HERE/"FROZEN48_SCHEDULE.json").read_bytes())
    assert len(rows) == 48 and all(r["status"] == "unattempted" for r in rows)


def test_manual_normalized_merit_gradient_and_authentic_unwrapping():
    initial, prepare, factory, _, _ = fixture()
    native = factory(prepare({}))
    wrapper = NormalizedExactOracle(native, initial)
    receipt = wrapper.evaluate(initial)
    phi0 = receipt.raw.objective
    wrapper.accept(receipt)
    assert native.current is receipt.raw and receipt.objective == 1.
    gradient = wrapper.gradient(receipt).gradient
    raw_gradient = native.gradient(receipt.raw).gradient
    np.testing.assert_allclose(gradient, raw_gradient/phi0, rtol=2e-14)
    p = np.array([.3, -.2, .1])
    for h in (1e-5, 5e-6):
        fd = (wrapper.evaluate(initial+h*p).objective-wrapper.evaluate(initial-h*p).objective)/(2*h)
        assert np.isclose(fd, gradient@p, rtol=5e-8, atol=1e-9)
    assert wrapper.phi0 == phi0
    with pytest.raises(AttributeError):
        wrapper.phi0 = 2*phi0


def test_foreign_and_forged_normalized_receipts_rejected_before_native_dispatch():
    initial, prepare, factory, _, _ = fixture()
    first = NormalizedExactOracle(factory(prepare({})), initial)
    second = NormalizedExactOracle(factory(prepare({})), initial)
    with pytest.raises(ValueError, match="already belongs"):
        NormalizedExactOracle(first.native, initial)
    genuine = first.evaluate(initial)
    for invalid in (genuine, dataclasses.replace(genuine, owner=second.token)):
        with pytest.raises(ValueError, match="foreign"):
            second.accept(invalid)
    forged = dataclasses.replace(genuine)
    with pytest.raises(ValueError, match="foreign"):
        first.gradient(forged)
    assert "gradient_attempts" not in first.counts


@pytest.mark.parametrize("arm", D.ARMS)
def test_actual_public_scipy_optimizers_improve_tiny_dense_data_fit(arm):
    envelope, attempt, initial, _ = run(arm)
    observed = D.observe_return(envelope, D.perf_counter())
    assert observed["eligible"] and envelope["payload"]["common_Phi_ratio"] < 1e-5
    assert envelope["payload"]["admission_receipt"]["actual_public_SciPy_TRF_function"]
    assert not np.array_equal(envelope["selected_model"], initial)
    assert attempt["preparation_completed"] and attempt["final_native_cumulative"]
    assert json.loads(envelope["serialized"])["truth_used_for_policy"] is False


def test_rejected_residual_and_solver_result_are_never_selected():
    timing = Clock()
    with patch.object(D, "least_squares", callback_replay):
        envelope, _, initial, _ = run(D.ARMS[1], timing=timing)
    np.testing.assert_array_equal(envelope["selected_model"], initial*.97)
    assert envelope["payload"]["TRF_result_not_used_for_selection"]["result_x_sha256_not_selected"] == D.model_sha(initial*.93)
    assert D.observe_return(envelope, timing())["eligible"]


def test_preparation_consumes_absolute_optimizer_budget_without_reset():
    timing = Clock()
    initial, prepare, _, _, _ = fixture()

    def slow_prepare(attempt):
        timing.advance(7.6)
        return prepare(attempt)

    envelope, attempt, _, owners = run(D.ARMS[0], timing=timing, prepare_override=slow_prepare)
    assert not owners and envelope["selected_model"] is None
    assert envelope["payload"]["status"] == "budget_exhausted"
    assert not D.observe_return(envelope, timing())["eligible"]
    np.testing.assert_array_equal(attempt["initial_model"], initial)


def test_source_admission_is_charged_before_prepare_or_owner_construction():
    timing = Clock()
    _, prepare, factory, callbacks, owners = fixture()
    calls = []

    def slow_admission():
        result = admission()
        timing.advance(7.6)
        return result

    def marked_prepare(attempt):
        calls.append("prepare")
        return prepare(attempt)

    attempt = {}
    envelope = D.execute_one(arm=D.ARMS[0], admit=slow_admission, prepare=marked_prepare,
        native_factory=factory, callbacks_factory=callbacks, attempt=attempt, clock=timing)
    assert attempt["admission_completed"] and not attempt["preparation_attempted"] and not calls and not owners
    assert envelope["payload"]["status"] == "budget_exhausted"
    assert not D.observe_return(envelope, timing())["eligible"]


def test_supplied_pre_entry_time_is_not_reset_by_API_entry_or_solver():
    timing = Clock()
    _, prepare, factory, callbacks, owners = fixture()
    attempt = {}
    envelope = D.execute_one(arm=D.ARMS[1], admit=admission, prepare=prepare,
        native_factory=factory, callbacks_factory=callbacks, attempt=attempt,
        absolute_started=timing()-7.6, clock=timing)
    assert not owners and not attempt["preparation_attempted"]
    assert envelope["payload"]["status"] == "budget_exhausted"
    assert envelope["payload"]["timing_origin"] == "supplied_pre_entry_absolute"
    assert np.isclose(envelope["payload"]["pre_entry_seconds"], 7.6)
    assert envelope["payload"]["complete_OS_process_eligibility_available"] is False


def test_late_callback_cannot_replace_completed_initial_point():
    timing = Clock()

    def late(fun, x0, jac, bounds, callback, **options):
        residual = fun(x0)
        jac(x0).rmatvec(residual)
        model = x0*.97
        residual = fun(model)
        jac(model).rmatvec(residual)
        timing.advance(7.6)
        callback(SimpleNamespace(x=model, cost=.5*residual@residual))

    with patch.object(D, "least_squares", late):
        envelope, _, initial, _ = run(D.ARMS[1], timing=timing)
    assert envelope["payload"]["status"] == "budget_exhausted"
    np.testing.assert_array_equal(envelope["selected_model"], initial)


def test_late_accept_bookkeeping_restores_actual_previous_owned_point():
    timing = Clock()
    initial, prepare, factory, callbacks, owners = fixture()

    def make_native(problem):
        native = factory(problem)
        native.accept_hook = lambda count: timing.advance(.2) if count == 2 else None
        return native

    def late_commit(fun, x0, jac, bounds, callback, **options):
        residual = fun(x0)
        jac(x0).rmatvec(residual)
        model = x0*.97
        residual = fun(model)
        jac(model).rmatvec(residual)
        timing.advance(7.4)
        callback(SimpleNamespace(x=model, cost=.5*residual@residual))

    attempt = {}
    with patch.object(D, "least_squares", late_commit):
        envelope = D.execute_one(arm=D.ARMS[1], admit=admission, prepare=prepare, native_factory=make_native,
            callbacks_factory=callbacks, attempt=attempt, clock=timing)
    np.testing.assert_array_equal(envelope["selected_model"], initial)
    np.testing.assert_array_equal(owners[0].current.model, initial)
    assert attempt["native_accept_completed"] == 3
    assert any(e.get("within_budget") is False for e in envelope["payload"]["TRF_scalar_events"])


def test_serialization_is_inside_complete_return_cap():
    timing = Clock()

    def slow_serializer(payload):
        result = D.encoded(payload)
        timing.advance(8.01)
        return result

    with patch.object(D, "least_squares", callback_replay):
        envelope, _, _, _ = run(D.ARMS[1], timing=timing, serializer=slow_serializer)
    assert envelope["payload"]["policy_valid"]
    assert not D.observe_return(envelope, timing())["eligible"]


def test_serialization_failure_keeps_all_available_prefix_branches():
    attempt = {}
    _, prepare, factory, callbacks, _ = fixture()

    def fail(payload):
        raise OSError("manufactured serialization failure")

    with patch.object(D, "least_squares", callback_replay), pytest.raises(OSError):
        D.execute_one(arm=D.ARMS[1], admit=admission, prepare=prepare, native_factory=factory,
            callbacks_factory=callbacks, attempt=attempt, serialize=fail)
    assert all(k in attempt for k in ("component_prefix", "final_native_cumulative",
        "last_eligible", "return_payload_before_serialization", "preparation_prefix"))
    assert attempt["phase"] == "complete_payload_serialization"


@pytest.mark.parametrize("arm", D.ARMS)
def test_common_boundary_initializer_rejected_without_perturbation_or_owner(arm):
    timing = Clock()

    def boundary(attempt):
        return D.PolicyInput(initial_model=np.array([1/36], dtype=np.float64),
            preparation_costs={"manufactured_setup_calls": 1})

    envelope, _, _, owners = run(arm, timing=timing, prepare_override=boundary)
    assert not owners and not envelope["payload"]["policy_valid"]
    assert envelope["payload"]["status"] == "failed"


def test_actual_first_trf_query_must_equal_registered_direct_initializer():
    def shifted(fun, x0, **kwargs):
        fun(x0+1e-7)

    with patch.object(D, "least_squares", shifted):
        envelope, _, _, owners = run(D.ARMS[1], timing=Clock())
    assert owners[0].counts["evaluations"] == 0
    assert "moved direct initializer" in envelope["payload"]["error"]["message"]


def test_failed_preparation_preserves_partial_observation_prefix_unknown():
    def interrupted(attempt):
        attempt["preparation_prefix"] = {"known_completed_observations": 3, "next_dispatch_inflight": True}
        raise RuntimeError("manufactured partial observation generation")

    envelope, attempt, _, owners = run(D.ARMS[0], timing=Clock(), prepare_override=interrupted)
    assert not owners and not envelope["payload"]["policy_valid"]
    assert envelope["payload"]["unknown_work"] and envelope["payload"]["preparation_work_unknown"]
    assert attempt["preparation_prefix"]["known_completed_observations"] == 3


def test_interrupted_baseline_preserves_normalization_engine_and_native_prefixes():
    initial, prepare, factory, callbacks, _ = fixture()
    attempt = {}

    def interrupted(problem):
        native = factory(problem)
        native.stop_gradient = True
        return native

    with pytest.raises(KeyboardInterrupt, match="manufactured interrupted"):
        D.execute_one(arm=D.ARMS[0], admit=admission, prepare=prepare, native_factory=interrupted,
            callbacks_factory=callbacks, attempt=attempt, clock=Clock())
    assert attempt["unknown_work"] and attempt["normalization_prefix"]["unknown_work"]
    assert attempt["engine_pending_prefix"]["no_completed_return_claim"]
    assert attempt["final_native_cumulative"]["gradient_calls"] == 1
    np.testing.assert_array_equal(attempt["initial_model"], initial)
