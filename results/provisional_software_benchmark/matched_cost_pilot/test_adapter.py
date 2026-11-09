"""Synthetic engineering checks only; no project arrays or runtime outcomes."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wavecert.optimization.paid_descent import TrajectoryOptions
from wavecert.physics.coarse_dst_reference import LinearSolveFailure
from wavecert.physics.exact_merit import ExactMeritOracle
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock

from . import adapter

GMRES = dict(rtol=1e-10, atol=1e-12, maxiter=100, restart=20)


def problem(route):
    physics = Helmholtz2D((5, 7), spacing=.11, damping_strength=.9)
    model = np.linspace(.08, .15, physics.n)
    sources = (2, 7) if route == "forward_states" else (2, 7, 12, 18, 24, 30)
    receivers = (2, 7, 7, 14, 20) if route == "forward_states" else (2, 7, 7, 2, 7)
    blocks = []
    for frequency in (1.2, 1.6):
        for source in sources:
            observed = physics.solve_state(model*1.03, physics.source(source), frequency)[list(receivers)]
            blocks.append(SurveyBlock(source, frequency, observed, f"{source}@{frequency}"))
    return physics, model, tuple(blocks), receivers


@pytest.mark.parametrize("route", ["forward_states", "receiver_map"])
def test_uncertified_and_certified_controls_use_identical_fields_gradient_and_direction_and_pay_extra_certificate_work(route):
    p, model, blocks, receivers = problem(route)
    proposals, hooks, oracles = [], [], []
    for method in adapter.METHODS[3:]:
        oracle = ExactMeritOracle(p, blocks, receivers)
        receipt = oracle.evaluate(model)
        oracle.accept(receipt)
        hook = adapter.CachedFieldProposal(oracle, p, mode=method, rank=12, sweeps=4, gmres=GMRES)
        before = oracle.cost_snapshot()["physics"]
        proposal = hook.propose(model, receipt)
        assert oracle.cost_snapshot()["physics"] == before  # No hidden fine recovery/verification.
        proposals.append(proposal)
        hooks.append(hook)
        oracles.append(oracle)
    raw, certified = proposals
    assert raw.diagnostics["field_sha256"] == certified.diagnostics["field_sha256"]
    assert raw.diagnostics["gradient_sha256"] == certified.diagnostics["certificate"]["approximate_gradient_sha256"]
    assert np.array_equal(raw.direction, certified.direction)
    assert certified.direction is certified.certificate_direction
    assert raw.certified_descent is None
    assert hooks[0].cost_snapshot()["certificate_local"] == {}
    before = hooks[1].cost_snapshot()
    assert before["certificate_local"]["paid_field_residual_columns"] > 0
    assert before["certificate_dispatches"]["column_norm_dispatches"] > 0
    certificate = certified.diagnostics["certificate"]
    for key, value in certificate["costs"]["local"].items():
        assert before["certificate_local"][key] == value
    # Cumulative stability is setup+query once, not plus a second query receipt.
    assert before["stability"] == hooks[1].stability.cost_snapshot()
    hooks[1].propose(model, oracles[1].evaluate(model))
    after = hooks[1].cost_snapshot()
    assert after["certificate_local"]["paid_field_residual_columns"] == 2*before["certificate_local"]["paid_field_residual_columns"]


@pytest.mark.parametrize("route", ["forward_states", "receiver_map"])
def test_same_M_gmres_has_same_coarse_initializer_true_residuals_and_numerically_exact_gradient(route):
    p, model, blocks, receivers = problem(route)
    oracle = ExactMeritOracle(p, blocks, receivers)
    receipt = oracle.evaluate(model)
    hook = adapter.CachedFieldProposal(oracle, p, mode=adapter.METHODS[2], rank=12, sweeps=4, gmres=GMRES)
    proposal = hook.propose(model, receipt)
    exact = oracle.gradient(receipt)
    expected_direction = -exact.gradient/np.linalg.norm(exact.gradient)
    np.testing.assert_allclose(proposal.direction, expected_direction, rtol=2e-8, atol=2e-9)
    reports = proposal.diagnostics["gmres_reports"]
    assert len(reports) == 2
    assert all(r["converged"] and all(c["true_residual_norm"] <= c["tolerance"] for c in r["columns"]) for r in reports)
    assert all(r["warm_start_sha256"] == proposal.diagnostics["fixed_sweep_receipt"]["field_sha256"][str(r["context_identity"]["frequency_hz"])] for r in reports)
    counts = hook.cost_snapshot()
    assert counts["normfree"]["certificate_svd_evaluations"] == counts["normfree"]["fine_direct_rhs"] == 0
    assert counts["adapter"]["gmres_report_columns"] == sum(len(r["columns"]) for r in reports)


def test_gmres_nonconvergence_fails_closed_and_keeps_partial_reports_and_paid_query_work():
    p, model, blocks, receivers = problem("forward_states")
    oracle = ExactMeritOracle(p, blocks, receivers)
    receipt = oracle.evaluate(model)
    hook = adapter.CachedFieldProposal(oracle, p, mode=adapter.METHODS[2], rank=12, sweeps=4,
        gmres={**GMRES, "maxiter": 0})
    with pytest.raises(LinearSolveFailure):
        hook.propose(model, receipt)
    assert hook.failures and not hook.failures[0]["completed_reports"][0]["converged"]
    assert hook.cost_snapshot()["adapter"]["gmres_failed_columns"] > 0
    assert hook.cost_snapshot()["normfree"]["true_residual_checks"] > 0
    assert hook.cost_snapshot()["normfree"]["coarse_adjoint_rhs"] > 0


def test_all_five_controls_run_only_through_common_paid_engine_and_keep_independent_ownership():
    p, model, blocks, receivers = problem("forward_states")
    for method in adapter.METHODS:
        result = adapter.run_control(method,
            physics_config=dict(shape=p.shape, spacing=p.spacing, damping_width=p.damping_width,
                damping_strength=p.damping_strength), model=model, blocks=blocks, receivers=receivers,
            options=TrajectoryOptions(budget_seconds=10., maximum_steps=1, initial_step=.0001, maximum_step=.0001),
            lower_bound=.01, upper_bound=.3, rank=12, sweeps=4, gmres=GMRES, lbfgsb={}, clock=lambda: 0.)
        assert result["status"] != "failed", result["message"]
        assert result["costs_complete"] and not result["truth_used_for_policy"]
        assert result["total_online_costs"]["oracle"]["physics"]["forward_rhs"] > 0
        assert result["final_objective"] <= result["initial_objective"]


def test_actual_certificate_result_direction_is_not_recomputed_and_cost_ledgers_are_not_double_counted(monkeypatch):
    p, model, blocks, receivers = problem("forward_states")
    oracle = ExactMeritOracle(p, blocks, receivers)
    receipt = oracle.evaluate(model)
    hook = adapter.CachedFieldProposal(oracle, p, mode=adapter.METHODS[4], rank=12, sweeps=4, gmres=GMRES)
    direction = np.linspace(-1., -.2, p.n)
    fake = SimpleNamespace(direction=direction, exact_fallback_required=False, certified_descent=True,
        certified_upper_derivative=-.1, costs=dict(local={"a": 3}, dispatches={"b": 4}, stability_query={"ignored_duplicate": 99}),
        to_dict=lambda: dict(note="synthetic accounting stub"))
    monkeypatch.setattr(adapter, "evaluate_cached_merit", lambda *args: fake)
    result = hook.propose(model, receipt)
    assert result.direction is result.certificate_direction is direction
    assert hook.cost_snapshot()["certificate_local"] == {"a": 3}
    assert hook.cost_snapshot()["certificate_dispatches"] == {"b": 4}
    assert "ignored_duplicate" not in hook.cost_snapshot()["stability"]
    with pytest.raises(ValueError):
        adapter.run_control("wrong", physics_config={}, model=model, blocks=blocks, receivers=receivers,
            options=None, lower_bound=None, upper_bound=None, rank=12, sweeps=4, gmres=GMRES, lbfgsb={}, clock=lambda: 0.)


def test_changed_model_cannot_borrow_own_current_receipt_or_foreign_policy_fields():
    p, model, blocks, receivers = problem("receiver_map")
    oracle, other = ExactMeritOracle(p, blocks, receivers), ExactMeritOracle(p, blocks, receivers)
    receipt = oracle.evaluate(model)
    hook = adapter.CachedFieldProposal(oracle, p, mode=adapter.METHODS[3], rank=12, sweeps=4, gmres=GMRES)
    with pytest.raises(ValueError, match="differs"):
        hook.propose(model*1.01, receipt)
    with pytest.raises(ValueError, match="foreign"):
        hook.propose(model, other.evaluate(model))
    with pytest.raises(ValueError):
        adapter.physical_gradient(oracle, replace(oracle.export_merit_fields(receipt)), {})
