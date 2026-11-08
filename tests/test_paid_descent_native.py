"""Tiny native physics integration only; no measured runtime or project data."""

import json
from pathlib import Path

import numpy as np
import pytest

import wavecert
from wavecert.optimization import paid_descent as engine
from wavecert.physics import exact_merit as exact
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock


def _problem():
    physics = Helmholtz2D((5, 6), spacing=.05, damping_width=1, damping_strength=4.)
    rng = np.random.default_rng(224)
    truth = rng.uniform(.19, .23, physics.n)
    receivers, blocks = (3, 8, 8, 21), []
    for frequency in (2.3, 3.1):
        for source in (9, 12):
            observed = physics.solve_state(truth, physics.source(source), frequency)[list(receivers)]
            blocks.append(SurveyBlock(source, frequency, observed, f"{source}@{frequency}"))
    return physics, np.full(physics.n, .24), receivers, blocks


def test_native_paid_lu_field_reuse_and_own_oracle_proxy_give_same_accepted_trajectory():
    p, model, receivers, blocks = _problem()
    created, policies = [], []

    def oracle_factory():
        oracle = exact.ExactMeritOracle(p, blocks, receivers,
            reference_package_root=Path(wavecert.__file__).resolve().parent)
        created.append(oracle)
        return oracle

    class OwnPaidFieldProxy:
        """An exact direction adapter for API parity, not a cheap proxy claim."""

        def __init__(self, oracle):
            self.oracle, self.calls = oracle, 0

        def propose(self, m, receipt):
            snapshot = self.oracle.export_merit_fields(receipt)
            self.oracle.validate_merit_fields(snapshot)
            assert np.array_equal(m, snapshot.model)
            self.calls += 1
            return engine.DirectionProposal(-self.oracle.gradient(receipt).gradient,
                diagnostics=dict(purpose="same paid exact direction through public proxy hook"))

        def cost_snapshot(self):
            return dict(proposals=self.calls)

    options = engine.TrajectoryOptions(budget_seconds=100., maximum_steps=2,
        initial_step=.0001, maximum_step=.0001, accepted_step_growth=1.)
    for proxy_factory in (None, OwnPaidFieldProxy):
        # Frozen abstract clock: synthetic correctness only, never runtime evidence.
        result = engine.run_paid_descent(oracle_factory=oracle_factory, proxy_factory=proxy_factory,
            m0=model, lower_bound=.12, upper_bound=.4, options=options, clock=lambda: 0.)
        policies.append(result)
        assert result["status"] == "maximum_steps"
        assert len(result["checkpoints"]) == 3 and all(s["accepted"] for s in result["steps"])
        assert np.all(np.diff([c["objective"] for c in result["checkpoints"]]) < 0)
        counts = result["total_online_costs"]["oracle"]["physics"]
        # Every distinct accepted merit was paid, but next gradients reused its U/LU.
        assert counts["factorizations"] == 6
        assert counts["forward_rhs"] == 12 and counts["adjoint_rhs"] == 8
        assert counts["svd_evaluations"] == counts["dense_inverse_rhs"] == 0
        assert result["total_online_seconds"] == 0.  # Explicitly not measured.
        json.dumps(engine.trajectory_to_dict(result), allow_nan=False)
    assert created[0] is not created[1]
    assert np.allclose(policies[0]["final_model"], policies[1]["final_model"], rtol=0., atol=1e-15)
    assert policies[0]["final_objective"] == pytest.approx(policies[1]["final_objective"], rel=1e-14)


def test_native_lbfgsb_reports_its_own_termination_and_reuses_fun_jac_merits():
    p, model, receivers, blocks = _problem()
    result = engine.run_exact_lbfgsb(oracle_factory=lambda: exact.ExactMeritOracle(p, blocks, receivers,
        reference_package_root=Path(wavecert.__file__).resolve().parent),
        m0=model, lower_bound=.12, upper_bound=.4,
        options=engine.TrajectoryOptions(budget_seconds=100., maximum_steps=2,
            initial_step=.0001, maximum_step=.0001), clock=lambda: 0.)
    assert result["status"] in {"solver_terminated", "solver_success"}
    assert result["solver_result"] is not None
    assert result["solver_result"]["nit"] <= 2
    assert result["final_objective"] <= result["initial_objective"]
    costs = result["total_online_costs"]["oracle"]
    assert costs["oracle"]["objective_cache_hits"] >= result["solver_result"]["njev"]
    assert costs["physics"]["factorizations"] == 2*costs["oracle"]["objective_evaluations"]
    assert costs["physics"]["forward_rhs"] == 2*costs["physics"]["factorizations"]
    assert "native exact L-BFGS-B" in result["acceptance"]
    json.dumps(engine.trajectory_to_dict(result), allow_nan=False)
