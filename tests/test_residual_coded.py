from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp

from wavecert.experiments.residual_coded_controls import measure_residual_coded_controls
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.physics.residual_coded import (
    best_exact_objective_gradient,
    exact_survey_objective_gradient,
    matrix_residual_coded_gradient,
    residual_coded_objective_gradient,
)


class NonreciprocalHelmholtz(Helmholtz2D):
    def operator(self, m, frequency_hz):
        return super().operator(m, frequency_hz) + sp.diags(
            np.full(self.n-1, 0.7+0.45j), 1, shape=(self.n, self.n), format="csr",
        )


def _point_case(physics_class=Helmholtz2D):
    p = physics_class(shape=(7, 8), damping_width=2, damping_strength=3.7)
    rng = np.random.default_rng(4612)
    m = rng.uniform(0.19, 0.25, p.n)
    receivers = (9, 12, 9)
    blocks = tuple(SurveyBlock(source, frequency,
        rng.normal(size=3)+1j*rng.normal(size=3), f"{source}@{frequency}")
        for frequency in (3.0, 4.0) for source in (9, 12, 18, 22))
    return p, m, receivers, blocks


def _ordinary(p, m, receivers, blocks):
    objective, gradient = 0.0, np.zeros(p.n)
    for b in blocks:
        value, grad, _, _ = p.objective_and_gradient(
            m, p.source(b.source_index), b.observed, receivers, b.frequency_hz,
        )
        objective += value
        gradient += grad
    return objective, gradient


@pytest.mark.parametrize("strategy,expected_rhs", [("ordinary", 16), ("symmetry", 8), ("coded", 8), ("best", 8)])
def test_point_routes_preserve_repeated_receiver_objective_and_gradient(strategy, expected_rhs):
    p, m, receivers, blocks = _point_case()
    objective, gradient = _ordinary(replace(p), m, receivers, blocks)
    result = exact_survey_objective_gradient(physics=p, m=m, blocks=blocks,
        receiver_indices=receivers, strategy=strategy)
    assert result.objective == pytest.approx(objective, rel=1e-12)
    assert np.allclose(result.gradient, gradient, rtol=1e-10, atol=1e-11)
    assert result.expected_rhs == expected_rhs
    costs = p.cost_snapshot()
    assert costs["forward_rhs"] + costs["adjoint_rhs"] == expected_rhs
    assert costs["factorizations"] == 2
    assert costs["svd_evaluations"] == costs["stability_rhs"] == 0


def test_coded_point_route_and_best_need_no_transpose_symmetry():
    p, m, receivers, blocks = _point_case(NonreciprocalHelmholtz)
    assert (p.operator(m, 3.0)-p.operator(m, 3.0).T).nnz > 0
    objective, gradient = _ordinary(replace(p), m, receivers, blocks)
    for method in (residual_coded_objective_gradient, best_exact_objective_gradient):
        control = replace(p)
        result = method(physics=control, m=m, blocks=blocks, receiver_indices=receivers)
        assert result.objective == pytest.approx(objective, rel=1e-12)
        assert np.linalg.norm(result.gradient-gradient)/np.linalg.norm(gradient) < 1e-11
        assert set(result.frequency_strategies.values()) == {"coded"}
        assert result.expected_rhs == 8
        assert control.cost_snapshot()["forward_rhs"] == control.cost_snapshot()["adjoint_rhs"] == 4
    with pytest.raises(ValueError, match="requires A.T"):
        exact_survey_objective_gradient(physics=replace(p), m=m, blocks=blocks,
            receiver_indices=receivers, strategy="symmetry")


@pytest.mark.parametrize("sources,receivers,expected", [((18,), (9, 10, 11, 12), "ordinary"),
    ((9, 12, 18, 22, 25, 29), (7, 13), "coded"), ((9, 12), (9, 12), "symmetry")])
def test_best_chooses_the_lowest_declared_rhs_route(sources, receivers, expected):
    p = Helmholtz2D(shape=(7, 8))
    m = np.full(p.n, 0.22)
    blocks = tuple(SurveyBlock(s, 3.0, np.ones(len(receivers), dtype=complex), str(s)) for s in sources)
    result = best_exact_objective_gradient(physics=p, m=m, blocks=blocks, receiver_indices=receivers)
    assert result.frequency_strategies == {3.0: expected}
    c = len(set(sources) & set(receivers))
    assert result.expected_rhs == min(2*len(sources), 2*len(receivers), len(receivers)+len(sources)-c)


def test_general_complex_sources_sampling_and_nonsymmetric_operator_match_adjoint_and_finite_difference():
    p = NonreciprocalHelmholtz(shape=(6, 7), damping_width=2, damping_strength=3.7)
    rng = np.random.default_rng(12517)
    n, s, r, frequency = p.n, 7, 3, 3.0
    m = rng.uniform(0.18, 0.26, n)
    q = (rng.normal(size=(n, s))+1j*rng.normal(size=(n, s)))*0.1
    sampling = (rng.normal(size=(r, n))+1j*rng.normal(size=(r, n)))/np.sqrt(n)
    truth = replace(p)._solve(m+0.008*np.sin(np.arange(n)), q, frequency, category="forward_rhs")
    observed = sampling @ truth
    noise = rng.normal(size=observed.shape)+1j*rng.normal(size=observed.shape)
    observed += 0.03*np.linalg.norm(observed)*noise/np.linalg.norm(noise)
    ordinary = replace(p)
    state = ordinary._solve(m, q, frequency, category="forward_rhs")
    residual = sampling @ state-observed
    adjoint = ordinary._solve(m, sampling.conjugate().T @ residual, frequency,
                              category="adjoint_rhs", adjoint=True)
    omega2 = (2*np.pi*frequency)**2
    expected = omega2*np.real(np.sum(np.conjugate(adjoint)*state, axis=1))
    objective, gradient = matrix_residual_coded_gradient(physics=p, m=m, source_matrix=q,
        receiver_matrix=sampling, observed_matrix=observed, frequency_hz=frequency)
    assert objective == pytest.approx(0.5*np.linalg.norm(residual)**2, rel=1e-11)
    assert np.linalg.norm(gradient-expected)/np.linalg.norm(expected) < 1e-11
    assert p.cost_snapshot()["forward_rhs"] == p.cost_snapshot()["adjoint_rhs"] == r
    assert ordinary.cost_snapshot()["forward_rhs"] == ordinary.cost_snapshot()["adjoint_rhs"] == s

    direction = rng.normal(size=n)
    direction /= np.linalg.norm(direction)
    fd_solver = replace(p)

    def phi(model):
        value = sampling @ fd_solver._solve(model, q, frequency, category="forward_rhs")-observed
        return 0.5*np.linalg.norm(value)**2

    exact_dd = float(np.dot(expected, direction))
    errors = [abs((phi(m+h*direction)-phi(m-h*direction))/(2*h)-exact_dd)/abs(exact_dd)
              for h in (1e-3, 3e-4, 1e-4)]
    assert min(errors) < 2e-6

    # This complex test would not accidentally pass if C were conjugated twice.
    check = replace(p)
    c = check._solve(m, sampling.conjugate().T, frequency, category="adjoint_rhs", adjoint=True).conjugate().T
    coded = check._solve(m, q @ (c @ q-observed).conjugate().T, frequency, category="forward_rhs")
    wrong = omega2*np.real(np.sum(c.conjugate().T*coded, axis=1))
    assert np.linalg.norm(wrong-expected)/np.linalg.norm(expected) > 0.1


def test_general_matrix_helper_preserves_repeated_sampling_with_distinct_observations():
    p, m, receivers, blocks = _point_case(NonreciprocalHelmholtz)
    group = blocks[:4]
    q = np.stack([p.source(b.source_index) for b in group], axis=1)
    sampling = np.eye(p.n, dtype=complex)[list(receivers)]
    observed = np.stack([b.observed for b in group], axis=1)
    phi, gradient = matrix_residual_coded_gradient(physics=p, m=m, source_matrix=q,
        receiver_matrix=sampling, observed_matrix=observed, frequency_hz=3.0)
    expected_phi, expected_gradient = _ordinary(replace(p), m, receivers, group)
    assert phi == pytest.approx(expected_phi, rel=1e-12)
    assert np.allclose(gradient, expected_gradient, rtol=1e-10, atol=1e-11)
    assert p.cost_snapshot()["forward_rhs"] == p.cost_snapshot()["adjoint_rhs"] == len(receivers)


@pytest.mark.parametrize("change", ["source_shape", "receiver_shape", "observed_shape", "nonfinite", "frequency"])
def test_general_matrix_helper_rejects_invalid_inputs(change):
    p = Helmholtz2D(shape=(4, 5))
    m, q, sampling, data, frequency = np.full(p.n, 0.22), np.ones((p.n, 2)), np.ones((3, p.n)), np.ones((3, 2)), 3.0
    if change == "source_shape":
        q = q.T
    elif change == "receiver_shape":
        sampling = sampling.T
    elif change == "observed_shape":
        data = data.T
    elif change == "nonfinite":
        q[0, 0] = np.nan
    else:
        frequency = 0.0
    with pytest.raises(ValueError):
        matrix_residual_coded_gradient(physics=p, m=m, source_matrix=q,
            receiver_matrix=sampling, observed_matrix=data, frequency_hz=frequency)


@pytest.mark.parametrize("sources,receivers", [(2, 6), (4, 4), (6, 2)])
def test_new_control_benchmark_matches_counts_and_disables_timings_by_default(sources, receivers):
    case = measure_residual_coded_controls(sources=sources, receivers=receivers, shape=(12, 12))
    assert case["all_checks_pass"]
    assert case["best_rhs_no_greater_than_initial_map_rhs"]
    assert len(case["methods"]) == 6
    for method in case["methods"]:
        assert method["executed_rhs"] == method["expected_rhs"]
        assert method["costs"]["factorizations"] == 2
        assert method["median_elapsed_seconds"] is None
        assert method["elapsed_seconds"] == []
    assert case["finite_difference_best_relative_error"] < 1e-5
