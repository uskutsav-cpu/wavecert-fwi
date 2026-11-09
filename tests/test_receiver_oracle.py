from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp

from wavecert.experiments.receiver_controls import measure_receiver_controls
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.physics.receiver_oracle import (
    receiver_map_directional_oracle,
    reciprocity_objective_gradient,
)


def _case():
    physics = Helmholtz2D(shape=(7, 8), damping_strength=3.7, damping_width=2)
    rng = np.random.default_rng(614)
    model = rng.uniform(0.19, 0.25, physics.n)
    receivers = (9, 12, 9)  # repeated observations have distinct noise/residuals
    blocks = tuple(SurveyBlock(s, f, np.array([0.2+0.1j, -0.1+0.3j, 0.4-0.2j]), f"{s}@{f}")
                   for f in (3.0, 4.0) for s in (9, 12, 18, 22))
    direction = rng.normal(size=physics.n)
    direction /= np.linalg.norm(direction)
    return physics, model, receivers, blocks, direction


def _ordinary(physics, model, receivers, blocks):
    objective, gradient = 0.0, np.zeros(physics.n)
    for b in blocks:
        value, grad, _, _ = physics.objective_and_gradient(
            model, physics.source(b.source_index), b.observed, receivers, b.frequency_hz,
        )
        objective += value
        gradient += grad
    return objective, gradient


def test_exact_receiver_controls_preserve_complex_adjoint_and_repeated_sampling():
    physics, model, receivers, blocks, direction = _case()
    a = physics.operator(model, 3.0)
    assert (a - a.T).nnz == 0
    assert (a - a.conjugate().T).nnz > 0
    objective, gradient = _ordinary(replace(physics), model, receivers, blocks)
    oracle_physics, reciprocal_physics = replace(physics), replace(physics)
    oracle_objective, derivative = receiver_map_directional_oracle(
        physics=oracle_physics, m=model, blocks=blocks, receiver_indices=receivers, direction=direction,
    )
    reciprocal = reciprocity_objective_gradient(
        physics=reciprocal_physics, m=model, blocks=blocks, receiver_indices=receivers,
    )
    assert oracle_objective == pytest.approx(objective, rel=1e-12)
    assert derivative == pytest.approx(float(np.dot(gradient, direction)), rel=1e-11)
    assert reciprocal.objective == pytest.approx(objective, rel=1e-12)
    assert np.allclose(reciprocal.gradient, gradient, rtol=1e-11, atol=1e-12)
    assert oracle_physics.cost_snapshot()["stability_rhs"] == 2 * 2 * 2
    assert reciprocal.expected_rhs == 2 * (2 + 4 - 2)
    costs = reciprocal_physics.cost_snapshot()
    assert costs["stability_rhs"] + costs["forward_rhs"] == reciprocal.expected_rhs
    assert costs["adjoint_rhs"] == costs["svd_evaluations"] == 0


def test_oracle_matches_independent_finite_difference_of_objective():
    physics, model, receivers, blocks, direction = _case()
    _, derivative = receiver_map_directional_oracle(
        physics=replace(physics), m=model, blocks=blocks, receiver_indices=receivers, direction=direction,
    )

    def objective(m):
        p = replace(physics)
        return sum(p.objective(m, p.source(b.source_index), b.observed, receivers, b.frequency_hz)
                   for b in blocks)

    step = 1e-4
    fd = (objective(model + step*direction) - objective(model - step*direction)) / (2*step)
    assert fd == pytest.approx(derivative, rel=1e-6, abs=1e-10)


def test_reciprocity_keeps_ordinary_path_when_it_costs_less():
    physics, model, _, _, _ = _case()
    receivers = (9, 10, 11, 12, 13, 14)
    blocks = (SurveyBlock(22, 3.0, np.zeros(6, dtype=complex), "one"),)
    result = reciprocity_objective_gradient(
        physics=physics, m=model, blocks=blocks, receiver_indices=receivers,
    )
    assert result.frequency_strategies == {3.0: "ordinary-adjoint"}
    assert result.expected_rhs == 2
    costs = physics.cost_snapshot()
    assert costs["forward_rhs"] == costs["adjoint_rhs"] == 1
    assert costs["stability_rhs"] == 0


def test_reciprocity_rejects_a_nonsymmetric_operator():
    class Nonsymmetric(Helmholtz2D):
        def operator(self, m, frequency_hz):
            return super().operator(m, frequency_hz) + sp.csr_matrix(([1.0], ([0], [1])), shape=(self.n, self.n))

    physics, model, receivers, blocks, _ = _case()
    physics = Nonsymmetric(shape=physics.shape, damping_strength=3.7, damping_width=2)
    with pytest.raises(ValueError, match="transpose symmetry"):
        reciprocity_objective_gradient(physics=physics, m=model, blocks=blocks, receiver_indices=receivers)


@pytest.mark.parametrize("change", ["model_shape", "complex_model", "nan_model", "direction_shape",
                                    "nan_direction", "bad_receiver", "bad_source", "bad_data", "nan_data",
                                    "nan_frequency", "duplicate_block", "empty_blocks", "empty_receivers"])
def test_receiver_oracle_rejects_invalid_inputs(change):
    physics, model, receivers, blocks, direction = _case()
    if change == "model_shape":
        model = model.reshape(physics.shape)
    elif change == "complex_model":
        model = model.astype(complex)
    elif change == "nan_model":
        model[0] = np.nan
    elif change == "direction_shape":
        direction = direction.reshape(physics.shape)
    elif change == "nan_direction":
        direction[0] = np.nan
    elif change == "bad_receiver":
        receivers = (-1, 12, 9)
    elif change == "bad_source":
        blocks = (replace(blocks[0], source_index=True),)
    elif change == "bad_data":
        blocks = (replace(blocks[0], observed=np.zeros((3, 1))),)
    elif change == "nan_data":
        blocks = (replace(blocks[0], observed=np.array([np.nan, 0, 0])),)
    elif change == "nan_frequency":
        blocks = (replace(blocks[0], frequency_hz=np.nan),)
    elif change == "duplicate_block":
        blocks += (blocks[0],)
    elif change == "empty_blocks":
        blocks = ()
    else:
        receivers = ()
    with pytest.raises(ValueError):
        receiver_map_directional_oracle(
            physics=physics, m=model, blocks=blocks, receiver_indices=receivers, direction=direction,
        )


@pytest.mark.parametrize("sources,receivers", [(2, 6), (4, 4), (6, 2)])
def test_control_benchmark_checks_counters_and_agreement(sources, receivers):
    case = measure_receiver_controls(sources=sources, receivers=receivers, shape=(12, 12), repeats=1)
    assert case["all_checks_pass"]
    for method in case["methods"]:
        assert method["expected_rhs"] == method["executed_rhs"]
        assert method["costs"]["factorizations"] == 2
    assert case["finite_difference_best_relative_error"] < 1e-5
