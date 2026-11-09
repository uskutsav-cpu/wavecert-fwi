from dataclasses import replace

import numpy as np
import pytest

from wavecert.physics.devito_backend import (
    DevitoAcousticConfig,
    DevitoAcousticFWI,
    _edge_padding_pullback,
)


def test_edge_padding_gradient_is_the_adjoint_of_parameter_extension():
    rng = np.random.default_rng(92)
    shape = (4, 6)
    nbl = 3
    direction = rng.normal(size=shape)
    padded_gradient = rng.normal(size=tuple(n + 2 * nbl for n in shape))
    lhs = float(np.sum(padded_gradient * np.pad(direction, nbl, mode="edge")))
    rhs = float(np.sum(_edge_padding_pullback(padded_gradient, shape, nbl) * direction))
    assert np.isclose(lhs, rhs, rtol=1e-13, atol=1e-13)


@pytest.fixture(scope="module")
def devito_case():
    pytest.importorskip("devito")
    pytest.importorskip("examples.seismic")
    shape = (16, 20)
    x, z = np.indices(shape)
    truth_velocity = 2.0 + 0.2 * np.exp(-((x - 8) ** 2 + (z - 10) ** 2) / 10)
    source = np.array([[70.0, 20.0]])
    receivers = np.column_stack((np.linspace(20, 130, 8), np.full(8, 20.0)))
    config = DevitoAcousticConfig(nbl=8, tn=250.0, max_velocity=2.8, dtype="float64")
    truth = DevitoAcousticFWI(
        truth_velocity, source_coordinates=source, receiver_coordinates=receivers, config=config,
    )
    observed, _ = truth.forward()
    inverse = DevitoAcousticFWI(
        np.full(shape, 2.0), source_coordinates=source, receiver_coordinates=receivers,
        config=replace(config, dt=truth.dt),
    )
    return inverse, observed, truth.time_values


def test_devito_sampling_stays_fixed_during_velocity_updates(devito_case):
    inverse, observed, times = devito_case
    try:
        inverse.update_velocity(np.full(inverse.model.shape, 2.5))
        predicted, _ = inverse.forward()
        assert predicted.shape == observed.shape
        np.testing.assert_array_equal(inverse.time_values, times)
        with pytest.raises(ValueError, match="max_velocity"):
            inverse.update_velocity(np.full(inverse.model.shape, 3.0))
    finally:
        inverse.update_velocity(np.full(inverse.model.shape, 2.0))


def test_devito_rejects_misaligned_or_unlabelled_observations(devito_case):
    inverse, observed, times = devito_case
    with pytest.raises(ValueError, match="observed_time_values"):
        inverse.objective_and_gradient(observed)
    with pytest.raises(ValueError, match="shape"):
        inverse.objective_and_gradient(observed[:-1], observed_time_values=times[:-1])
    with pytest.raises(ValueError, match="sample times"):
        inverse.objective_and_gradient(observed, observed_time_values=times * 1.01)


def test_devito_fixed_sampling_adjoint_matches_full_and_interior_finite_differences(devito_case):
    inverse, observed, times = devito_case
    before = np.array(inverse.model.vp.data, copy=True)
    result = inverse.check_gradient(observed, observed_time_values=times)
    assert result["passed"]
    assert set(result["best_relative_error_by_direction"]) == {"full-domain-including-edges", "interior"}
    assert max(result["best_relative_error_by_direction"].values()) < 1e-5
    np.testing.assert_allclose(inverse.model.vp.data, before, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(inverse.time_values, times)


def test_devito_bounded_optimization_reduces_fixed_objective(devito_case):
    inverse, observed, times = devito_case
    try:
        initial = inverse.objective(observed, observed_time_values=times)
        result = inverse.run_lbfgsb(
            observed, observed_time_values=times, maxiter=2, vmin=1.5, vmax=2.8, ftol=1e-8,
        )
        assert np.isfinite(result.fun) and result.fun < initial
        np.testing.assert_array_equal(inverse.time_values, times)
        with pytest.raises(ValueError, match="CFL|max_velocity"):
            inverse.run_lbfgsb(observed, observed_time_values=times, maxiter=1, vmax=4.0)
    finally:
        inverse.update_velocity(np.full(inverse.model.shape, 2.0))


def test_devito_optimizer_restores_returned_iterate_after_rejected_callback(monkeypatch):
    from types import SimpleNamespace

    from scipy import optimize

    adapter = DevitoAcousticFWI.__new__(DevitoAcousticFWI)
    adapter.model = SimpleNamespace(vp=SimpleNamespace(data=np.full((2, 2), 2.0)), nbl=0, shape=(2, 2))
    adapter.config = DevitoAcousticConfig(max_velocity=3.0)
    adapter.dt = 1.0
    adapter._cfl_velocity_product = 4.0
    adapter._observed_array = lambda observed, times: np.asarray(observed)
    velocities = []
    adapter.update_velocity = lambda vp: velocities.append(np.array(vp, copy=True))
    returned = np.full(4, 1.0 / 2.2**2)

    def minimize(*args, **kwargs):
        # Simulate a final rejected line-search callback at another model.
        adapter.update_velocity(np.full((2, 2), 2.7))
        return optimize.OptimizeResult(x=returned, fun=1.0, success=False)

    monkeypatch.setattr(optimize, "minimize", minimize)
    result = adapter.run_lbfgsb(np.zeros((1, 1)), observed_time_values=np.array([0.0]), vmax=3.0)
    np.testing.assert_array_equal(result.x, returned)
    np.testing.assert_allclose(velocities[-1], np.full((2, 2), 2.2))
    assert not np.array_equal(velocities[0], velocities[-1])
