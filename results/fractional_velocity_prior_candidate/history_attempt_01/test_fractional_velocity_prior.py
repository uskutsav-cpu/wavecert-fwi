"""Manufactured, project-data-free checks of a classical prior component."""

import math

import numpy as np
import pytest

from fractional_velocity_prior import FractionalVelocityPrior, normalized_objective


def scalar_reference(model, initial, spacing, length):
    """Independent scalar value calculation, without the vector graph operator."""
    rows, cols = initial.shape
    q = {}
    for row in range(rows):
        for col in range(cols):
            v = 1.0 / math.sqrt(float(model[row, col]))
            v0 = 1.0 / math.sqrt(float(initial[row, col]))
            q[row, col] = v / v0 - 1.0
    terms = [x * x for x in q.values()]
    for (row, col), value in q.items():
        for other in ((row + 1, col), (row, col + 1)):
            if other in q:
                terms.append((length / spacing) ** 2 * (value - q[other]) ** 2)
    return math.fsum(terms) / (2 * rows * cols)


@pytest.fixture
def manufactured():
    rows, cols = np.indices((5, 7), dtype=np.float64)
    initial_v = 2.0 + 0.1 * rows + 0.04 * cols
    model_v = initial_v * (1.0 + 0.03 * rows - 0.02 * cols + 0.007 * rows * cols)
    return initial_v**-2, model_v**-2


def test_scalar_reference_and_float64(manufactured):
    initial, model = manufactured
    prior = FractionalVelocityPrior(initial, 0.025)
    result = prior.evaluate(model)
    expected = scalar_reference(model, initial, 0.025, 0.1)
    assert result.value == pytest.approx(expected, rel=2e-15, abs=2e-16)
    assert result.value == result.zeroth_order + result.roughness
    assert result.gradient.dtype == np.float64
    assert result.gradient.shape == model.shape


def test_exact_initial_zero_and_input_copy(manufactured):
    initial, _ = manufactured
    original = initial.copy()
    prior = FractionalVelocityPrior(initial, 0.025)
    initial[:] = 0.5
    result = prior.evaluate(original)
    assert result.value == result.zeroth_order == result.roughness == 0.0
    assert np.array_equal(result.gradient, np.zeros_like(original))


def test_constant_fractional_update_interpretation():
    initial = np.full((4, 6), 0.25)
    model = initial / 1.1**2
    prior = FractionalVelocityPrior(initial, 0.025)
    result = prior.evaluate(model)
    assert result.value == pytest.approx(0.005, abs=2e-17)
    assert result.roughness == 0.0
    combined = normalized_objective(2.0, np.zeros_like(model), 2.0, model, prior, 100.0)
    assert combined.value == pytest.approx(1.5, abs=2e-15)
    assert combined.weighted_prior_value == pytest.approx(0.5, abs=2e-15)


@pytest.mark.parametrize("location,degree", [((0, 0), 2), ((0, 2), 3), ((2, 2), 4)])
def test_actual_graph_boundary_degrees(location, degree):
    initial = np.full((5, 7), 0.25)
    model = initial.copy()
    model[location] /= 1.2**2
    result = FractionalVelocityPrior(initial, 0.025).evaluate(model)
    assert result.value == pytest.approx(0.2**2 * (1 + 16 * degree) / (2 * 35))


@pytest.mark.parametrize("mode", ["corner", "edge", "interior", "mixed"])
def test_directional_derivative_against_scalar_reference(manufactured, mode):
    initial, model = manufactured
    direction = np.zeros_like(model)
    if mode == "mixed":
        for row in range(model.shape[0]):
            for col in range(model.shape[1]):
                direction[row, col] = math.sin(2.3 * row + 0.7 * col + 0.4)
    else:
        direction[{"corner": (0, 0), "edge": (0, 3), "interior": (2, 3)}[mode]] = 1
    direction /= math.sqrt(math.fsum(float(x * x) for x in direction.flat))
    prior = FractionalVelocityPrior(initial, 0.025)
    analytic = math.fsum(
        float(a * b) for a, b in zip(prior.evaluate(model).gradient.flat, direction.flat)
    )
    errors = []
    for step in (2e-5, 1e-5):
        plus = scalar_reference(model + step * direction, initial, 0.025, 0.1)
        minus = scalar_reference(model - step * direction, initial, 0.025, 0.1)
        centered = (plus - minus) / (2 * step)
        assert centered == pytest.approx(analytic, rel=2e-6, abs=2e-8)
        errors.append(abs(centered - analytic))
    assert errors[1] <= 0.35 * errors[0] + 5e-11


def test_zero_smoothing_and_single_row_graph():
    initial = np.full((1, 3), 0.25)
    model = initial / np.array([[1.1, 1.2, 1.3]]) ** 2
    result = FractionalVelocityPrior(initial, 0.1, 0.0).evaluate(model)
    assert result.roughness == 0
    assert result.value == pytest.approx((0.1**2 + 0.2**2 + 0.3**2) / 6)
    nonzero = FractionalVelocityPrior(initial, 0.1, 0.2).evaluate(model)
    assert nonzero.value == pytest.approx(scalar_reference(model, initial, 0.1, 0.2))


def test_lambda_zero_has_same_explicit_normalization(manufactured):
    initial, model = manufactured
    prior = FractionalVelocityPrior(initial, 0.025)
    gradient = np.arange(model.size, dtype=np.float64).reshape(model.shape) - 3
    result = normalized_objective(6.0, gradient, 3.0, model, prior, 0.0)
    assert result.value == result.normalized_data_value == 2.0
    assert result.weighted_prior_value == 0.0
    assert result.prior is None
    assert np.array_equal(result.gradient, gradient / 3.0)
    assert np.array_equal(gradient, np.arange(model.size).reshape(model.shape) - 3)


def test_composed_objective_directional_derivative(manufactured):
    initial, model = manufactured
    prior = FractionalVelocityPrior(initial, 0.025)
    direction = np.ones_like(model) / math.sqrt(model.size)
    data_gradient = model - 0.1
    data_value = 0.5 * math.fsum(float(x * x) for x in data_gradient.flat)
    result = normalized_objective(data_value, data_gradient, 0.7, model, prior, 100)
    analytic = math.fsum(float(a * b) for a, b in zip(result.gradient.flat, direction.flat))
    values = []
    step = 1e-6
    for sign in (1, -1):
        displaced = model + sign * step * direction
        data = 0.5 * math.fsum(float((x - 0.1) ** 2) for x in displaced.flat)
        values.append(data / 0.7 + 100 * scalar_reference(displaced, initial, 0.025, 0.1))
    assert (values[0] - values[1]) / (2 * step) == pytest.approx(analytic, rel=1e-8, abs=1e-7)


@pytest.mark.parametrize("bad", [[], [1.0], [[0.0]], [[-1.0]], [[math.nan]], [[math.inf]], [[1j]], [[True]], [["1"]]])
def test_invalid_initial_models(bad):
    with pytest.raises(ValueError):
        FractionalVelocityPrior(bad, 0.025)


@pytest.mark.parametrize("name,bad", [("grid_spacing", 0), ("grid_spacing", -1), ("grid_spacing", math.inf), ("grid_spacing", True), ("smoothing_length", -1), ("smoothing_length", math.nan)])
def test_invalid_geometry(name, bad):
    kwargs = {"grid_spacing": 0.025, "smoothing_length": 0.1, name: bad}
    with pytest.raises(ValueError):
        FractionalVelocityPrior(np.ones((2, 3)), **kwargs)


@pytest.mark.parametrize("bad", [np.ones((2, 2)), np.zeros((2, 3)), np.full((2, 3), math.nan)])
def test_invalid_evaluation_model(bad):
    prior = FractionalVelocityPrior(np.ones((2, 3)), 0.025)
    with pytest.raises(ValueError):
        prior.evaluate(bad)


@pytest.mark.parametrize("field,bad", [("data_value", -1), ("data_value", math.nan), ("initial_data_value", 0), ("initial_data_value", math.inf), ("weight", -1), ("weight", math.nan), ("data_gradient", np.ones((1, 1))), ("data_gradient", np.full((2, 3), math.inf))])
def test_invalid_composed_inputs(field, bad):
    initial = np.ones((2, 3))
    prior = FractionalVelocityPrior(initial, 0.025)
    kwargs = dict(data_value=1.0, data_gradient=np.zeros_like(initial), initial_data_value=1.0, model_m=initial, prior=prior, weight=100.0)
    kwargs[field] = bad
    with pytest.raises(ValueError):
        normalized_objective(**kwargs)
