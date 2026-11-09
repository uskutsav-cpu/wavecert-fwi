"""Classical fractional-velocity update penalty; no wave solver or truth input.

All arithmetic is ordinary float64, without a verified roundoff enclosure.
Grid spacing and smoothing length must use the same length unit.
"""

import math
from dataclasses import dataclass

import numpy as np


def _scalar(value, name, *, positive=False):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _array(value, name, *, shape=None, positive=False):
    raw = np.asarray(value)
    if raw.ndim != 2 or min(raw.shape, default=0) < 1:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} has the wrong shape")
    if not np.all(np.isfinite(result)) or (positive and np.any(result <= 0)):
        raise ValueError(f"{name} must contain finite{' positive' if positive else ''} values")
    return result


@dataclass(frozen=True)
class PriorEvaluation:
    value: float
    zeroth_order: float
    roughness: float
    gradient: np.ndarray


@dataclass(frozen=True)
class ObjectiveEvaluation:
    value: float
    normalized_data_value: float
    weighted_prior_value: float
    gradient: np.ndarray
    # None means the lambda=0 arm did not evaluate the prior.
    prior: PriorEvaluation | None


class FractionalVelocityPrior:
    """Anchor a penalty to supplied initial squared slowness.

    R = [sum(q**2) + (ell/h)**2 sum_edges((q_i-q_j)**2)] / (2*N),
    q = m**(-1/2) / initial_m**(-1/2) - 1.

    Each horizontal/vertical nearest-neighbour edge appears exactly once.
    There is no wraparound edge, ghost value, or division by edge count.
    Zero smoothing length leaves only the fractional-update penalty.
    """

    def __init__(self, initial_m, grid_spacing, smoothing_length=None):
        initial = _array(initial_m, "initial_m", positive=True)
        self.shape = initial.shape
        self.grid_spacing = _scalar(grid_spacing, "grid_spacing", positive=True)
        self.smoothing_length = (
            4.0 * self.grid_spacing
            if smoothing_length is None
            else _scalar(smoothing_length, "smoothing_length")
        )
        if not math.isfinite(self.smoothing_length):
            raise ValueError("smoothing_length must be finite")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            self._initial_velocity = np.reciprocal(np.sqrt(initial)).copy()
            self._edge_weight = (self.smoothing_length / self.grid_spacing) ** 2
        if not math.isfinite(self._edge_weight):
            raise ValueError("smoothing length/spacing ratio is too large")
        self._initial_velocity.flags.writeable = False

    def evaluate(self, model_m):
        model = _array(model_m, "model_m", shape=self.shape, positive=True)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            velocity = np.reciprocal(np.sqrt(model))
            q = velocity / self._initial_velocity - 1.0
            vertical = q[1:, :] - q[:-1, :]
            horizontal = q[:, 1:] - q[:, :-1]
            laplacian = np.zeros(self.shape, dtype=np.float64)
            laplacian[1:, :] += vertical
            laplacian[:-1, :] -= vertical
            laplacian[:, 1:] += horizontal
            laplacian[:, :-1] -= horizontal
            count = model.size
            zeroth = float(np.sum(q * q, dtype=np.float64) / (2.0 * count))
            roughness = float(
                self._edge_weight
                * (np.sum(vertical * vertical, dtype=np.float64)
                   + np.sum(horizontal * horizontal, dtype=np.float64))
                / (2.0 * count)
            )
            gradient = (
                (q + self._edge_weight * laplacian)
                * (-0.5 * velocity / model / self._initial_velocity)
                / count
            )
            value = zeroth + roughness
        if not math.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("prior evaluation is not finite in float64")
        return PriorEvaluation(value, zeroth, roughness, gradient)


def normalized_objective(
    data_value, data_gradient, initial_data_value, model_m, prior, weight
):
    """Combine externally supplied data terms with the classical prior.

    Both arms use the same paid positive initial-data normalization. Weight zero
    returns Phi/Phi0 and g/Phi0 and does not evaluate the prior. No PDE, data
    generation, initialization, file access, tuning, or stopping occurs here.
    The caller must pay for all inputs and this function within its run clock.
    """
    if not isinstance(prior, FractionalVelocityPrior):
        raise TypeError("prior must be a FractionalVelocityPrior")
    value = _scalar(data_value, "data_value")
    initial = _scalar(initial_data_value, "initial_data_value", positive=True)
    coefficient = _scalar(weight, "weight")
    model = _array(model_m, "model_m", shape=prior.shape, positive=True)
    gradient = _array(data_gradient, "data_gradient", shape=prior.shape)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        normalized = value / initial
        result_gradient = gradient / initial
        result_prior = prior.evaluate(model) if coefficient > 0 else None
        weighted_prior = coefficient * result_prior.value if result_prior else 0.0
        if result_prior:
            result_gradient = result_gradient + coefficient * result_prior.gradient
        result_value = normalized + weighted_prior
    if not math.isfinite(result_value) or not np.all(np.isfinite(result_gradient)):
        raise FloatingPointError("normalized objective is not finite in float64")
    return ObjectiveEvaluation(
        result_value, normalized, weighted_prior, result_gradient, result_prior
    )
