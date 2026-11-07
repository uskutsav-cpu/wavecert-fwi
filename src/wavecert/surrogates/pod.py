"""Classical fixed real-basis POD Galerkin surrogate of the reference operator.

Its state/JVP/adjoint are consistent derivatives of the reduced model. They are
not certificates of fine-PDE accuracy. Online mass projection, reduced solves,
and fine-grid operations have different costs and must not be conflated.
"""

from __future__ import annotations

import io
import json
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.linalg as la

from wavecert.physics.helmholtz import Helmholtz2D


class PODGalerkinSurrogate:
    """Galerkin wavefield surrogate with a copied immutable real basis/operator.

    The supplied physics configuration is snapshotted at construction. Later
    mutation of that external object does not change this reduced model.
    Model and frequency changes are handled by bounded content-keyed caches.
    """

    _COST_KEYS = (
        "reduced_factorizations", "reduced_forward_rhs", "reduced_tangent_rhs",
        "reduced_adjoint_rhs", "mass_assemblies", "factor_cache_hits", "mass_cache_hits",
        "source_projections", "tangent_rhs_projections", "adjoint_rhs_projections",
        "fine_state_lifts", "fine_tangent_lifts", "fine_adjoint_lifts",
    )

    def __init__(self, physics: Helmholtz2D, basis: np.ndarray, *,
                 factorization_cache_size: int = 8, mass_cache_size: int = 8,
                 orthogonality_tolerance: float = 1e-10):
        if len(physics.shape) != 2 or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0 for value in physics.shape):
            raise ValueError("reference grid must have two positive integer dimensions")
        if isinstance(physics.damping_width, (bool, np.bool_)) or not isinstance(physics.damping_width, (int, np.integer)) or physics.damping_width <= 0:
            raise ValueError("reference damping width must be a positive integer")
        values = np.asarray(basis)
        if np.iscomplexobj(values) or values.ndim != 2:
            raise ValueError("POD basis must be a real two-dimensional array")
        values = np.array(values, dtype=np.float64, order="C", copy=True)
        if values.shape[0] != physics.n or not 1 <= values.shape[1] <= physics.n or not np.all(np.isfinite(values)):
            raise ValueError("POD basis dimensions and finite values must match the reference grid")
        if not np.isfinite(orthogonality_tolerance) or orthogonality_tolerance <= 0:
            raise ValueError("orthogonality tolerance must be finite and positive")
        self.orthogonality_error = float(np.max(np.abs(values.T @ values - np.eye(values.shape[1]))))
        if not np.isfinite(self.orthogonality_error) or self.orthogonality_error > orthogonality_tolerance:
            raise ValueError("POD basis columns must be orthonormal")
        self.shape = tuple(physics.shape)
        self.n, self.reduced_dimension = values.shape
        self._physics_config = {
            "shape": list(self.shape), "spacing": float(physics.spacing),
            "damping_width": int(physics.damping_width), "damping_strength": float(physics.damping_strength),
        }
        if not np.isfinite(physics.spacing) or physics.spacing <= 0 or not np.isfinite(physics.damping_strength) or physics.damping_strength < 0:
            raise ValueError("reference spacing/damping configuration must be valid and finite")
        reference = Helmholtz2D(**{**self._physics_config, "shape": self.shape})
        self._basis = values
        self._stiffness = np.asarray(values.T @ (reference.stiffness @ values), dtype=float)
        self._damping = np.asarray(values.T @ (reference.damping[:, None] * values), dtype=float)
        for array in (self._basis, self._stiffness, self._damping):
            array.setflags(write=False)
        self.factorization_cache_size = factorization_cache_size
        self.mass_cache_size = mass_cache_size
        self._factors, self._masses = OrderedDict(), OrderedDict()
        self._cost = {}
        self._validate_cache_sizes()
        self.offline_projection_cost = {
            "stiffness_projections": 1, "damping_projections": 1,
            "full_dimension": self.n, "reduced_dimension": self.reduced_dimension,
        }

    @property
    def physics_config(self) -> dict:
        # Metadata must describe the coefficients actually projected above.
        # Returning a copy also protects the nested mutable shape list.
        return {**self._physics_config, "shape": list(self._physics_config["shape"])}

    @property
    def basis(self) -> np.ndarray:
        view = self._basis.view()
        view.setflags(write=False)
        return view

    def _count(self, key: str, amount: int = 1) -> None:
        self._cost[key] = self._cost.get(key, 0) + amount

    def _validate_cache_sizes(self) -> None:
        for name in ("factorization_cache_size", "mass_cache_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for cache, limit in ((self._factors, self.factorization_cache_size), (self._masses, self.mass_cache_size)):
            while len(cache) > limit:
                cache.popitem(last=False)

    def clear_solver_cache(self) -> None:
        self._factors.clear()
        self._masses.clear()

    def reset_cost_counters(self, *, clear_cache: bool = False) -> None:
        self._cost.clear()
        if clear_cache:
            self.clear_solver_cache()

    def cost_snapshot(self) -> dict:
        return {"full_dimension": self.n, "reduced_dimension": self.reduced_dimension,
                **{key: int(self._cost.get(key, 0)) for key in self._COST_KEYS}}

    def _model(self, m) -> np.ndarray:
        if np.iscomplexobj(m):
            raise ValueError("squared slowness must be real")
        model = np.asarray(m, dtype=np.float64).reshape(-1)
        if model.size != self.n or not np.all(np.isfinite(model)) or np.any(model <= 0):
            raise ValueError("squared slowness must be finite, positive and match the grid")
        return np.ascontiguousarray(model)

    def _vector(self, values, *, real: bool, name: str) -> np.ndarray:
        if real and np.iscomplexobj(values):
            raise ValueError(f"{name} must be real")
        array = np.asarray(values, dtype=np.float64 if real else np.complex128).reshape(-1)
        if array.size != self.n or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite and match the grid")
        return array

    def _receivers(self, receiver_indices) -> np.ndarray:
        raw = np.asarray(tuple(receiver_indices))
        if raw.ndim == 1 and raw.size == 0:
            return np.empty(0, dtype=int)
        if raw.ndim != 1 or raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(raw >= self.n):
            raise ValueError("receiver indices must be valid integer grid indices")
        return raw.astype(int, copy=False)

    def _frequency(self, value) -> tuple[float, float]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError("frequency must be finite and positive")
        omega = 2 * np.pi * float(value)
        if not np.isfinite(omega) or omega > np.sqrt(np.finfo(float).max):
            raise ValueError("frequency would overflow the reduced operator")
        return omega, omega**2

    def _mass(self, model: np.ndarray, key: bytes) -> np.ndarray:
        if key in self._masses:
            self._count("mass_cache_hits")
            self._masses.move_to_end(key)
            return self._masses[key]
        mass = self._basis.T @ (model[:, None] * self._basis)
        mass.setflags(write=False)
        self._count("mass_assemblies")
        if self.mass_cache_size:
            self._masses[key] = mass
            while len(self._masses) > self.mass_cache_size:
                self._masses.popitem(last=False)
        return mass

    def _prepare(self, m, frequency_hz):
        self._validate_cache_sizes()
        model = self._model(m)
        omega, omega2 = self._frequency(frequency_hz)
        content = model.tobytes()
        key = (content, float(frequency_hz))
        if key in self._factors:
            self._count("factor_cache_hits")
            self._factors.move_to_end(key)
            return model, omega2, self._factors[key]
        operator = self._stiffness - omega2 * self._mass(model, content) + 1j * omega * self._damping
        if not np.all(np.isfinite(operator)):
            raise FloatingPointError("reduced operator overflowed")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", la.LinAlgWarning)
                factor = la.lu_factor(operator, check_finite=False)
        except la.LinAlgWarning as exc:
            raise np.linalg.LinAlgError("reduced Galerkin operator is singular") from exc
        self._count("reduced_factorizations")
        if self.factorization_cache_size:
            self._factors[key] = factor
            while len(self._factors) > self.factorization_cache_size:
                self._factors.popitem(last=False)
        return model, omega2, factor

    def _solve(self, factor, rhs, *, category: str, adjoint: bool = False):
        values = np.asarray(rhs, dtype=np.complex128)
        if values.ndim not in (1, 2) or values.shape[0] != self.reduced_dimension or not np.all(np.isfinite(values)):
            raise ValueError("reduced right-hand side has invalid dimensions or nonfinite values")
        result = la.lu_solve(factor, values, trans=2 if adjoint else 0, check_finite=False)
        self._count(category, 1 if values.ndim == 1 else values.shape[1])
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("reduced solve returned nonfinite values")
        return result

    def _state(self, factor, q):
        source = self._vector(q, real=False, name="source")
        self._count("source_projections")
        reduced = self._solve(factor, self._basis.T @ source, category="reduced_forward_rhs")
        self._count("fine_state_lifts")
        return self._basis @ reduced

    def state(self, m, q, frequency_hz):
        _, _, factor = self._prepare(m, frequency_hz)
        return self._state(factor, q)

    def state_and_jvp(self, m, direction, q, frequency_hz):
        _, omega2, factor = self._prepare(m, frequency_hz)
        vector = self._vector(direction, real=True, name="model direction")
        state = self._state(factor, q)
        self._count("tangent_rhs_projections")
        tangent = self._solve(factor, omega2 * (self._basis.T @ (vector * state)), category="reduced_tangent_rhs")
        self._count("fine_tangent_lifts")
        return state, self._basis @ tangent

    def jvp(self, m, direction, q, frequency_hz):
        return self.state_and_jvp(m, direction, q, frequency_hz)[1]

    def _data_adjoint(self, factor, state, data_dual, receivers, omega2):
        dual = np.asarray(data_dual, dtype=np.complex128).reshape(-1)
        if len(dual) != len(receivers) or not np.all(np.isfinite(dual)):
            raise ValueError("receiver dual must be finite and match receiver count")
        injection = np.zeros(self.n, dtype=np.complex128)
        np.add.at(injection, receivers, dual)
        self._count("adjoint_rhs_projections")
        adjoint = self._solve(factor, self._basis.T @ injection, category="reduced_adjoint_rhs", adjoint=True)
        self._count("fine_adjoint_lifts")
        return omega2 * np.real(np.conjugate(self._basis @ adjoint) * state)

    def data_vjp(self, m, q, receiver_indices, frequency_hz, data_dual):
        _, omega2, factor = self._prepare(m, frequency_hz)
        receivers = self._receivers(receiver_indices)
        state = self._state(factor, q)
        return self._data_adjoint(factor, state, data_dual, receivers, omega2)

    def objective_and_gradient(self, m, q, observed, receiver_indices, frequency_hz):
        _, omega2, factor = self._prepare(m, frequency_hz)
        receivers = self._receivers(receiver_indices)
        observations = np.asarray(observed, dtype=np.complex128).reshape(-1)
        if len(observations) != len(receivers) or not np.all(np.isfinite(observations)):
            raise ValueError("observations must be finite and match receiver count")
        state = self._state(factor, q)
        residual = state[receivers] - observations
        objective = .5 * float(np.vdot(residual, residual).real)
        gradient = self._data_adjoint(factor, state, residual, receivers, omega2)
        return objective, gradient, state, residual

    def objective(self, m, q, observed, receiver_indices, frequency_hz):
        receivers = self._receivers(receiver_indices)
        observations = np.asarray(observed, dtype=np.complex128).reshape(-1)
        if len(observations) != len(receivers) or not np.all(np.isfinite(observations)):
            raise ValueError("observations must be finite and match receiver count")
        residual = self.state(m, q, frequency_hz)[receivers] - observations
        return .5 * float(np.vdot(residual, residual).real)

    def save(self, path: str | Path, *, metadata: dict | None = None):
        stream = io.BytesIO()
        np.savez_compressed(stream, basis=self._basis,
                            physics_config=np.asarray(json.dumps(self.physics_config, sort_keys=True)),
                            metadata=np.asarray(json.dumps(metadata or {}, sort_keys=True, allow_nan=False)))
        Path(path).write_bytes(stream.getvalue())

    @classmethod
    def load(cls, path: str | Path, *, rank: int | None = None, **kwargs):
        with np.load(path, allow_pickle=False) as archive:
            basis = archive["basis"].copy()
            config = json.loads(str(archive["physics_config"]))
        if rank is not None:
            if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= basis.shape[1]:
                raise ValueError("requested rank must be a positive prefix of the saved basis")
            basis = basis[:, :rank]
        config["shape"] = tuple(config["shape"])
        return cls(Helmholtz2D(**config), basis, **kwargs)
